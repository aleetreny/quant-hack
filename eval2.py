"""Local evaluator for Round 2: reward distribution UNKNOWN.

Usage:
    uv run python eval2.py strat_a.py [strat_b.py ...] [--games 200] [--seed 0] [--dists all]
                           [--procs 8] [--lockout 15] [--csv out.csv]

For each distribution in a suite of plausible "unknown" distributions (light/heavy tails, negative
values, discrete, bimodal, non-stationary), each game draws T=2000 rewards (common random numbers:
game g of dist d uses seed (seed, d_index, g), so all strategies see identical reward sequences).

The oracle knows the true per-round marginal distribution and plays the exact optimal-stopping DP
(for non-stationary dists it knows the realised mean path). Main metric per game:
    ratio = strategy_score / oracle_score     (1.0 = as good as knowing the distribution)
Reported with 95% CIs per distribution and aggregated (equal weight per distribution).
With several strategies, paired differences vs the FIRST strategy are reported (same games),
which gives much tighter CIs than comparing the unpaired means.

Strategy files must define play(reward, lockout, t, T, your_score, other_scores, is_locked).
A fresh exec per game, so module-level state does not leak between games.
"""
import argparse
import bisect
import csv
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

T = 2000
N_OPP = 23
REF_N = 200_000


# --------------------------------------------------------------------------- distributions
# Each entry: name -> (kind, spec)
#   kind "iid":  spec(rng, n) -> samples   (stationary; oracle from a big reference sample)
#   kind "path": spec(rng, T) -> (rewards, means)  exponential with per-round mean path (oracle analytic)

def _iid(f):
    return ("iid", f)


def _regime_path(rng, T):
    means = np.empty(T)
    t = 0
    m = rng.choice([4.0, 10.0, 25.0])
    while t < T:
        L = int(rng.integers(150, 500))
        means[t:t + L] = m
        m = rng.choice([4.0, 10.0, 25.0])
        t += L
    return rng.exponential(means), means


def _drift_path(rng, T):
    a, b = (5.0, 20.0) if rng.random() < 0.5 else (20.0, 5.0)
    means = np.linspace(a, b, T)
    return rng.exponential(means), means


DISTS = {
    "exp10":      _iid(lambda r, n: r.exponential(10.0, n)),
    "exp2":       _iid(lambda r, n: r.exponential(2.0, n)),
    "exp100":     _iid(lambda r, n: r.exponential(100.0, n)),
    "unif0_20":   _iid(lambda r, n: r.uniform(0, 20, n)),
    "unif50_60":  _iid(lambda r, n: r.uniform(50, 60, n)),
    "normal10_3": _iid(lambda r, n: r.normal(10, 3, n)),
    "normal0_10": _iid(lambda r, n: r.normal(0, 10, n)),           # half the rewards negative
    "gamma5":     _iid(lambda r, n: r.gamma(5.0, 2.0, n)),
    "lognorm":    _iid(lambda r, n: r.lognormal(1.5, 1.0, n)),       # heavy-ish tail
    "lognorm_hv": _iid(lambda r, n: r.lognormal(1.0, 1.6, n)),       # heavy tail
    "pareto2.5":  _iid(lambda r, n: 5.0 * (r.pareto(2.5, n) + 1.0)),
    "pareto1.5":  _iid(lambda r, n: 2.0 * (r.pareto(1.5, n) + 1.0)), # infinite variance
    "bimodal":    _iid(lambda r, n: np.where(r.random(n) < 0.9, r.exponential(5.0, n), r.normal(50, 5, n))),
    "rare_spike": _iid(lambda r, n: np.where(r.random(n) < 0.01, r.uniform(200, 400, n), r.uniform(0, 10, n))),
    "discrete":   _iid(lambda r, n: r.choice([1.0, 5.0, 10.0, 50.0], size=n, p=[0.5, 0.3, 0.15, 0.05])),
    "beta_skew":  _iid(lambda r, n: 100 * r.beta(0.5, 5.0, n)),
    "drift":      ("path", _drift_path),
    "regime":     ("path", _regime_path),
}


# --------------------------------------------------------------------------- oracle
class _Excess:
    """E[(X - c)^+] for an empirical sample."""

    def __init__(self, x):
        self.x = np.sort(np.asarray(x, dtype=float)).tolist()
        n = len(self.x)
        suf = [0.0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suf[i] = suf[i + 1] + self.x[i]
        self.suf, self.n = suf, n

    def __call__(self, c):
        i = bisect.bisect_right(self.x, c)
        return (self.suf[i] - c * (self.n - i)) / self.n


def oracle_cutoffs_iid(excess, G):
    """c[s] for s turns left: accept iff reward >= c[s]."""
    V = [0.0] * (T + 1)
    c = [0.0] * (T + 1)
    for s in range(1, T + 1):
        c[s] = V[s - 1] - (V[s - G] if s >= G else 0.0)
        V[s] = V[s - 1] + excess(c[s])
    return c


def oracle_cutoffs_path(means, G):
    """c_t indexed by round t (1..T) for exponential rewards with known per-round means."""
    W = [0.0] * (T + G + 2)
    c = [0.0] * (T + 2)
    for t in range(T, 0, -1):
        ct = W[t + 1] - W[t + G]
        m = means[t - 1]
        c[t] = ct
        W[t] = W[t + 1] + m * math.exp(-max(ct, 0.0) / m)
    return c


_ORACLE_CACHE = {}


def _oracle_iid(dname, G):
    key = (dname, G)
    if key not in _ORACLE_CACHE:
        ref = DISTS[dname][1](np.random.default_rng(12345), REF_N)
        _ORACLE_CACHE[key] = oracle_cutoffs_iid(_Excess(ref), G)
    return _ORACLE_CACHE[key]


def _play_cutoffs(rewards, cut_of_t, LOCK):
    sc, lu = 0.0, 0
    for t in range(1, T + 1):
        if t > lu and rewards[t - 1] >= cut_of_t(t):
            sc += rewards[t - 1]
            lu = t + LOCK
    return sc


def _opponents(rewards, cut_of_t, LOCK, rng):
    """23 opponents: oracle DP with jittered cutoffs. Returns per-round score and lock arrays."""
    jit = np.ones(N_OPP)
    jit[8:] = 1.0 + rng.normal(0, 0.05, N_OPP - 8)
    cuts = np.array([cut_of_t(t) for t in range(1, T + 1)])
    sc = np.zeros(N_OPP)
    lu = np.zeros(N_OPP, dtype=int)
    out_s = np.zeros((T, N_OPP))
    out_l = np.zeros((T, N_OPP), dtype=bool)
    for t in range(1, T + 1):
        free = t > lu
        take = free & (rewards[t - 1] >= cuts[t - 1] * jit)
        sc = sc + np.where(take, rewards[t - 1], 0.0)
        lu = np.where(take, t + LOCK, lu)
        out_s[t - 1] = sc
        out_l[t - 1] = t <= lu
    return out_s, out_l


# --------------------------------------------------------------------------- game runner
def _run_game(args):
    srcs, paths, dname, di, seed, g, LOCK = args
    G = LOCK + 1
    rng = np.random.default_rng([seed, di, g])
    kind, spec = DISTS[dname]
    if kind == "iid":
        rewards = spec(rng, T)
        oc = _oracle_iid(dname, G)
        cut_of_t = lambda t: oc[T - t + 1]
    else:
        rewards, means = spec(rng, T)
        oc = oracle_cutoffs_path(means, G)
        cut_of_t = lambda t: oc[t]
    rewards = np.asarray(rewards, dtype=float)
    oracle = _play_cutoffs(rewards, cut_of_t, LOCK)
    opp_s, opp_l = _opponents(rewards, cut_of_t, LOCK, rng)
    others_by_t = [[]] + [list(zip(opp_s[t].tolist(), opp_l[t].tolist())) for t in range(T - 1)]
    rlist = rewards.tolist()
    out = []
    for src, path in zip(srcs, paths):
        ns = {"__name__": "solution_eval", "__file__": path}
        exec(compile(src, path, "exec"), ns)
        play = ns["play"]
        score, lu = 0.0, 0
        t0 = time.perf_counter()
        err = 0
        for t in range(1, T + 1):
            locked = t <= lu
            r = rlist[t - 1]
            try:
                a = play(r, LOCK, t, T, score, others_by_t[t - 1], locked)
            except Exception:
                a, err = False, err + 1
            if a and not locked:
                score += r
                lu = t + LOCK
        out.append((score, time.perf_counter() - t0, err))
    return dname, g, oracle, out


def _ci(x):
    x = np.asarray(x, dtype=float)
    m = x.mean()
    h = 1.96 * x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else float("nan")
    return m, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dists", default="all", help="comma list or 'all'")
    ap.add_argument("--lockout", type=int, default=15)
    ap.add_argument("--procs", type=int, default=None)
    ap.add_argument("--csv", default=None, help="append per-dist summary rows here")
    ap.add_argument("--quiet", action="store_true", help="only aggregate lines")
    a = ap.parse_args()

    dnames = list(DISTS) if a.dists == "all" else a.dists.split(",")
    srcs = [open(p).read() for p in a.paths]
    names = [os.path.basename(p) for p in a.paths]
    jobs = [(srcs, a.paths, d, list(DISTS).index(d), a.seed, g, a.lockout) for d in dnames for g in range(a.games)]
    t0 = time.time()
    with Pool(a.procs or max(1, (os.cpu_count() or 2) - 2)) as p:
        res = p.map(_run_game, jobs, chunksize=4)
    elapsed = time.time() - t0

    # ratio[d][k] = array over games
    K = len(srcs)
    ratio = {d: [[] for _ in range(K)] for d in dnames}
    raw = {d: [[] for _ in range(K)] for d in dnames}
    orc = {d: [] for d in dnames}
    tmax = [0.0] * K
    errs = [0] * K
    for d, g, o, out in res:
        orc[d].append(o)
        for k, (s, tm, e) in enumerate(out):
            ratio[d][k].append(s / o if o > 0 else (1.0 if s >= o else 0.0))
            raw[d][k].append(s)
            tmax[k] = max(tmax[k], tm)
            errs[k] += e

    print(f"games/dist={a.games} seed={a.seed} lockout={a.lockout} dists={len(dnames)} time={elapsed:.0f}s")
    rows = []
    if not a.quiet:
        hdr = f"{'dist':12s} {'oracle':>9s} " + " ".join(f"{n[:22]:>24s}" for n in names)
        print(hdr)
        for d in dnames:
            cells = []
            for k in range(K):
                m, h = _ci(ratio[d][k])
                cells.append(f"{m:.4f}+-{h:.4f}".rjust(24))
                rows.append([names[k], d, a.games, a.seed, a.lockout, m, h, float(np.mean(raw[d][k])), float(np.mean(orc[d]))])
            print(f"{d:12s} {np.mean(orc[d]):9.1f} " + " ".join(cells))
    else:
        for d in dnames:
            for k in range(K):
                m, h = _ci(ratio[d][k])
                rows.append([names[k], d, a.games, a.seed, a.lockout, m, h, float(np.mean(raw[d][k])), float(np.mean(orc[d]))])

    # aggregate: equal weight per distribution; per-game vector = mean over dists of ratio (games aligned by g)
    G_ = a.games
    agg = [np.mean([np.asarray(ratio[d][k]) for d in dnames], axis=0) for k in range(K)]
    # sort per dist by g to align (map preserves order, so already aligned)
    print("\nAGGREGATE mean ratio (equal weight per dist), 95% CI; worst-dist ratio:")
    for k in range(K):
        m, h = _ci(agg[k])
        worst = min((np.mean(ratio[d][k]), d) for d in dnames)
        print(f"  {names[k]:36s} {m:.4f} +- {h:.4f}   worst={worst[0]:.3f} ({worst[1]})  max_game_time={tmax[k]:.2f}s errors={errs[k]}")
        rows.append([names[k], "AGGREGATE", a.games, a.seed, a.lockout, m, h, float("nan"), float("nan")])
    if K > 1:
        print(f"\nPAIRED diff vs {names[0]} (positive = better), 95% CI:")
        for k in range(1, K):
            m, h = _ci(agg[k] - agg[0])
            sig = "*" if abs(m) > h else " "
            print(f"  {names[k]:36s} {m:+.4f} +- {h:.4f} {sig}")
    if a.csv:
        new = not os.path.exists(a.csv)
        with open(a.csv, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["strategy", "dist", "games", "seed", "lockout", "ratio_mean", "ratio_ci95", "score_mean", "oracle_mean"])
            w.writerows(rows)


if __name__ == "__main__":
    main()
