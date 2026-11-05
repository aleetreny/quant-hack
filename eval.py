"""Local evaluator for Round 1 (Exponential mean 10, lockout 15, T=2000).

Usage:
    uv run python eval.py path/to/solution.py [--games 4000] [--seed 0] [--target 2750]

Each game: a fresh copy of the solution source is exec'd (state does not leak between games),
rewards ~ Exp(mean 10), 23 opponents playing the expected-value-optimal DP on the same rewards
(their scores are passed in other_scores). Common random numbers: game g uses seed (seed, g),
so two solutions evaluated with the same --seed see identical reward sequences.

Leaderboard ranks on BEST score over runs, so the key metrics are the upper tail:
    P>=target, and E[max of n] for n runs (from the empirical distribution).
"""
import argparse
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

T, LOCK, MEAN, N_OPP = 2000, 15, 10.0, 23


def _dp_cutoffs():
    G = LOCK + 1
    V = [0.0] * (T + 1)
    c = [0.0] * (T + 1)
    for s in range(1, T + 1):
        c[s] = V[s - 1] - (V[s - G] if s >= G else 0.0)
        V[s] = V[s - 1] + MEAN * math.exp(-c[s] / MEAN)
    return c


CUT = _dp_cutoffs()


def _opp_scores(rewards, rng):
    """Score trajectory (after each round) and lock flags for opponents: EV-DP with jittered cutoffs."""
    out_s = np.zeros((N_OPP, T))
    out_l = np.zeros((N_OPP, T), dtype=bool)
    for k in range(N_OPP):
        jitter = 1.0 + (0 if k < 8 else rng.normal(0, 0.05))
        sc, lu = 0.0, 0
        for t in range(1, T + 1):
            locked = t <= lu
            if not locked and rewards[t - 1] >= CUT[T - t + 1] * jitter:
                sc += rewards[t - 1]
                lu = t + LOCK
                locked = True
            out_s[k, t - 1] = sc
            out_l[k, t - 1] = t <= lu and t < lu + 1 and locked
    return out_s, out_l


def _run_game(args):
    src, path, seed, g = args
    rng = np.random.default_rng([seed, g])
    rewards = rng.exponential(MEAN, T)
    opp_s, opp_l = _opp_scores(rewards, rng)
    ns = {"__name__": "solution_eval", "__file__": path}
    exec(compile(src, path, "exec"), ns)
    play = ns["play"]
    score, lu = 0.0, 0
    t0 = time.perf_counter()
    for t in range(1, T + 1):
        locked = t <= lu
        others = [] if t == 1 else [(float(opp_s[k, t - 2]), bool(opp_l[k, t - 2])) for k in range(N_OPP)]
        r = float(rewards[t - 1])
        a = play(r, LOCK, t, T, float(score), others, locked)
        if a and not locked:
            score += r
            lu = t + LOCK
    return score, time.perf_counter() - t0, float(opp_s[0, -1])


def e_max(scores, n):
    """E[max of n iid draws] from the empirical distribution."""
    s = np.sort(scores)
    m = len(s)
    i = np.arange(1, m + 1)
    w = (i / m) ** n - ((i - 1) / m) ** n
    return float((w * s).sum())


def evaluate(path, games=4000, seed=0, procs=None):
    src = open(path).read()
    with Pool(procs or max(1, (os.cpu_count() or 2) - 2)) as p:
        res = p.map(_run_game, [(src, path, seed, g) for g in range(games)], chunksize=8)
    sc = np.array([r[0] for r in res])
    tm = np.array([r[1] for r in res])
    evaluate.ev = np.array([r[2] for r in res])
    return sc, tm


def win_best_of(sc, ev, n, reps=20000, seed=1):
    """Paired: P(our best of n runs > EV-DP team's best of the same n runs), ties count half."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sc), size=(reps, n))
    a, b = sc[idx].max(1), ev[idx].max(1)
    return float((a > b + 1e-9).mean() + 0.5 * (np.abs(a - b) <= 1e-9).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", type=float, default=2750.0)
    ap.add_argument("--procs", type=int, default=None)
    a = ap.parse_args()
    sc, tm = evaluate(a.path, a.games, a.seed, a.procs)
    se = sc.std() / math.sqrt(len(sc))
    p = (sc >= a.target).mean()
    print(f"file={a.path} games={len(sc)} seed={a.seed}")
    print(f"mean={sc.mean():.2f} (se {se:.2f})  std={sc.std():.1f}  median={np.median(sc):.1f}")
    print(f"P>={a.target:.0f}={p:.4f} (se {math.sqrt(p*(1-p)/len(sc)):.4f})  P>=2700={(sc>=2700).mean():.4f}  P>=2800={(sc>=2800).mean():.4f}")
    print("E[max of n]: " + "  ".join(f"n={n}:{e_max(sc, n):.1f}" for n in (3, 5, 10, 20)))
    ev = evaluate.ev
    print("P(our best-of-n > EV-DP best-of-n, same rewards): " + "  ".join(f"n={n}:{win_best_of(sc, ev, n):.3f}" for n in (1, 5, 10, 20)))
    print(f"max game time={tm.max():.2f}s  mean={tm.mean():.3f}s")


if __name__ == "__main__":
    main()
