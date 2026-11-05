"""Round 2 best-of-K contest: P(our best strictly beats the EV-DP field's best) on shared runs.

Every run all teams see the same rewards; each run may carry its own location shift (ETH round 2:
mean varied by run with sd ~0.1). The field plays K0 runs before we join, then K more, and between
runs we resubmit with target B = (field best so far) + delta.

Field (EV-DP clones and their usual variants):
    oracle   EV-DP that knows this run's distribution exactly
    emp k    online empirical EV-DP (solution.py MODE="empirical": warmup 8, refit every 100) with
             thresholds x k, k in KAPPAS (k = 1.05 was the ETH round-2 winner)
Contenders (target DP, fall back to their own EV-DP when the target is reached or hopeless):
    O  oracle target-DP (knows the run's distribution: upper bound)
    P  prior shape from earlier runs (the mined distribution) + this run's shift learned online,
       shift_hat = sum(r - mean0) / (t + N0), table switched at CHECK_P
    E  no prior: empirical EV-DP until E_FITS[0], then a target table from a KDE of this run's
       rewards, refitted at each E_FITS (the cost of learning the distribution online)
Baselines (no target, so independent of delta):
    X1, X1.03  EV-DP on the known base law (shift 0), thresholds x 1 / x 1.03 (e.g. exact N(-0.2,1))
    Pev        EV-DP on the prior shape + this run's shift learned online (candidate, TARGET=None)

`uv run python research/r2_contest.py <dist> [contests] [K]`   (dist in target_dp_general.DISTS)
"""
import math
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from target_dp_general import (DISTS, L, T, _ndtri, ev_gap_sorted, grid_step, pmf_on_grid,  # noqa: E402
                               simulate, target_table)

K0 = 5
KAPPAS = (0.95, 1.0, 1.05, 1.1)
EMP_WARMUP, EMP_REFIT, EMP_GRID = 8, 100, 256
CHECK_P = (1, 50, 150, 300, 600, 1000)
E_FITS = (200, 700)
N0 = 100.0
DELTAS = (0.25, 0.5, 1.0)       # in units of the within-run EV-DP score sd
KDE_Z = _ndtri((np.arange(16) + 0.5) / 16)

G = {}


def setup(name):
    dist = DISTS[name]
    q0 = dist.quantiles(20000)
    sd0 = float(q0.std())
    g = np.random.default_rng(12345)
    gap0 = ev_gap_sorted(q0)
    pil = simulate(dist.sample(g, (3000, T)), lambda n, s, r: r >= gap0[n])
    sig = float(pil.std())
    step = 0.01 * sd0
    m = int(math.ceil(max(4 * dist.run_sd, 0.3 * sd0) / step))
    shifts = np.arange(-m, m + 1) * step
    sh = g.normal(0, dist.run_sd, 3000)
    pil2 = simulate(dist.sample(g, (3000, T)) + sh[:, None], lambda n, s, r: r >= gap0[n])
    B_max = float(pil2.max()) + 4 * sig
    h = grid_step(q0, B_max)
    D = int(B_max / h) + 2
    t0 = time.time()
    Wall = np.empty((len(shifts), T + 1, D + 1), np.float32)
    gaps = np.empty((len(shifts), T + 1))
    for k, s in enumerate(shifts):
        Wall[k] = target_table(pmf_on_grid(q0 + s, h), D)
        gaps[k] = ev_gap_sorted(q0 + s)
    G.update(dist=dist, q0=q0, mean0=float(q0.mean()), sd0=sd0, sig=sig, mu_ev=float(pil.mean()),
             shifts=shifts, step=step, h=h, D=D, Wall=Wall, gaps=gaps)
    print(f"[{name}] EV-DP mean {pil.mean():.2f} sd {sig:.2f} (with run shifts: sd {pil2.std():.2f})  "
          f"h={h:.4f} D={D}  {len(shifts)} shift tables in {time.time() - t0:.0f}s", flush=True)


def shift_index(s):
    return np.clip(np.rint(s / G["step"]).astype(np.int64) + len(G["shifts"]) // 2, 0, len(G["shifts"]) - 1)


def emp_gaps(x):
    """Empirical EV-DP thresholds for each row's samples x (R, t): gap (N + 1, R), N = T - t + 1."""
    R, t = x.shape
    srt = np.sort(x, axis=1)
    if t > EMP_GRID:
        srt = srt[:, ((np.arange(EMP_GRID) + 0.5) * t / EMP_GRID).astype(int)]
    N = T - t + 1
    V = np.zeros((N + 1, R))
    gap = np.zeros((N + 1, R))
    for n in range(1, N + 1):
        skip = V[n - 1]
        acc = V[n - 1 - L] if n - 1 - L > 0 else np.zeros(R)
        gap[n] = np.maximum(skip - acc, 0.0)
        V[n] = np.maximum(srt + acc[:, None], skip[:, None]).mean(1)
    return gap


def emp_threshold_path(rw):
    """thr (R, T + 1): online empirical EV-DP threshold used at each round t (inf during warmup)."""
    R = rw.shape[0]
    thr = np.full((R, T + 1), np.inf)
    gap = None
    for t in range(1, T + 1):
        if t <= EMP_WARMUP:
            continue
        if gap is None or (t - EMP_WARMUP - 1) % EMP_REFIT == 0:
            gap, t_fit = emp_gaps(rw[:, :t]), t
        thr[:, t] = gap[T - t + 1]
    return thr


LEA_Z = _ndtri((np.arange(200) + 0.5) / 200)


def lea_threshold_path(rw, H=600):
    """Lea's solution_round2.py: EV-DP on every reward seen plus 200 N(-0.2, 1) pseudo-samples whose
    weight fades linearly to 0 at 300 real rewards; the prior is replaced by a normal fitted to the
    data once the mean is > 4 se from -0.2 or sd / 1 leaves [0.7, 1.4] (after 20 rewards). Refit every
    round for t <= 20, every 5 to 300, then every 10; horizon H rounds (flat cut beyond)."""
    R = rw.shape[0]
    thr = np.zeros((R, T + 1))
    switched = np.zeros(R, bool)
    gap = None
    for t in range(1, T + 1):
        if gap is None or t <= 20 or (t <= 300 and t % 5 == 0) or t % 10 == 0:
            x = rw[:, :t]
            m, s = x.mean(1), x.std(1)
            if t >= 20:
                switched |= (np.abs(m + 0.2) > 4 / np.sqrt(t)) | (s < 0.7) | (s > 1.4)
            pm = np.where(switched, m, -0.2)
            ps = np.where(switched, np.maximum(s, 1e-6), 1.0)
            prior = pm[:, None] + ps[:, None] * LEA_Z[None, :]
            srt = np.sort(x, axis=1)
            if t > EMP_GRID:
                srt = srt[:, ((np.arange(EMP_GRID) + 0.5) * t / EMP_GRID).astype(int)]
            wr = t / srt.shape[1]                    # weight of each real grid point
            wp = max(0.0, 1.0 - t / 300.0)           # weight of each pseudo-sample
            grid = np.concatenate((srt, prior), axis=1)
            w = np.concatenate((np.full(srt.shape[1], wr), np.full(200, wp)))
            w = w / w.sum()
            N = min(H, T - t + 1)
            V = np.zeros((N + 1, R))
            gap = np.zeros((N + 1, R))
            for n in range(1, N + 1):
                skip = V[n - 1]
                acc = V[n - 1 - L] if n - 1 - L > 0 else np.zeros(R)
                gap[n] = np.maximum(skip - acc, 0.0)
                V[n] = np.maximum(grid + acc[:, None], skip[:, None]) @ w
            fit_t = t
        n = T - t + 1
        thr[:, t] = gap[min(n, len(gap) - 1)]
    return thr


def kde_pmf(x, h):
    bw = 1.06 * x.std() * len(x) ** -0.2
    return pmf_on_grid((x[:, None] + bw * KDE_Z[None, :]).ravel(), h)


def decide(W3, ridx, n, d, r, ev_acc, h):
    D = W3.shape[2] - 1
    d = d + 0.5 * h
    i = np.clip(np.rint(d / h).astype(np.int64), 0, D)
    Sv = W3[ridx, n - 1, i]
    ia = np.rint((d - r) / h).astype(np.int64)
    Av = np.where(ia <= 0, 1.0, W3[ridx, max(n - 1 - L, 0), np.clip(ia, 0, D)])
    ok = (i > 0) & (np.maximum(Av, Sv) > 1e-9)
    return np.where(ok, (Av > Sv + 1e-12) & (r > 0), ev_acc)


def contest(args):
    seed, K = args
    g = np.random.default_rng(seed)
    dist, h, Wall, gaps = G["dist"], G["h"], G["Wall"], G["gaps"]
    nD = len(DELTAS)
    R = K0 + K
    sh = g.normal(0.0, dist.run_sd, R)
    rw = dist.sample(g, (R, T)) + sh[:, None]
    sidx = shift_index(sh)

    # ---- field
    thr = emp_threshold_path(rw)
    field = {"oracle": simulate(rw, lambda n, s, r: r >= gaps[sidx, n])}
    for k in KAPPAS:
        tk = thr * k
        field[f"emp{k}"] = simulate(rw, lambda n, s, r, tk=tk: r >= tk[:, T - n + 1])
    fs = np.max(np.stack(list(field.values())), axis=0)          # field best per run
    best_before = np.maximum.accumulate(fs)[K0 - 1:R - 1]        # field best before each of our runs
    B = (best_before[:, None] + np.array(DELTAS)[None, :] * G["sig"]).ravel()   # rows: run k, delta j

    # ---- contenders, rows (K * nD)
    run = np.repeat(np.arange(K0, R), nD)
    x = rw[run]
    out = {}
    # O: oracle target
    ridx = sidx[run]
    out["O"] = simulate(x, lambda n, s, r: decide(Wall, ridx, n, B - s, r, r >= gaps[ridx, n], h))
    # P: prior shape + online shift
    csum = np.cumsum(rw[K0:] - G["mean0"], axis=1)
    state = {"p": shift_index(np.zeros(K))[np.repeat(np.arange(K), nD)]}

    def pol_p(n, s, r):
        t = T - n + 1
        if t in CHECK_P:
            est = csum[:, t - 1] / (t + N0)
            state["p"] = shift_index(est)[np.repeat(np.arange(K), nD)]
        p = state["p"]
        return decide(Wall, p, n, B - s, r, r >= gaps[p, n], h)
    out["P"] = simulate(x, pol_p)
    # E: no prior, KDE target tables refitted at E_FITS
    thr_c = thr[run]
    st = {"W": None}

    def pol_e(n, s, r):
        t = T - n + 1
        ev = r >= thr_c[:, t]
        if t in E_FITS:
            N = T - t + 1
            Wk = np.stack([target_table(kde_pmf(rw[K0 + k, :t], h), G["D"], T=N) for k in range(K)])
            st["W"], st["ridx"] = Wk, np.repeat(np.arange(K), nD)
        if st["W"] is None:
            return ev
        return decide(st["W"], st["ridx"], n, B - s, r, ev, h)
    out["E"] = simulate(x, pol_e)

    # baselines, rows (K)
    xk = rw[K0:]
    c0 = len(G["shifts"]) // 2
    base = {f"X{k}": simulate(xk, lambda n, s, r, k=k: r >= k * gaps[c0, n]) for k in (1.0, 1.03)}
    stp = {"p": np.full(K, c0)}

    def pol_pev(n, s, r):
        t = T - n + 1
        if t in CHECK_P:
            stp["p"] = shift_index(csum[:, t - 1] / (t + N0))
        return r >= gaps[stp["p"], n]
    base["Pev"] = simulate(xk, pol_pev)
    lthr = lea_threshold_path(xk)
    base["Lea"] = simulate(xk, lambda n, s, r: (r >= lthr[:, T - n + 1]) & (r > 0))

    res = {}
    f_final = fs.max()
    o_final = field["oracle"].max()
    for c, sc in out.items():
        best = sc.reshape(K, nD).max(0)
        res[c] = (best > f_final, best > o_final, sc.reshape(K, nD).mean(0))
    for c, sc in base.items():
        res[c] = (np.array([sc.max() > f_final] * nD), np.array([sc.max() > o_final] * nD),
                  np.array([sc.mean()] * nD))
    k105 = field["emp1.05"][K0:].max()
    res["emp1.05"] = (None, np.array([k105 > o_final] * nD), None)
    return res, f_final, field["oracle"][K0:].mean()


def main():
    name = sys.argv[1]
    C = int(sys.argv[2]) if len(sys.argv) > 2 else 320
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    setup(name)
    t0 = time.time()
    with get_context("fork").Pool(min(16, os.cpu_count() or 1)) as pool:
        rs = pool.map(contest, [(1000 + i, K) for i in range(C)], chunksize=1)
    print(f"[{name}] C={C} K={K} K0={K0}  ({time.time() - t0:.0f}s)  field final best mean "
          f"{np.mean([r[1] for r in rs]):.2f}, oracle EV run mean {np.mean([r[2] for r in rs]):.2f}")
    se = lambda p: math.sqrt(max(p * (1 - p), 1e-9) / C)
    for c in ("O", "P", "E", "X1.0", "X1.03", "Pev", "Lea"):
        wf = np.mean([r[0][c][0] for r in rs], axis=0)
        wo = np.mean([r[0][c][1] for r in rs], axis=0)
        mn = np.mean([r[0][c][2] for r in rs], axis=0)
        for j, dl in enumerate(DELTAS):
            print(f"RESULT {name} {c} delta={dl:.2f}sd ({dl * G['sig']:.2f}): P(beat field)={wf[j]:.3f}"
                  f"+-{se(wf[j]):.3f}  P(beat oracle-EV clones)={wo[j]:.3f}  mean run score {mn[j]:.2f}")
    wk = np.mean([r[0]["emp1.05"][1][0] for r in rs])
    print(f"RESULT {name} emp1.05 (ETH winner style): P(beat oracle-EV clones)={wk:.3f}+-{se(wk):.3f}")


if __name__ == "__main__":
    main()
