"""Q2: utility-DP (terminal = CDF of the group's final best) vs target-DP, paired best-of-K contests.

Every run all strategies see the same fresh rewards as the EV-DP clone group. Before run k (m = K - k
runs left including this one), with group best X and our best Yb, the utility-DP maximises
    U(S) = P(max(Yb, S) > max(X, M_m)),  M_m = max of m independent EV-DP scores,
i.e. U(S) = 1[max(Yb,S) > X] * G(max(Yb,S))^(c*m) with G the EV-DP score CDF (c < 1 discounts the
current run, whose group score is correlated with ours). The table is rebuilt every run; contests are
grouped by (Z = max(X, Yb) rounded to ZQ, lead, m).
`R1_C=10000 python research/r1_contest.py [K,K,...] [c,c,...]`
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from r1_lib import T, MU, play, play_ev, value_tables
from target_dp import target_tables

B0 = 2800.0
SMAX = 3700.0
ZQ = 10.0
H = 1.0
C = int(os.environ.get("R1_C", 10000))  # env so spawned workers see it
FIXED = (2950.0, 3000.0, 3050.0)
DELTAS = (60.0, 100.0, 150.0)

_x = np.sort(np.load("_local/r1/pool.npz")["x"])
GRID_S = SMAX - np.arange(int(SMAX / H) + 1) * H  # score at d-index i


def G(s):
    return np.searchsorted(_x, s, side="right") / len(_x)


GS = G(GRID_S)


def utility_F0(Z, lead, m_eff):
    s = GRID_S
    if lead:
        return G(np.maximum(s, Z)) ** m_eff
    return np.where(s > Z, GS ** m_eff, 0.0)


def rewards(seed, rows=None):
    rw = np.random.default_rng(seed).exponential(MU, (C, T))
    return rw if rows is None else rw[rows]


_W = None


def job_target(args):
    seed, i, goals = args
    global _W
    if _W is None:
        _W = target_tables(SMAX + 50)
    rw = rewards(seed, slice(i, None, 16))
    return play(rw, goals, _W, H)


def job_utility(args):
    seed, rows, Z, lead, m_eff = args
    W = value_tables(utility_F0(Z, lead, m_eff), h=H)
    return rows, play(rewards(seed, rows), SMAX, W, H, tie_ev=True)


def main(K, cs, pool, rng):
    X = np.full(C, B0)
    names = [f"fixed {b:.0f}" for b in FIXED] + [f"X+{d:.0f}" for d in DELTAS] + [f"utility c={c}" for c in cs]
    Yb = {n: np.zeros(C) for n in names}
    t0 = time.time()
    for k in range(K):
        m = K - k
        seed = int(rng.integers(1 << 62))
        # target-DP strategies: one vectorised call, goals (C, nstrat)
        goals = np.stack([np.full(C, b) for b in FIXED] + [X + d for d in DELTAS], 1)
        jobs = [(seed, i, goals[i::16]) for i in range(16)]
        tasks = []
        for c in cs:
            n = f"utility c={c}"
            Z = np.rint(np.maximum(X, Yb[n]) / ZQ) * ZQ
            lead = Yb[n] > X
            keys = {}
            for i, key in enumerate(zip(Z, lead)):
                keys.setdefault(key, []).append(i)
            tasks += [(c, (seed, np.array(rows), z, ld, c * m)) for (z, ld), rows in keys.items()]
        res_t = pool.map_async(job_target, jobs)
        res_u = pool.map_async(job_utility, [a for _, a in tasks], chunksize=1)
        x = play_ev(rewards(seed))
        yt = np.empty((C, goals.shape[1]))
        for i, part in enumerate(res_t.get()):
            yt[i::16] = part
        for j, n in enumerate(names[:goals.shape[1]]):
            Yb[n] = np.maximum(Yb[n], yt[:, j])
        yu = {c: np.zeros(C) for c in cs}
        for (c, _), (rows, sc) in zip(tasks, res_u.get()):
            yu[c][rows] = sc
        for c in cs:
            n = f"utility c={c}"
            Yb[n] = np.maximum(Yb[n], yu[c])
        X = np.maximum(X, x)
        print(f"  K={K} run {k+1}/{K}: {len(tasks)} utility tables, {time.time()-t0:.0f}s", file=sys.stderr, flush=True)
    print(f"\n### K = {K}, C = {C} paired contests, group best final mean {X.mean():.0f}")
    for n in names:
        w, l = (Yb[n] > X).mean(), (Yb[n] < X).mean()
        se = np.sqrt(w * (1 - w) / C)
        print(f"{n:16s} win {w:.4f} (±{se:.4f})  lose {l:.4f}  tie {1-w-l:.4f}")
    base = Yb[names[-1]] > X
    for n in names[:-1]:
        o = Yb[n] > X
        print(f"   paired diff utility c={cs[-1]} - {n}: {(base.astype(int) - o).mean():+.4f} "
              f"(±{(base.astype(int) - o).std() / np.sqrt(C):.4f})")
    sys.stdout.flush()


if __name__ == "__main__":
    KS = [int(k) for k in (sys.argv[1] if len(sys.argv) > 1 else "10,20,40,60").split(",")]
    CS = [float(c) for c in (sys.argv[2] if len(sys.argv) > 2 else "0.5,1.0").split(",")]
    rng = np.random.default_rng(2024)
    with Pool(16) as pool:
        for K in KS:
            main(K, CS, pool, rng)
