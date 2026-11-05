"""Round 2: which no-target rule to run against a given field, best of K shared runs.

Contenders: exact-law EV-DP x k for k in KS, Lea's solution_round2.py (r2_contest.lea_threshold_path),
target-DP with prior + online shift (r2_contest "P", delta 0.5 sd). Two fields:
    sim   oracle EV + online empirical EV x {0.95, 1, 1.05, 1.1}  (as in r2_contest.py)
    live  mimics the round-2 run of 19:13 (research/r2_field_kappa.py): exact-law EV x
          {1.0, 1.12, 1.35, 1.45, 1.52} + online empirical EV x {1.0, 1.05, 1.1} + Lea
A contender that is also in the field (k = 1.0, Lea in "live") ties it at best and never wins.
`uv run python research/r2_kappa_contest.py <dist> [contests] [K]`
"""
import math
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2_contest as rc  # noqa: E402
from target_dp_general import T, simulate  # noqa: E402

KS = (0.97, 1.0, 1.01, 1.02, 1.03, 1.05, 1.08, 1.1, 1.15, 1.2, 1.3)
LIVE_EXACT = (1.0, 1.12, 1.35, 1.45, 1.52)
DELTA = 0.5


def contest(args):
    seed, K = args
    G = rc.G
    g = np.random.default_rng(seed)
    dist, h, Wall, gaps = G["dist"], G["h"], G["Wall"], G["gaps"]
    R = rc.K0 + K
    sh = g.normal(0.0, dist.run_sd, R)
    rw = dist.sample(g, (R, T)) + sh[:, None]
    sidx = rc.shift_index(sh)
    c0 = len(G["shifts"]) // 2
    thr = rc.emp_threshold_path(rw)
    lthr = rc.lea_threshold_path(rw)
    sc = {"oracle": simulate(rw, lambda n, s, r: r >= gaps[sidx, n])}
    for k in (0.95, 1.0, 1.05, 1.1):
        sc[f"emp{k}"] = simulate(rw, lambda n, s, r, k=k: r >= k * thr[:, T - n + 1])
    for k in sorted(set(KS) | set(LIVE_EXACT)):
        sc[f"X{k}"] = simulate(rw, lambda n, s, r, k=k: r >= k * gaps[c0, n])
    sc["Lea"] = simulate(rw, lambda n, s, r: (r >= lthr[:, T - n + 1]) & (r > 0))
    fields = {"sim": ["oracle", "emp0.95", "emp1.0", "emp1.05", "emp1.1"],
              "live": [f"X{k}" for k in LIVE_EXACT] + ["emp1.0", "emp1.05", "emp1.1", "Lea"]}
    out = {}
    for fname, members in fields.items():
        fs = np.max(np.stack([sc[m] for m in members]), axis=0)
        final = fs.max()
        best_before = np.maximum.accumulate(fs)[rc.K0 - 1:R - 1]
        B = best_before + DELTA * G["sig"]
        x = rw[rc.K0:]
        csum = np.cumsum(x - G["mean0"], axis=1)
        st = {"p": np.full(K, c0)}

        def pol_p(n, s, r):
            t = T - n + 1
            if t in rc.CHECK_P:
                st["p"] = rc.shift_index(csum[:, t - 1] / (t + rc.N0))
            p = st["p"]
            return rc.decide(Wall, p, n, B - s, r, r >= gaps[p, n], h)
        tgt = simulate(x, pol_p)
        res = {"P(target)": (tgt.max() > final, tgt.mean())}
        for k in KS:
            v = sc[f"X{k}"][rc.K0:]
            res[f"X{k}"] = (v.max() > final, v.mean())
        res["Lea"] = (sc["Lea"][rc.K0:].max() > final, sc["Lea"][rc.K0:].mean())
        out[fname] = res
    return out


def main():
    name = sys.argv[1]
    C = int(sys.argv[2]) if len(sys.argv) > 2 else 320
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    rc.setup(name)
    t0 = time.time()
    with get_context("fork").Pool(min(16, os.cpu_count() or 1)) as pool:
        rs = pool.map(contest, [(5000 + i, K) for i in range(C)], chunksize=1)
    print(f"[{name}] C={C} K={K} ({time.time() - t0:.0f}s): P(our best strictly beats the field's best)")
    for fname in ("sim", "live"):
        for c in rs[0][fname]:
            w = np.mean([r[fname][c][0] for r in rs])
            m = np.mean([r[fname][c][1] for r in rs])
            print(f"KAPPA {name} field={fname:4s} {c:10s} P={w:.3f}+-{math.sqrt(max(w * (1 - w), 1e-9) / C):.3f}"
                  f"  mean run score {m:.2f}", flush=True)


if __name__ == "__main__":
    main()
