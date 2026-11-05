"""Round 2 head-to-head on the same N(-0.2, 1) runs: Lea's rule vs Ale3 (solution.py) vs exact EV-DP.

Lea = research/r2_contest.lea_threshold_path (EV-DP on seen rewards + fading N(-0.2,1) pseudo-samples).
Ale3 = the real solution.py, one fresh instance per run. Then best-of-K contests resampled from the
paired pool, including against the field (best so far FIELD0, future = exact EV-DP clones).
`python research/r2_lea_vs_ale3.py [runs] [K,K]`
"""
import os
import sys
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from arena import load_solution
from r2_decide import EXACT, T, L
from r2_contest import lea_threshold_path

FIELD0 = 135.67


def score_thr(rw, thr):
    R = rw.shape[0]
    s = np.zeros(R); lock = np.zeros(R, int)
    for t in range(1, T + 1):
        r = rw[:, t - 1]
        a = (r > 0) & (r >= thr(t)) & (t > lock)
        s += np.where(a, r, 0.0); lock = np.where(a, t + L, lock)
    return s


def run_sol(r):
    p = load_solution()
    s, lu = 0.0, 0
    for t in range(1, T + 1):
        lk = t <= lu
        if p(float(r[t - 1]), L, t, T, s, [], lk) and not lk:
            s += r[t - 1]; lu = t + L
    return s


def work(seed):
    rw = np.random.default_rng(seed).normal(-0.2, 1.0, (100, T))
    lthr = lea_threshold_path(rw)
    lea = score_thr(rw, lambda t: lthr[:, t])
    ex = score_thr(rw, lambda t: EXACT[T - t + 1])
    ale3 = np.array([run_sol(r) for r in rw])
    return np.stack([lea, ale3, ex], 1)


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2400
    Ks = [int(k) for k in (sys.argv[2] if len(sys.argv) > 2 else "6,10").split(",")]
    with get_context("fork").Pool(12) as p:
        res = np.concatenate(p.map(work, [500 + i for i in range(N // 100)]))
    lea, ale3, ex = res.T
    n = len(res)
    print(f"{n} runs N(-0.2,1) emparejados")
    for k, v in (("Lea", lea), ("Ale3", ale3), ("DP exacta x1.00", ex)):
        print(f"  {k:16s} media {v.mean():.2f}  sd {v.std():.2f}")
    d = ale3 - lea
    print(f"  Ale3 - Lea: {d.mean():+.3f} +- {d.std() / np.sqrt(n):.3f} por run; Ale3 > Lea en {np.mean(d > 0):.0%}, "
          f"empate {np.mean(d == 0):.0%}")
    g = np.random.default_rng(1)
    for K in Ks:
        idx = g.integers(0, n, (40000, K))
        bl, ba, be = lea[idx].max(1), ale3[idx].max(1), ex[idx].max(1)
        top = np.maximum(FIELD0, be)
        print(f"  K={K}: P(best Ale3 > best Lea) {np.mean(ba > bl):.3f}, P(best Lea > best Ale3) {np.mean(bl > ba):.3f} | "
              f"P(> campo {FIELD0} y clones exactos): Ale3 {np.mean(ba > top):.3f}  Lea {np.mean(bl > top):.3f}  "
              f"exacta {np.mean(be > FIELD0):.3f}")
