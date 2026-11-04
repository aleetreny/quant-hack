"""Round 2: the code we submitted (r2_prev_submitted.py) vs Ale3 (solution.py) on the same sequences.

Per game: score of prev (fresh instance), prev (one instance kept across games, in case the server
reuses it), Ale3, and the exact N(-0.2, 1) DP reference; cold-start diagnostics for prev.
Then best-of-K contests resampled from the paired pool.
`python research/r2_compare_prev.py [games_per_dist] [dists]`
"""
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from arena import load_solution, ROOT
from r2_decide import EXACT, T, L

PREV = ROOT / "research" / "r2_prev_submitted.py"
DISTS = {
    "N(-0.2,1)": lambda g: g.normal(-0.2, 1, T),
    "N(0,1)": lambda g: g.normal(0, 1, T),
    "pareto(3)": lambda g: g.pareto(3, T),
    "N(-0.2,1.3)": lambda g: g.normal(-0.2, 1.3, T),
    "N(U(-0.35,0.02),1)": lambda g: g.normal(g.uniform(-0.35, 0.02), 1, T),
}


def run(p, r, trace=False):
    s, lu, acc = 0.0, 0, []
    for t in range(1, T + 1):
        lk = t <= lu
        if p(float(r[t - 1]), L, t, T, s, [], lk) and not lk:
            s += r[t - 1]; lu = t + L; acc.append((t, r[t - 1]))
    return (s, acc) if trace else s


def ref(r, kap=1.0):
    s, lu = 0.0, 0
    for t in range(1, T + 1):
        if t > lu and r[t - 1] > 0 and r[t - 1] >= kap * EXACT[T - t + 1]:
            s += r[t - 1]; lu = t + L
    return s


def work(args):
    dist, seed, G = args
    g = np.random.default_rng(seed)
    persistent = load_solution(PREV)
    out = []
    for _ in range(G):
        r = DISTS[dist](g)
        sp, acc = run(load_solution(PREV), r, trace=True)
        early = [x for t, x in acc if t <= 30]
        out.append((sp, run(persistent, r), run(load_solution(), r), ref(r),
                    len(early), sum(early), sum(1 for _, x in acc if x <= 0)))
    return out


if __name__ == "__main__":
    G = int(sys.argv[1]) if len(sys.argv) > 1 else 240
    names = (sys.argv[2] if len(sys.argv) > 2 else ";".join(DISTS)).split(";")
    rng = np.random.default_rng(5)
    for dist in names:
        jobs = [(dist, 1000 + 7 * i, G // 12) for i in range(12)]
        with Pool(12) as p:
            res = np.array([row for part in p.map(work, jobs) for row in part])
        prev, prev_keep, ale3, exact = res[:, 0], res[:, 1], res[:, 2], res[:, 3]
        n = len(res)
        print(f"\n### {dist}: {n} partidas emparejadas")
        for k, v in (("enviado (instancia nueva)", prev), ("enviado (instancia reutilizada)", prev_keep),
                     ("Ale3", ale3), ("DP exacta N(-0.2,1) x1.00", exact)):
            print(f"  {k:32s} media {v.mean():8.2f}  sd {v.std():6.2f}")
        d = ale3 - prev
        print(f"  Ale3 - enviado: {d.mean():+.2f} +- {d.std() / np.sqrt(n):.2f} por partida; Ale3 mejor en {np.mean(d > 0):.0%}")
        print(f"  enviado, primeros 30 turnos: {res[:, 4].mean():.2f} aceptaciones/partida, suma {res[:, 5].mean():+.2f}; "
              f"aceptaciones con reward <= 0: {res[:, 6].mean():.2f}/partida")
        for K in (10, 30):
            idx = rng.integers(0, n, (20000, K))
            bp, ba, bk = prev[idx].max(1), ale3[idx].max(1), prev_keep[idx].max(1)
            print(f"  K={K}: P(mejor Ale3 > mejor enviado) {np.mean(ba > bp):.3f} | vs reutilizada {np.mean(ba > bk):.3f}")
