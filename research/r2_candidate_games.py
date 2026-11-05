"""Full 2000-round games of research/r2_solution_candidate.py: score, P(score > TARGET), time.

`uv run python research/r2_candidate_games.py [games] [dist]`
"""
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2_solution_candidate as cand  # noqa: E402
from target_dp_general import DISTS, T, L  # noqa: E402

cand.DEBUG = True


def game(play, rw):
    score, lock, others = 0.0, 0, []
    for t in range(1, T + 1):
        a = play(float(rw[t - 1]), L, t, T, score, others, t <= lock)
        if a and t > lock:
            score += rw[t - 1]
            lock = t + L
        others = [(0.0, False)] * 23
    return score


def one(args):
    target, pq, rw = args
    cand.TARGET, cand.PRIOR_Q = target, pq
    t0 = time.time()
    s = game(cand.Solution().play, rw)
    return s, time.time() - t0


if __name__ == "__main__":
    G = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    dist = DISTS[sys.argv[2] if len(sys.argv) > 2 else "eth"]
    g = np.random.default_rng(5)
    sh = g.normal(0, dist.run_sd, G)
    rws = dist.sample(g, (G, T)) + sh[:, None]
    prior = list(dist.quantiles(400))
    B = float(sys.argv[3]) if len(sys.argv) > 3 else 125.0
    for name, target, pq in (("EV, no prior", None, None), ("EV, prior", None, prior),
                             ("target, no prior", B, None), ("target, prior", B, prior)):
        with get_context("fork").Pool(min(16, os.cpu_count() or 1)) as pool:
            out = pool.map(one, [(target, pq, rws[k]) for k in range(G)], chunksize=1)
        sc, tm = np.array([o[0] for o in out]), [o[1] for o in out]
        print(f"{name:18s} mean {sc.mean():7.2f} sd {sc.std():5.2f}  P(>{B:g}) {(sc > B).mean():.3f}+-{np.sqrt((sc > B).mean() * (1 - (sc > B).mean()) / G):.3f}  "
              f"time/game mean {np.mean(tm):.1f}s max {np.max(tm):.1f}s", flush=True)
