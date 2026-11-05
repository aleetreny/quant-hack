"""Pool of independent runs: EV-DP score x and target-DP score y_B for a grid of targets B.
Every policy here depends only on the run's own reward sequence, so best-of-K contests are built by
resampling runs from the pool. `python research/r1_pool.py [N] [out]`
"""
import sys
from multiprocessing import Pool
import numpy as np
from r1_lib import T, MU, play, play_ev
from target_dp import target_tables

BS = np.arange(2800, 3401, 10.0)
CHUNK = 5000


def work(seed):
    W = target_tables(3700)
    rw = np.random.default_rng(seed).exponential(MU, (CHUNK, T))
    return play_ev(rw), play(rw, BS[None, :], W).astype(np.float32)


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
    out = sys.argv[2] if len(sys.argv) > 2 else "_local/r1/pool.npz"
    with Pool(12) as p:
        res = p.map(work, range(1000, 1000 + N // CHUNK))
    x = np.concatenate([r[0] for r in res]); Y = np.concatenate([r[1] for r in res])
    np.savez(out, x=x, Y=Y, Bs=BS)
    print(f"pool N={len(x)}: EV mean {x.mean():.1f} sd {x.std():.1f} q99 {np.quantile(x,.99):.0f}")
