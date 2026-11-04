"""Fine kappa sweep, late-only perturbation, and robustness to a group whose EV-DP differs from ours.
Policy: threshold kappa * gap[n] for rounds t > t0 (EV-DP before). Group: kappa_g * gap (1 = ours).
`python research/r1_kappa2.py [N] [contests]`
"""
import sys
from multiprocessing import Pool
import numpy as np
from r1_lib import T, L, MU, GAP

# (kappa, t0) variants; group variants kappa_g
VARS = [(1.0, 0), (1.003, 0), (1.007, 0), (1.01, 0), (1.015, 0), (1.03, 0),
        (1.02, 1000), (1.05, 1000), (1.05, 1500), (1.1, 1500), (1.1, 1800), (1.2, 1800)]
GROUPS = (1.0, 0.999, 1.001, 0.99, 1.01)
CHUNK = 10000


def play_v(rw, kap, t0):
    N = rw.shape[0]; nv = len(kap)
    s = np.zeros((N, nv)); lock = np.zeros((N, nv), np.int32)
    for t in range(1, T + 1):
        r = rw[:, t - 1][:, None]
        k = np.where(t > t0, kap, 1.0)
        a = (r >= k * GAP[T - t + 1]) & (t > lock)
        s += np.where(a, r, 0.0); lock = np.where(a, t + L, lock)
    return s


def work(seed):
    rw = np.random.default_rng(seed).exponential(MU, (CHUNK, T))
    kap = np.array([v[0] for v in VARS] + list(GROUPS)); t0 = np.array([v[1] for v in VARS] + [0] * len(GROUPS))
    return play_v(rw, kap, t0).astype(np.float32)


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 400000
    C = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
    with Pool(12) as p:
        S = np.concatenate(p.map(work, range(9000, 9000 + N // CHUNK))).astype(float)
    np.savez("_local/r1/pool_kappa2.npz", S=S)
    nv = len(VARS)
    B0 = 2800.0
    for gi, kg in enumerate(GROUPS):
        x = S[:, nv + gi]
        print(f"\n## group plays EV threshold x{kg}")
        for K in (10, 20, 40, 60):
            idx = np.random.default_rng(100 + K).integers(0, N, (C, K))
            X = np.maximum(B0, x[idx].max(1))
            out = []
            for j, (k, t0) in enumerate(VARS):
                Yb = S[idx, j].max(1)
                out.append(f"{k}@{t0}:{(Yb > X).mean():.3f}/{(Yb < X).mean():.3f}")
            print(f"K={K} win/lose: " + "  ".join(out), flush=True)
