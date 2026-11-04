"""Near-EV perturbations: EV-DP with the threshold scaled by kappa. On shared sequences the score y
stays highly correlated with the group's x, so we beat the group's record run about half the time.
Pool of (x, y_kappa) pairs (float64, so runs where y never diverges stay exact ties), then
best-of-K contests vs the group (best starts at B0).
`python research/r1_kappa.py [N] [contests]`
"""
import sys
from multiprocessing import Pool
import numpy as np
from r1_lib import T, L, MU, GAP, play_ev

KAPPAS = (0.97, 0.99, 1.01, 1.02, 1.025, 1.03, 1.035, 1.04, 1.05, 1.06, 1.08, 1.1)
CHUNK = 10000


def play_k(rw, kap):
    """rw (N, T); kap (nk,) -> scores (N, nk)."""
    N = rw.shape[0]
    s = np.zeros((N, len(kap))); lock = np.zeros((N, len(kap)), np.int32)
    for t in range(1, T + 1):
        r = rw[:, t - 1][:, None]
        a = (r >= kap * GAP[T - t + 1]) & (t > lock)
        s += np.where(a, r, 0.0); lock = np.where(a, t + L, lock)
    return s


def work(seed):
    rw = np.random.default_rng(seed).exponential(MU, (CHUNK, T))
    return play_ev(rw), play_k(rw, np.array(KAPPAS))  # float64: exact ties must stay ties


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 400000
    C = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
    with Pool(12) as p:
        res = p.map(work, range(5000, 5000 + N // CHUNK))
    x = np.concatenate([r[0] for r in res]); Y = np.concatenate([r[1] for r in res]).astype(float)
    np.savez("_local/r1/pool_kappa.npz", x=x, Y=Y, kappas=KAPPAS)
    print("kappa   mean    sd(y-x)  corr   P(y>x | x>q99)  P(y<x | x>q99)")
    hi = x > np.quantile(x, .99)
    for j, k in enumerate(KAPPAS):
        y = Y[:, j]
        print(f"{k:5.3f} {y.mean():7.1f} {np.std(y-x):8.1f}  {np.corrcoef(x,y)[0,1]:.3f}  {(y[hi]>x[hi]).mean():.3f}  {(y[hi]<x[hi]).mean():.3f}")
    B0 = 2800.0
    for K in (10, 20, 40, 60):
        idx = np.random.default_rng(100 + K).integers(0, len(x), (C, K))
        X = np.maximum(B0, x[idx].max(1))
        out = []
        for j, k in enumerate(KAPPAS):
            Yb = Y[idx, j].max(1)
            out.append(f"{k}:{(Yb>X).mean():.3f}/{(Yb<X).mean():.3f}")
        print(f"K={K} win/lose (tie = rest): " + "  ".join(out), flush=True)
