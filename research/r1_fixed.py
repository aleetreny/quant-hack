"""Q1/Q3 on the run pool: fixed target B vs B = current best + delta, best-of-K contests vs the
EV-DP clone group (group best starts at B0). `python research/r1_fixed.py [contests]`
Win = our best > group best; tie = equal; lose = below (EV-DP always ties: P(lose)=0, P(win)=0).
"""
import sys
import numpy as np

C = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
B0 = 2800.0
KS = (10, 20, 40, 60)
z = np.load("_local/r1/pool.npz"); x, Y, Bs = z["x"], z["Y"].astype(float), z["Bs"]
N = len(x)
g = np.random.default_rng(11)


def col(B):
    return np.clip(np.rint((B - Bs[0]) / (Bs[1] - Bs[0])).astype(int), 0, len(Bs) - 1)


def contest(K, rule):
    """rule(X, Yb) -> target per contest. Returns (win, tie, lose, mean group best)."""
    idx = g.integers(0, N, (C, K))
    X = np.full(C, B0); Yb = np.zeros(C)
    for k in range(K):
        c = col(rule(X, Yb, K - k))
        Yb = np.maximum(Yb, Y[idx[:, k], c])
        X = np.maximum(X, x[idx[:, k]])
    return (Yb > X).mean(), (Yb == X).mean(), (Yb < X).mean(), X.mean()


rows = []
for K in KS:
    print(f"\n### K = {K}  (group best final: mean of max(B0, K EV runs))")
    for B in range(2850, 3151, 50):
        w, t, l, mx = contest(K, lambda X, Yb, m, B=B: np.full(C, float(B)))
        print(f"fixed B={B}: win {w:.3f}  lose {l:.3f}  tie {t:.4f}   (group best mean {mx:.0f})")
    for d in (30, 60, 90, 120, 150, 200):
        w, t, l, _ = contest(K, lambda X, Yb, m, d=d: X + d)
        w2, t2, l2, _ = contest(K, lambda X, Yb, m, d=d: np.maximum(X, Yb) + d)
        print(f"dynamic delta={d:3d}: [X+d] win {w:.3f} lose {l:.3f}   [max(X,Y)+d] win {w2:.3f} lose {l2:.3f}")
    sys.stdout.flush()
