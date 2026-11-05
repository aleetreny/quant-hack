"""Best-of-K contest on shared reward sequences: EV-DP clones vs one target-DP team.

Each run every team sees the same rewards. The DP clones tie every run, so their best is the max
of EV-DP scores. We resubmit between runs with target = max(B0, group best so far) + delta.
`python research/contest.py [contests] [K]`
"""
import sys
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
from target_dp import T, L, MU, ev_gap, target_tables

C = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
K = int(sys.argv[2]) if len(sys.argv) > 2 else 20
H = 1.0
W = target_tables(3600, h=H)
D = W.shape[1] - 1
gap = ev_gap() * MU


def run(rw, B):
    """rw (N, T); B (N,) targets (np.inf -> pure EV). Returns EV scores, target scores."""
    N = rw.shape[0]
    sx = np.zeros(N); sy = np.zeros(N)
    lx = np.zeros(N, int); ly = np.zeros(N, int)
    for t in range(1, T + 1):
        n = T - t + 1
        r = rw[:, t - 1]
        ev = r >= gap[n]
        ax = ev & (t > lx)
        d = B - sy
        i = np.clip(np.rint(d / H).astype(int), 0, D)
        Sv = W[n - 1, i]
        ia = np.rint((d - r) / H).astype(int)
        Av = np.where(ia <= 0, 1.0, W[max(n - 1 - L, 0), np.clip(ia, 0, D)])
        decided = (i > 0) & (np.maximum(Av, Sv) > 1e-9)
        ay = np.where(decided, Av > Sv + 1e-12, ev) & (t > ly)
        sx += np.where(ax, r, 0); lx = np.where(ax, t + L, lx)
        sy += np.where(ay, r, 0); ly = np.where(ay, t + L, ly)
    return sx, sy


g = np.random.default_rng(7)
for B0 in [float(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "2634.77").split(",")]:
    for delta in [float(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "80,150,250").split(",")]:
        best_x = np.full(C, B0); best_y = np.zeros(C)
        for k in range(K):
            rw = g.exponential(MU, (C, T))
            sx, sy = run(rw, best_x + delta)
            best_y = np.maximum(best_y, sy)
            best_x = np.maximum(best_x, sx)
        win = (best_y > best_x).mean()
        print(f"B0={B0:.0f} K={K} delta={delta:3.0f}: P(target-DP strictly beats DP group) = {win:.3f}  "
              f"(DP group final best mean {best_x.mean():.0f})", flush=True)
