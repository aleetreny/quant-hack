"""Q1/Q3 follow-ups on the run pool (paired contests):
- K-aware target q_p(m) = quantile p of max(X, M_m) (M_m = best of the m EV-DP runs left) vs
  max(X, Y) + d, for higher p;
- robustness when the assumed runs-left m is wrong by a factor (true K fixed);
- hedging: play plain EV-DP on a fraction of the runs, which ties the group whenever their final
  record falls on one of those runs. Reports win / tie / lose.
`python research/r1_rules2.py [contests]`
"""
import sys
import numpy as np

C = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
B0 = 2800.0
z = np.load("_local/r1/pool.npz"); x, Y, Bs = z["x"], z["Y"].astype(float), z["Bs"]
N = len(x); xs = np.sort(x)


def col(B):
    return np.clip(np.rint((B - Bs[0]) / (Bs[1] - Bs[0])).astype(int), 0, len(Bs) - 1)


def kaware(p, d, f=1.0):
    def rule(X, Yb, m):
        mm = max(1.0, m * f)
        return np.maximum(np.maximum(X, np.quantile(xs, p ** (1 / mm))), np.maximum(X, Yb) + d)
    return rule


def contest(K, rule, seed, ev_every=0):
    g = np.random.default_rng(seed)
    idx = g.integers(0, N, (C, K))
    X = np.full(C, B0); Yb = np.zeros(C)
    for k in range(K):
        j = idx[:, k]
        if ev_every and (K - 1 - k) % ev_every == 0:  # last run is always an EV run when hedging
            y = x[j]
        else:
            y = Y[j, col(rule(X, Yb, K - k))]
        Yb = np.maximum(Yb, y)
        X = np.maximum(X, x[j])
    return (Yb > X).mean(), (Yb == X).mean(), (Yb < X).mean()


for K in (10, 20, 40, 60):
    seed = 100 + K
    print(f"\n### K={K}")
    for p in (0.6, 0.7, 0.8, 0.9):
        print(f"q{int(p*100)} v max(X,Y)+d: " + "  ".join(
            f"d={d}:{contest(K, kaware(p, d), seed)[0]:.3f}" for d in (0, 25, 50, 75)))
    print("q70 d=50, assumed m = f * true m: " + "  ".join(
        f"f={f}:{contest(K, kaware(0.7, 50, f), seed)[0]:.3f}" for f in (0.25, 0.5, 1.0, 2.0, 4.0)))
    for e in (4, 3, 2):
        w, t, l = contest(K, kaware(0.7, 50), seed, ev_every=e)
        print(f"hedge: EV on 1 run in {e}: win {w:.3f} tie {t:.3f} lose {l:.3f}")
    sys.stdout.flush()
