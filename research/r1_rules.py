"""Q1 refinements on the run pool: fine fixed-B sweep, fixed floor + dynamic bump, and K-aware
targets from quantiles of the group's final best. `python research/r1_rules.py [contests]`
"""
import sys
import numpy as np

C = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
B0 = 2800.0
z = np.load("_local/r1/pool.npz"); x, Y, Bs = z["x"], z["Y"].astype(float), z["Bs"]
N = len(x); xs = np.sort(x)


def Ginv(p):
    return np.quantile(xs, p)


def col(B):
    return np.clip(np.rint((B - Bs[0]) / (Bs[1] - Bs[0])).astype(int), 0, len(Bs) - 1)


def contest(K, rule, seed):
    g = np.random.default_rng(seed)  # same seed -> same contests for every rule (paired)
    idx = g.integers(0, N, (C, K))
    X = np.full(C, B0); Yb = np.zeros(C)
    for k in range(K):
        Yb = np.maximum(Yb, Y[idx[:, k], col(rule(X, Yb, K - k))])
        X = np.maximum(X, x[idx[:, k]])
    return (Yb > X).mean()


for K in (10, 20, 40, 60):
    seed = 100 + K
    fx = {B: contest(K, lambda X, Yb, m, B=B: np.full(C, float(B)), seed) for B in range(2880, 3121, 10)}
    bB = max(fx, key=fx.get)
    qm = {p: max(B0, Ginv(p ** (1 / K))) for p in (0.3, 0.5, 0.7)}
    gm = np.maximum(B0, np.random.default_rng(0).choice(x, (40000, K)).max(1)).mean()
    print(f"\n### K={K}: group final best q30/q50/q70 = " + "/".join(f"{qm[p]:.0f}" for p in qm)
          + f", mean {gm:.0f}")
    print("fixed B: " + "  ".join(f"{B}:{w:.3f}" for B, w in fx.items() if B % 20 == 0)
          + f"   -> best {bB} ({fx[bB]:.3f})")
    for Bf in (bB - 50, bB, bB + 50):
        out = []
        for d in (50, 100, 150):
            w = contest(K, lambda X, Yb, m, Bf=Bf, d=d: np.maximum(Bf, np.maximum(X, Yb) + d), seed)
            out.append(f"d={d}:{w:.3f}")
        print(f"floor {Bf} v max(X,Y)+d: " + "  ".join(out))
    for p in (0.3, 0.4, 0.5, 0.6, 0.7):
        out = []
        for d in (0, 50, 100):
            # target = quantile p of max(X, M_m) over the m runs left, never below max(X,Y)+d
            w = contest(K, lambda X, Yb, m, p=p, d=d: np.maximum(np.maximum(X, Ginv(p ** (1 / m))),
                                                                 np.maximum(X, Yb) + d), seed)
            out.append(f"d={d}:{w:.3f}")
        print(f"K-aware q{int(p * 100)}(max(X,M_m)) v max(X,Y)+d: " + "  ".join(out))
    sys.stdout.flush()
