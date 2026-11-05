"""Round 2 (hidden distribution, best score): which strategy to leave running.

Rewards N(MU, SD) (public engine 'b' = normal(-0.2, 1)). Strategies, all vectorised over runs:
  exact-k   DP on the true normal, threshold x k
  learn     DP on a normal fitted online (no prior), the typical "learn the distribution" team
  eth       ETH round-2 winner: normal shrunk towards N(0,1) with 300 pseudo-obs, x1.05
Field = clones of one family; contest = best of K shared runs. Win = our best > field best.
`python research/r2_decide.py [contests] [K,K]`
"""
import math
import sys

import numpy as np

T, L = 2000, 15
MU, SD = -0.2, 1.0
MS = np.round(np.arange(-0.6, 0.41, 0.02), 2)
SS = np.round(np.arange(0.70, 1.31, 0.02), 2)


def excess(c, m, s):
    c = max(c, 0.0)  # never take a non-positive reward
    z = (c - m) / s
    return s * math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi) + (m - c) * 0.5 * math.erfc(z / math.sqrt(2))


def gap_table(m, s):
    V = [0.0] * (T + 1)
    g = np.zeros(T + 1)
    for n in range(1, T + 1):
        acc = V[n - 1 - L] if n - 1 - L > 0 else 0.0
        c = V[n - 1] - acc
        g[n] = max(c, 0.0)
        V[n] = V[n - 1] + excess(c, m, s)
    return g


GRID = np.array([[gap_table(m, s) for s in SS] for m in MS])  # (nm, ns, T+1)
EXACT = gap_table(MU, SD)


def snap(m, s):
    i = np.clip(np.rint((m - MS[0]) / 0.02).astype(int), 0, len(MS) - 1)
    j = np.clip(np.rint((s - SS[0]) / 0.02).astype(int), 0, len(SS) - 1)
    return i, j


def thresholds(rw):
    """Per-strategy threshold arrays (C, T) indexed by t-1, for families that depend on data."""
    C = rw.shape[0]
    t = np.arange(1, T + 1)
    cs, cq = np.cumsum(rw, 1), np.cumsum(rw * rw, 1)
    n = t[None, :]
    rem = T - t + 1
    out = {}
    m = cs / n
    s = np.sqrt(np.maximum(cq / n - m * m, 1e-6))
    i, j = snap(m, s)
    out["learn"] = np.where(n > 10, GRID[i, j, rem[None, :]], np.inf)
    w = 300.0
    m2 = cs / (n + w)
    v2 = (cq - n * m * m + w * 1.0) / (n + w)
    i, j = snap(m2, np.sqrt(np.maximum(v2, 1e-6)))
    out["eth"] = 1.05 * GRID[i, j, rem[None, :]]
    return out


def play(rw, thr):
    """thr (C, T) threshold per round -> scores (C,)."""
    C = rw.shape[0]
    s = np.zeros(C)
    lock = np.zeros(C, int)
    for t in range(1, T + 1):
        r = rw[:, t - 1]
        a = (r > 0) & (r >= thr[:, t - 1]) & (t > lock)
        s += np.where(a, r, 0.0)
        lock = np.where(a, t + L, lock)
    return s


OURS = {"exacta x1.00": 1.0, "exacta x1.01": 1.01, "exacta x1.03": 1.03, "exacta x1.05": 1.05, "exacta x1.10": 1.10}
FIELDS = ("exacta x1.00", "learn", "eth")

if __name__ == "__main__":
    C = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    Ks = [int(k) for k in (sys.argv[2] if len(sys.argv) > 2 else "10,30").split(",")]
    g = np.random.default_rng(3)
    rem = T - np.arange(1, T + 1) + 1
    for K in Ks:
        best = {k: np.full(C, -np.inf) for k in list(OURS) + ["learn", "eth"]}
        means = {k: [] for k in best}
        for _ in range(K):
            rw = g.normal(MU, SD, (C, T))
            th = thresholds(rw)
            for k, kap in OURS.items():
                th[k] = np.broadcast_to(kap * EXACT[rem][None, :], rw.shape)
            for k in best:
                sc = play(rw, th[k])
                best[k] = np.maximum(best[k], sc)
                means[k].append(sc.mean())
        print(f"\n### K = {K} runs, {C} concursos")
        print("media por run: " + "  ".join(f"{k}={np.mean(v):.2f}" for k, v in means.items()))
        for f in FIELDS:
            print(f"campo = clones de {f:13s}: " + "  ".join(
                f"{k}: {np.mean(best[k] > best[f]):.3f}" for k in list(OURS) if k != f))
