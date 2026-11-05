"""Shadow-group play: we see the same rewards as the EV-DP clones, so we can track their score g live
(simulate their EV rule, or read it from other_scores). Rule: threshold kappa_b * gap while D = s - g
<= 0 (perturb to diverge), plain EV-DP once D > 0: whenever both are free and accept the same reward
the lock states re-align and the lead D > 0 is frozen for the rest of the run.
Variants also: shadow of a group whose rule differs slightly (kappa_g) while we assume exact EV.
`python research/r1_shadow.py [N] [contests]`
"""
import os
import sys
from multiprocessing import Pool
import numpy as np
from r1_lib import T, L, MU, GAP

# (kappa_behind, kappa_ahead); first entry = plain kappa 1.025 (no shadow) for reference
VARS = [(1.025, 1.025), (1.025, 1.0), (1.05, 1.0), (1.1, 1.0), (1.2, 1.0), (1.5, 1.0), (0.97, 1.0)]
GROUPS = (1.0, 0.999, 1.001, 0.99, 1.01)
if os.environ.get("R1_SHADOW_VARS"):  # e.g. "1.07:1,1.1:1" (env so spawned workers see it)
    VARS = [tuple(float(a) for a in v.split(":")) for v in os.environ["R1_SHADOW_VARS"].split(",")]
    GROUPS = (1.0, 0.999, 1.001)  # true group threshold multiplier (shadow assumes 1.0)
CHUNK = 10000


def work(seed):
    rw = np.random.default_rng(seed).exponential(MU, (CHUNK, T))
    N = CHUNK; nv = len(VARS); ng = len(GROUPS)
    kb = np.array([v[0] for v in VARS])[None, :, None]
    ka = np.array([v[1] for v in VARS])[None, :, None]
    kg = np.array(GROUPS)[None, None, :]
    # shadow (assumed exact EV), true group per kappa_g, ours per (variant, group)
    g = np.zeros((N, 1, 1)); lg = np.zeros((N, 1, 1), np.int32)
    xg = np.zeros((N, 1, ng)); lx = np.zeros((N, 1, ng), np.int32)
    s = np.zeros((N, nv, ng)); ls = np.zeros((N, nv, ng), np.int32)
    for t in range(1, T + 1):
        r = rw[:, t - 1][:, None, None]
        gap = GAP[T - t + 1]
        k = np.where(s - g > 0, ka, kb)
        a = (r >= k * gap) & (t > ls)
        ag = (r >= gap) & (t > lg)
        ax = (r >= kg * gap) & (t > lx)
        s += np.where(a, r, 0.0); ls = np.where(a, t + L, ls)
        g += np.where(ag, r, 0.0); lg = np.where(ag, t + L, lg)
        xg += np.where(ax, r, 0.0); lx = np.where(ax, t + L, lx)
    return xg[:, 0, :], s


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
    C = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
    with Pool(12) as p:
        res = p.map(work, range(7000, 7000 + N // CHUNK))
    X1 = np.concatenate([r[0] for r in res]); S = np.concatenate([r[1] for r in res])
    if not os.environ.get("R1_SHADOW_VARS"):
        np.savez("_local/r1/pool_shadow.npz", X=X1, S=S)
    B0 = 2800.0
    for gi, kgv in enumerate(GROUPS):
        x = X1[:, gi]
        print(f"\n## true group threshold x{kgv}; per-run P(y>x)/P(y<x) and mean y:")
        print("   " + "  ".join(f"{v}:{(S[:, j, gi] > x).mean():.3f}/{(S[:, j, gi] < x).mean():.3f} m{S[:, j, gi].mean():.0f}"
                             for j, v in enumerate(VARS)))
        for K in (10, 20, 40, 60):
            idx = np.random.default_rng(100 + K).integers(0, N, (C, K))
            X = np.maximum(B0, x[idx].max(1))
            out = [f"{v}:{(S[idx, j, gi].max(1) > X).mean():.3f}/{(S[idx, j, gi].max(1) < X).mean():.3f}"
                   for j, v in enumerate(VARS)]
            print(f"K={K} win/lose: " + "  ".join(out), flush=True)
