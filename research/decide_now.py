"""Which round-1 strategy to leave running, from the current leaderboard state.

Field: EV-DP clones (exact threshold) share best GROUP0; the leader holds LEADER (treated as fixed,
then playing like the clones). We start at OURS. Each run all see the same rewards. Win = our final
best > max(leader, clones' final best). Paired: every strategy sees the same contests.
`python research/decide_now.py [contests] [K,K,...]`
"""
import sys
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
from target_dp import T, L, MU, ev_gap, target_tables

LEADER, GROUP0, OURS = 2892.63, 2862.17, 2862.17
GAP = ev_gap() * MU
W = target_tables(3100)
D = W.shape[1] - 1

STRATS = {
    "EV exacta (x1.000)": ("k", 1.0),
    "Original repo (media online)": ("online", 1.0),
    "Ale1 target-DP 2990": ("target", 2990.0),
    "kappa x1.010": ("k", 1.010),
    "kappa x1.013": ("k", 1.013),
    "kappa x1.020": ("k", 1.020),
    "kappa x1.030": ("k", 1.030),
}


def play_all(rw):
    C = rw.shape[0]
    names = list(STRATS)
    s = np.zeros((C, len(names) + 1)); lock = np.zeros((C, len(names) + 1), int)
    csum = np.cumsum(rw, axis=1)
    for t in range(1, T + 1):
        n = T - t + 1
        r = rw[:, t - 1]
        acc = np.empty((C, len(names) + 1), bool)
        acc[:, -1] = r >= GAP[n]  # the clone group
        for j, nm in enumerate(names):
            kind, p = STRATS[nm]
            if kind == "k":
                acc[:, j] = r >= p * GAP[n]
            elif kind == "online":
                mu_hat = csum[:, t - 1] / t
                acc[:, j] = (r >= GAP[n] * mu_hat) & (t > 8)
            else:
                d = p - s[:, j]
                i = np.clip(np.rint(d).astype(int), 0, D)
                sv = W[n - 1, i]
                ia = np.rint(d - r).astype(int)
                av = np.where(ia <= 0, 1.0, W[max(n - 1 - L, 0), np.clip(ia, 0, D)])
                dec = (i > 0) & (np.maximum(av, sv) > 1e-9)
                acc[:, j] = np.where(dec, av > sv + 1e-12, r >= GAP[n])
        acc &= t > lock
        s += np.where(acc, r[:, None], 0.0)
        lock = np.where(acc, t + L, lock)
    return s[:, :-1], s[:, -1]


if __name__ == "__main__":
    C = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    Ks = [int(k) for k in (sys.argv[2] if len(sys.argv) > 2 else "10,20,40").split(",")]
    g = np.random.default_rng(2026)
    for K in Ks:
        ours = np.full((C, len(STRATS)), OURS)
        group = np.full(C, GROUP0)
        for _ in range(K):
            y, x = play_all(g.exponential(MU, (C, T)))
            ours = np.maximum(ours, y)
            group = np.maximum(group, x)
        top = np.maximum(group, LEADER)
        print(f"\n### K = {K} runs restantes, {C} concursos (mejor final del grupo: media {group.mean():.0f})")
        print(f"{'estrategia':30s} {'1o solo':>8s} {'empate':>8s} {'debajo':>8s} {'>= clones':>10s}")
        for j, nm in enumerate(STRATS):
            o = ours[:, j]
            print(f"{nm:30s} {np.mean(o > top):8.3f} {np.mean(o == top):8.3f} {np.mean(o < top):8.3f} "
                  f"{np.mean(o >= group):10.3f}")
