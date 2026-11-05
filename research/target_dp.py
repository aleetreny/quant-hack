"""Best-score rounds: maximise P(score > TARGET) instead of E[score].

Ranking is the best score over many runs, and every team running the expected-value DP ties to the
cent on each run (all teams see the same reward sequence). The only way to finish strictly first is
to beat the leader's best in some run, so the per-run objective is P(S > B), not E[S].

State: n rounds left, d = B - score still needed. W_n(d) = P(reach d with n rounds left), with
    W_0(d) = 1[d <= 0]
    W_n(d) = E_r[ max( W_{n-1}(d), W_{n-1-L}(d - r) ) ]
For exponential rewards the truncated convolution is O(D) per step (memoryless trick):
    sum_{j >= j*} p_j A(i - j) = rho^{j*} G(i - j*),   G = full convolution of A with the geometric pmf.

`python research/target_dp.py` compares EV-DP and target-DP per run and in a best-of-K contest.
"""
import math
import sys
import time

import numpy as np

T, L, MU = 2000, 15, 10.0


def ev_gap(T=T, L=L):
    # expected-value DP threshold per rounds-left n (exponential mean 1)
    W = [0.0] * (T + 1)
    for n in range(1, T + 1):
        acc = W[n - 1 - L] if n - 1 - L > 0 else 0.0
        W[n] = W[n - 1] + math.exp(-(W[n - 1] - acc))
    gap = np.zeros(T + 1)
    for n in range(1, T + 1):
        acc = W[n - 1 - L] if n - 1 - L > 0 else 0.0
        gap[n] = W[n - 1] - acc
    return gap  # threshold for mean 1; multiply by MU


def target_tables(B, T=T, L=L, mu=MU, h=1.0, block=400):
    """W[n] on grid d = i*h, i = 0..D (W = 1 for d <= 0). Returns float32 array (T+1, D+1).

    Rewards are rounded to the nearest grid point: p_0 = 1 - rho^0.5, p_j = c rho^j (j >= 1),
    c = (1 - rho) rho^-0.5, so the tail sum_{j >= j*} p_j A(i - j) = c rho^j* H(i - j*) with
    H(i) = A(i) + rho H(i - 1), H(-1) = 1 / (1 - rho).
    """
    D = int(math.ceil(B / h)) + 1
    rho = math.exp(-h / mu)
    c0 = (1 - rho) / math.sqrt(rho)
    p0 = 1 - math.sqrt(rho)
    W = np.zeros((T + 1, D + 1), np.float32)
    W[0, 0] = 1.0
    idx = np.arange(D + 1)
    for n in range(1, T + 1):
        S = W[n - 1].astype(np.float64)
        A = W[max(n - 1 - L, 0)].astype(np.float64)
        H = np.empty(D + 1)
        prev = 1.0 / (1 - rho)
        for s in range(0, D + 1, block):
            a = A[s:s + block]
            k = np.arange(len(a), dtype=float)
            pw = rho ** (-k)
            H[s:s + block] = rho ** (k + 1) * prev + np.cumsum(a * pw) / pw
            prev = H[s + len(a) - 1]
        Hm = np.concatenate(([1.0 / (1 - rho)], H))  # Hm[i + 1] = H(i)
        # c = #k with A(k) > S(i); k* = c - 1; j* = max(i - k*, 0)
        cnt = np.searchsorted(-A, -S, side="left")
        jstar = np.maximum(idx - (cnt - 1), 0)
        tail_p = np.where(jstar >= 1, rho ** (jstar - 0.5), 1.0)
        tail_v = np.where(jstar >= 1, c0 * rho ** jstar * Hm[np.clip(idx - jstar + 1, 0, D + 1)],
                          p0 * A + c0 * rho * Hm[idx])
        Wn = S * (1 - tail_p) + tail_v
        Wn[0] = 1.0
        W[n] = np.minimum(Wn, 1.0)
    return W


def simulate(rewards, policy):
    """rewards: (N, T). Vectorised over N independent runs. policy(n, score, r, mask) -> bool array."""
    N, TT = rewards.shape
    score = np.zeros(N)
    lock_until = np.zeros(N, int)
    for t in range(1, TT + 1):
        free = t > lock_until
        r = rewards[:, t - 1]
        acc = policy(TT - t + 1, score, r) & free
        score += np.where(acc, r, 0.0)
        lock_until = np.where(acc, t + L, lock_until)
    return score


def ev_policy(kappa=1.0):
    gap = ev_gap() * MU
    def f(n, s, r):
        return r >= kappa * gap[n]
    return f


def target_policy(B, h=1.0, W=None):
    W = target_tables(B, h=h) if W is None else W
    gap = ev_gap() * MU
    D = W.shape[1] - 1
    def f(n, s, r):
        d = B - s
        i = np.clip(np.rint(d / h).astype(int), 0, D)
        Sv = W[n - 1, i]
        m = max(n - 1 - L, 0)
        ia = np.rint((d - r) / h).astype(int)
        Av = np.where(ia <= 0, 1.0, W[m, np.clip(ia, 0, D)])
        ev = r >= gap[n]
        # fall back to EV-DP when the target is already reached or hopeless
        decided = (i > 0) & (np.maximum(Av, Sv) > 1e-9)
        return np.where(decided, Av > Sv + 1e-12, ev)
    return f


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    g = np.random.default_rng(1)
    rw = g.exponential(MU, (N, T))
    t0 = time.time()
    ev = simulate(rw, ev_policy())
    print(f"EV-DP: mean {ev.mean():.1f} sd {ev.std():.1f}  q50 {np.median(ev):.0f} q90 {np.quantile(ev,.9):.0f} "
          f"q99 {np.quantile(ev,.99):.0f} max {ev.max():.0f}  ({time.time()-t0:.1f}s)")
    for k in (0.9, 1.1, 1.2):
        sc = simulate(rw, ev_policy(k))
        print(f"EV x{k}: mean {sc.mean():.1f} sd {sc.std():.1f} q99 {np.quantile(sc,.99):.0f}")
    for B in (2800, 2900, 3000):
        t0 = time.time(); W = target_tables(B); tb = time.time() - t0
        sc = simulate(rw, target_policy(B, W=W))
        pe, pt = (ev > B).mean(), (sc > B).mean()
        print(f"B={B}: P(EV>B)={pe:.4f}  P(target>B)={pt:.4f}  x{pt/max(pe,1e-9):.2f}  "
              f"target mean {sc.mean():.0f}  W0={W[T, int(B)]:.4f}  table {tb:.2f}s")
