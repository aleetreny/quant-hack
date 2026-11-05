"""Round-1 helpers: value DP with an arbitrary terminal utility, and vectorised run simulation.

value_tables(F0) generalises target_dp.target_tables: on the grid d = i*h (d = Smax - score) it
solves V_n(d) = E_r[max(V_{n-1}(d), V_{n-1-L}(d - r))] with V_0(d) = F0[i] instead of 1[d <= 0].
The memoryless O(D) step only needs V non-increasing in d (true for any non-decreasing utility of
the score) and V = 1 for d <= 0, so F0[0] must be 1.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from target_dp import T, L, MU, ev_gap  # noqa: E402

GAP = ev_gap() * MU


def value_tables(F0, T=T, L=L, mu=MU, h=1.0, block=400):
    F0 = np.asarray(F0, np.float64)
    D = len(F0) - 1
    rho = math.exp(-h / mu)
    c0 = (1 - rho) / math.sqrt(rho)
    p0 = 1 - math.sqrt(rho)
    W = np.zeros((T + 1, D + 1), np.float32)
    W[0] = F0
    W[0, 0] = 1.0
    idx = np.arange(D + 1)
    k_full = np.arange(block, dtype=float)
    pw_full = rho ** (-k_full)
    lead_full = rho ** (k_full + 1)
    for n in range(1, T + 1):
        S = W[n - 1].astype(np.float64)
        A = W[max(n - 1 - L, 0)].astype(np.float64)
        H = np.empty(D + 1)
        prev = 1.0 / (1 - rho)
        for s in range(0, D + 1, block):
            a = A[s:s + block]
            m = len(a)
            pw = pw_full[:m]
            H[s:s + m] = lead_full[:m] * prev + np.cumsum(a * pw) / pw
            prev = H[s + m - 1]
        Hm = np.concatenate(([1.0 / (1 - rho)], H))
        cnt = np.searchsorted(-A, -S, side="left")
        jstar = np.maximum(idx - (cnt - 1), 0)
        tail_p = np.where(jstar >= 1, rho ** (jstar - 0.5), 1.0)
        tail_v = np.where(jstar >= 1, c0 * rho ** jstar * Hm[np.clip(idx - jstar + 1, 0, D + 1)],
                          p0 * A + c0 * rho * Hm[idx])
        Wn = S * (1 - tail_p) + tail_v
        Wn[0] = 1.0
        W[n] = np.minimum(Wn, 1.0)
    return W


def accept(W, n, d, r, h=1.0, tie_ev=False):
    """Decision of the table W (indexed by d = goal - score) for rounds-left n; EV fallback when the
    goal is reached or hopeless (same rule as target_dp.target_policy). tie_ev: also fall back to EV
    when the table is indifferent (flat utility regions)."""
    D = W.shape[1] - 1
    i = np.clip(np.rint(d / h).astype(np.int64), 0, D)
    Sv = W[n - 1, i]
    ia = np.rint((d - r) / h).astype(np.int64)
    Av = np.where(ia <= 0, 1.0, W[max(n - 1 - L, 0), np.clip(ia, 0, D)])
    decided = (i > 0) & (np.maximum(Av, Sv) > 1e-9)
    if tie_ev:
        decided &= np.abs(Av - Sv) > 1e-9 * np.maximum(Av, Sv)
    return np.where(decided, Av > Sv + 1e-12, r >= GAP[n])


def play(rw, goal, W, h=1.0, tie_ev=False):
    """rw (N, T) rewards; goal broadcastable to (N, ...) (target B, or Smax for a utility table).
    Returns final scores with shape broadcast(goal)."""
    N, TT = rw.shape
    goal = np.asarray(goal, float)
    shape = np.broadcast(np.empty((N,) + (1,) * max(goal.ndim - 1, 0)), goal).shape
    s = np.zeros(shape)
    lock = np.zeros(shape, np.int32)
    ext = (slice(None),) + (None,) * (len(shape) - 1)
    for t in range(1, TT + 1):
        r = np.broadcast_to(rw[:, t - 1][ext], shape)
        a = accept(W, TT - t + 1, goal - s, r, h, tie_ev) & (t > lock)
        s += np.where(a, r, 0.0)
        lock = np.where(a, t + L, lock)
    return s


def play_ev(rw):
    N, TT = rw.shape
    s = np.zeros(N)
    lock = np.zeros(N, np.int32)
    for t in range(1, TT + 1):
        r = rw[:, t - 1]
        a = (r >= GAP[TT - t + 1]) & (t > lock)
        s += np.where(a, r, 0.0)
        lock = np.where(a, t + L, lock)
    return s
