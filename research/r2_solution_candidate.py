# Round 2 candidate (hidden reward distribution, ranked on BEST score over runs).
#
# Every team sees the same rewards in a run, so teams running the expected-value DP finish together;
# the only way to rank strictly first is to beat the field's best in some run. With TARGET set,
# this maximises P(score > TARGET) instead of E[score]:
#     W_n(d) = E_r[ max( W_{n-1}(d), W_{n-1-L}(d - r) ) ],  W_0(d) = 1[d < 0],  d = TARGET - score
# solved on a grid (research/target_dp_general.py). When the target is reached or out of reach it
# falls back to the expected-value DP: accept iff r >= KAPPA * (V(n-1) - V(n-1-L)).
#
# Reward distribution:
#   PRIOR_Q set   - shape mined from earlier runs (research/r2_prior.py), shifted by this run's
#                   location estimated online (runs differ in mean); dropped if a KS test rejects it.
#   PRIOR_Q None  - empirical distribution of this run's rewards (kernel-smoothed for the target DP).
# Rewards are observed every round, including while locked. Never accepts a non-positive reward.

import math
from bisect import bisect_left, insort

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

TARGET = None        # score to beat (field best + margin); None = plain expected-value DP
KAPPA = 1.0          # multiplier on the expected-value threshold
PRIOR_Q = None       # sorted quantiles of the mined reward distribution, or None to learn online
SHIFT_N0 = 100.0     # pseudo-observations behind the prior location (per-run mean sd ~ 0.1 sd)
SHIFT_STEP = 0.01    # location estimate rounded to this many prior sds (tables are rebuilt on change)
KS_CRIT = 1.63       # 1% Kolmogorov-Smirnov critical value x sqrt(n) for dropping the prior
WARMUP = 8           # rounds observed before the first accept when no prior is set
REFIT_EVERY = 50     # rounds between expected-value refits
CHECK_PRIOR = (30, 150, 400, 800)  # target-table builds with a prior
CHECK_EMP = (200, 700)             # target-table builds without one
WORK = 1.5e5         # grid size budget D * J (about 1 s per table)
DEBUG = False        # re-raise exceptions (tests only); in a game any error means "skip"
KDE_Z = (-1.863, -1.318, -1.010, -0.776, -0.579, -0.402, -0.237, -0.078,
         0.078, 0.237, 0.402, 0.579, 0.776, 1.010, 1.318, 1.863)   # N(0,1) 16-quantiles


def _ev_gap(r_sorted, N, L):
    # Expected-value DP thresholds gap[n], n = rounds left, for an equally weighted sample
    r = np.asarray(r_sorted, float)
    m = len(r)
    top = np.concatenate(([0.0], np.cumsum(r[::-1])))
    V = [0.0] * (N + 1)
    gap = [0.0] * (N + 1)
    for n in range(1, N + 1):
        skip = V[n - 1]
        acc = V[n - 1 - L] if n - 1 - L > 0 else 0.0
        g = max(skip - acc, 0.0)
        gap[n] = g
        k = m - bisect_left(r, g) if m < 64 else m - int(np.searchsorted(r, g))
        V[n] = (top[k] + k * acc + (m - k) * skip) / m
    return gap


def _pmf(samples, h):
    # p[j] = P(round(r / h) = j), j = 1..J; p[0] = everything not worth accepting (j <= 0)
    j = np.rint(np.asarray(samples, float) / h).astype(np.int64)
    pos = j >= 1
    if not pos.any():
        return np.array([1.0, 0.0])
    p = np.bincount(np.where(pos, j, 0), minlength=int(j.max()) + 1) / len(j)
    c = np.cumsum(p[::-1])[::-1]
    J = max(1, int(np.nonzero(c > 1e-9)[0].max()))
    p = p[:J + 1].copy()
    p[0] = 1.0 - p[1:].sum()
    return p


def _target_table(p, D, N, L):
    # W[n, i] = max P(gain > (i - 1/2) h) with n rounds left; truncated convolution per step
    J = len(p) - 1
    pj = p[1:][::-1]
    tailP = np.concatenate((np.cumsum(p[1:][::-1])[::-1], [0.0]))
    W = np.zeros((N + 1, D + 1), np.float32)
    W[0, 0] = 1.0
    idx = np.arange(D + 1)
    ones = np.ones(J)
    for n in range(1, N + 1):
        S = W[n - 1].astype(np.float64)
        A = W[max(n - 1 - L, 0)].astype(np.float64)
        C = np.cumsum(sliding_window_view(np.concatenate((ones, A[:-1])), J) * pj, axis=1)
        cnt = np.searchsorted(-A, -S, side="left")
        jstar = np.maximum(idx - cnt + 1, 1)
        ok = jstar <= J
        tail_v = np.where(ok, C[idx, np.clip(J - jstar, 0, J - 1)], 0.0)
        tail_p = tailP[np.minimum(jstar, J) - 1] * ok
        Wn = S * (1.0 - tail_p) + tail_v
        Wn[0] = 1.0
        W[n] = np.minimum.accumulate(np.minimum(Wn, 1.0))
    return W


class Solution:
    def __init__(self):
        self.prior = None if PRIOR_Q is None else np.sort(np.asarray(PRIOR_Q, float))
        if self.prior is not None:
            self.p_mean, self.p_sd = float(self.prior.mean()), float(self.prior.std()) or 1.0
        self._reset(None, None)

    def _reset(self, T, L):
        self.T, self.L = T, L
        self.last_t = 0
        self.n_obs = 0              # rounds observed in an unbroken sequence from t = 1
        self.xs = []
        self.sum = 0.0
        self.use_prior = self.prior is not None
        self.ks_at = 0
        self.gap, self.gap_t, self.gap_key = None, -10**9, None
        self.W, self.W_key, self.h = None, None, None
        self.built = set()

    # ------------------------------------------------------------ distribution
    def _shift(self):
        s = (self.sum - self.n_obs * self.p_mean) / (self.n_obs + SHIFT_N0)
        step = SHIFT_STEP * self.p_sd
        return round(s / step) * step

    def _samples(self):
        if self.use_prior:
            return self.prior + self._shift()
        return np.asarray(self.xs)

    def _check_prior(self):
        n = len(self.xs)
        if not self.use_prior or n < 100 or n - self.ks_at < 50:
            return
        self.ks_at = n
        q = self.prior + self._shift()
        F = np.searchsorted(q, np.asarray(self.xs), side="right") / len(q)
        i = np.arange(1, n + 1)
        d = max(np.max(i / n - F), np.max(F - (i - 1) / n))
        if d * math.sqrt(n) > KS_CRIT:
            self.use_prior = False
            self.gap = self.W = None

    # ---------------------------------------------------------------- tables
    def _refit_ev(self, t):
        key = self._shift() if self.use_prior else None
        age = t - self.gap_t
        if self.gap is not None and age < REFIT_EVERY and (key == self.gap_key or age < 10):
            return
        self.gap = _ev_gap(self._samples(), self.T - t + 1, self.L)
        self.gap_t, self.gap_key = t, key

    def _maybe_build(self, t):
        checks = CHECK_PRIOR if self.use_prior else CHECK_EMP
        due = [c for c in checks if c <= t and (self.use_prior, c) not in self.built]
        if not due or self.n_obs != t or self.T - t < 2 * self.L:
            return
        for c in due:
            self.built.add((self.use_prior, c))
        if self.use_prior:
            key = self._shift()
            if self.W is not None and key == self.W_key:
                return
            s = self.prior + key
        else:
            x = np.asarray(self.xs)
            bw = 1.06 * x.std() * len(x) ** -0.2
            s, key = (x[:, None] + bw * np.asarray(KDE_Z)[None, :]).ravel(), None
        r_max = max(float(np.quantile(s, 1 - 1e-4)), 1e-9)
        B = max(float(TARGET), 1e-9)
        h = max(float(np.std(s)) / 10.0, math.sqrt(B * r_max / WORK), B / 4000.0)
        D = int(math.ceil(B / h)) + 2
        self.W = _target_table(_pmf(s, h), D, self.T - t + 1, self.L)
        self.W_key, self.h, self.W_N = key, h, self.T - t + 1

    def _target_accept(self, n, d, r, ev):
        W, h = self.W, self.h
        if W is None or n > self.W_N:
            return ev
        D = W.shape[1] - 1
        d = d + 0.5 * h
        i = min(max(int(round(d / h)), 0), D)
        if i <= 0:
            return ev                      # target already beaten: keep maximising the score
        Sv = float(W[n - 1, i])
        ia = int(round((d - r) / h))
        Av = 1.0 if ia <= 0 else float(W[max(n - 1 - self.L, 0), min(ia, D)])
        if max(Av, Sv) <= 1e-9:
            return ev                      # out of reach in this run
        return Av > Sv + 1e-12 and r > 0

    # ------------------------------------------------------------------ play
    def play(self, reward, lockout, t, T, your_score, other_scores, is_locked):
        try:
            reward = float(reward)
            if t == 1 or t <= self.last_t or T != self.T or lockout != self.L:
                self._reset(T, lockout)
            if t == self.last_t + 1 and self.n_obs == self.last_t:
                self.n_obs = t
            self.last_t = t
            if math.isfinite(reward):
                insort(self.xs, reward)
                self.sum += reward
            self._check_prior()
            if is_locked or t > T or not math.isfinite(reward) or reward <= 0:
                return False
            if self.L <= 0:
                return True
            if not self.use_prior and t <= WARMUP and T - t > self.L:
                return False
            n = T - t + 1
            self._refit_ev(t)
            ev = reward >= KAPPA * self.gap[n]
            if TARGET is None:
                return bool(ev)
            self._maybe_build(t)
            return bool(self._target_accept(n, float(TARGET) - float(your_score), reward, ev))
        except Exception:
            if DEBUG:
                raise
            return False


play = Solution().play
