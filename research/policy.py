"""Finite-horizon DP policy for the accept/skip lockout game.

Model: N turns; when free, accept -> reward added with prob q (contention / "may be added"),
then locked for LOCK turns. With n turns left (incl. current):
    V(n) = E_r[ max(q*r + V(n-1-LOCK), V(n-1)) ],   accept iff q*r >= V(n-1) - V(n-1-LOCK)

Modes:
  * known exponential (mean mu): closed form, E[(r-th)+] = mu*exp(-th/mu)
      V(n) = V(n-1) + q*mu*exp(-(V(n-1)-V(n-1-L))/(q*mu))
  * hidden distribution: empirical DP on every reward seen so far (rewards are visible while
    locked too), refit periodically.
q is learned online from score changes after our accepts (Beta-style pseudo-counts).
"""
from __future__ import annotations
import numpy as np


def dp_exponential(N, L, mu, q=1.0):
    V = np.zeros(N + 1)
    for n in range(1, N + 1):
        acc = V[n - 1 - L] if n - 1 - L > 0 else 0.0
        V[n] = V[n - 1] + q * mu * np.exp(-(V[n - 1] - acc) / (q * mu))
    return V


def dp_empirical(N, L, samples, q=1.0):
    r = np.sort(np.asarray(samples, float)); m = len(r)
    top = np.concatenate([[0.0], np.cumsum(r[::-1])])       # top[k] = sum of k largest
    V = np.zeros(N + 1)
    for n in range(1, N + 1):
        skip = V[n - 1]; acc = V[n - 1 - L] if n - 1 - L > 0 else 0.0
        k = m - np.searchsorted(r, (skip - acc) / q, side="left")
        V[n] = (q * top[k] + k * acc + (m - k) * skip) / m
    return V


class LockoutPolicy:
    def __init__(self, n_turns=2000, lock=15, exp_mean=None, warmup=20, window=None,
                 refit_every=50, q_prior=(2.0, 2.0), learn_q=True):
        self.N, self.L, self.mu = n_turns, lock, exp_mean
        self.warmup, self.window, self.refit_every = warmup, window, refit_every
        self.q_num, self.q_den = q_prior           # prior mean 1.0
        self.learn_q = learn_q
        self.rewards, self.t = [], 0
        self.locked_until, self.pending = -1, None
        self.V, self._fit_t, self._fit_q = None, -10**9, None
        if exp_mean is not None:
            self.warmup = 0

    @property
    def q(self):
        return min(1.0, max(1e-3, self.q_num / self.q_den))

    def _refit(self):
        if self.mu is not None:
            self.V = dp_exponential(self.N, self.L, self.mu, self.q)
        else:
            s = self.rewards[-self.window:] if self.window else self.rewards
            self.V = dp_empirical(self.N, self.L, s, self.q)
        self._fit_t, self._fit_q = self.t, self.q

    def threshold(self, n):
        acc = self.V[n - 1 - self.L] if n - 1 - self.L > 0 else 0.0
        return (self.V[n - 1] - acc) / self.q

    def decide(self, reward, score=0.0, locked=None):
        t = self.t; self.t += 1
        if self.pending is not None:               # score now reflects our last accept
            s0, r0 = self.pending; self.pending = None
            if self.learn_q and abs(r0) > 1e-12:
                self.q_num += float(np.clip((score - s0) / r0, 0, 1)); self.q_den += 1
        self.rewards.append(float(reward))
        is_locked = locked if locked is not None else t <= self.locked_until
        if is_locked or t < self.warmup or t >= self.N:
            return False
        need = (self.V is None or (self.mu is None and t - self._fit_t >= self.refit_every)
                or abs(self.q - self._fit_q) > 0.02)
        if need:
            self._refit()
        if reward >= self.threshold(self.N - t):
            self.locked_until = t + self.L
            self.pending = (score, float(reward))
            return True
        return False
