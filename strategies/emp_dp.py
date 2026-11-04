"""Baseline round 2: plug-in empirical DP.
Every reward is observed (even while locked). Keep a sorted history, and every K rounds recompute the
optimal-stopping cutoff for the remaining horizon using the empirical distribution as if it were true.
Before MIN_N observations: accept nothing (just learn)."""
import bisect

K = 10
MIN_N = 20
_x = []
_suf = None
_cut = [None]
_last = [-10**9]


def _excess(c, x, suf, n):
    i = bisect.bisect_right(x, c)
    return (suf[i] - c * (n - i)) / n


def _recompute(s, G):
    x = _x
    n = len(x)
    suf = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suf[i] = suf[i + 1] + x[i]
    V = [0.0] * (s + 1)
    c = 0.0
    for k in range(1, s + 1):
        c = V[k - 1] - (V[k - G] if k >= G else 0.0)
        V[k] = V[k - 1] + _excess(c, x, suf, n)
    return c


def play(reward, lockout, t, T, your_score, other_scores, is_locked):
    bisect.insort(_x, reward)
    if is_locked or len(_x) < MIN_N:
        return False
    s = T - t + 1
    if _cut[0] is None or t - _last[0] >= K:
        _cut[0] = _recompute(s, lockout + 1)
        _last[0] = t
    return reward >= _cut[0] and reward > 0
