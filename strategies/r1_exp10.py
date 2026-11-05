import math

# Round 1: rewards ~ Exponential(mean 10), every team that accepts gets the full reward,
# so other teams cannot change our score. Pure optimal stopping with lockout:
#   with s turns left (incl. this one), accept iff reward >= V(s-1) - V(s-G), G = lockout + 1
#   V(s) = V(s-1) + MEAN * exp(-cutoff(s) / MEAN)

MEAN = 10.0

_cache = {}


def _cutoffs(T, G):
    key = (T, G)
    c = _cache.get(key)
    if c is None:
        V = [0.0] * (T + 1)
        c = [0.0] * (T + 1)
        for s in range(1, T + 1):
            cs = V[s - 1] - (V[s - G] if s >= G else 0.0)
            c[s] = cs
            V[s] = V[s - 1] + MEAN * math.exp(-cs / MEAN)
        _cache[key] = c
    return c


def play(reward, lockout, t, T, your_score, other_scores, is_locked):
    try:
        if is_locked or reward <= 0:
            return False
        T = max(int(T), 1)
        G = max(int(lockout), 0) + 1
        s = min(max(T - int(t) + 1, 1), T)
        return bool(reward >= _cutoffs(T, G)[s])
    except Exception:
        return bool(not is_locked and reward > 20.0)
