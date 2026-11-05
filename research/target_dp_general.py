"""P(score > B) dynamic program for an arbitrary reward distribution (generalises target_dp.py).

State: n rounds left, d = B - score still needed, on a grid d = i*h, i = 0..D (i <= 0: reached).
Rewards are rounded to the same grid, p_j = P(round(r/h) = j); j <= 0 is never worth accepting
(it cannot reduce d and it costs L rounds), so only p_1..p_J matter and the rest is "skip" mass.
    W_0(i)  = 1[i <= 0]
    W_n(i)  = sum_j p_j max( W_{n-1}(i), W_{n-1-L}(i - j) )
W is non-increasing in i, so accepting j is worth it iff j >= j*(i) = i - c(i) + 1, where
c(i) = #{k >= 0 : A(k) > S(i)} (binary search). The truncated convolution sum_{j >= j*} p_j A(i-j)
is a reversed cumulative sum over a (D+1, J) sliding window of A, so each step is O(D J) numpy work.

The table does not depend on B: B only fixes the starting state d = B, so one table per
distribution answers every target up to D*h.

`uv run python research/target_dp_general.py` checks the predicted W against Monte Carlo.
"""
import math
import sys
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

T, L = 2000, 15


# ----------------------------------------------------------------------------- distributions
def _ndtri(u):
    # inverse normal CDF (Acklam), |rel err| < 1.2e-9; numpy only (the server has no scipy)
    u = np.asarray(u, float)
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    lo, hi = 0.02425, 1 - 0.02425
    x = np.empty_like(u)
    m = (u >= lo) & (u <= hi)
    q = u[m] - 0.5
    r = q * q
    x[m] = ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1))
    for sel, sgn, uu in ((u < lo, 1.0, u), (u > hi, -1.0, 1 - u)):
        q = np.sqrt(-2 * np.log(uu[sel]))
        x[sel] = sgn * (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    return x


def eth_tail(z):
    """N(0,1) with the upper tail compressed (slope 0.938 above z = 0.55).

    Fitted to the ETH round-2 mined quantiles: with mean -0.19 and sd 0.97 it gives q95 = 1.34 and
    q99 = 1.96 (a plain normal gives 1.40 and 2.07).
    """
    return np.where(z > 0.55, 0.55 + 0.938 * (z - 0.55), z)


class Dist:
    """Reward distribution: quantile function q(u) plus an optional per-run location shift sd."""

    def __init__(self, name, qf, run_sd=0.0):
        self.name, self.qf, self.run_sd = name, qf, run_sd

    def quantiles(self, m=20000, shift=0.0):
        return self.qf((np.arange(m) + 0.5) / m) + shift

    def sample(self, g, size, shift=0.0):
        return self.qf(np.clip(g.random(size), 1e-12, 1 - 1e-12)) + shift


def _q_bimodal(u):
    # 0.7 N(-1, 0.5) + 0.3 N(2, 0.7): invert the mixture CDF on a fine table
    xs = np.linspace(-5, 6, 20001)
    z1, z2 = (xs + 1) / 0.5, (xs - 2) / 0.7
    cdf = 0.7 * 0.5 * (1 + _erf(z1 / math.sqrt(2))) + 0.3 * 0.5 * (1 + _erf(z2 / math.sqrt(2)))
    return np.interp(u, cdf, xs)


def _erf(x):
    s = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
               + 0.254829592) * t * np.exp(-a * a)
    return s * y


DISTS = {
    "eth": Dist("eth: -0.19+0.97*lighttail(Z), run mean sd 0.1",
                lambda u: -0.19 + 0.97 * eth_tail(_ndtri(u)), run_sd=0.10),
    "normal": Dist("normal: N(0,1)", lambda u: _ndtri(u)),
    "lognormal": Dist("lognormal: exp(N(0,0.75)) - 1.3", lambda u: np.exp(0.75 * _ndtri(u)) - 1.3),
    "uniform": Dist("uniform: U(-2,3)", lambda u: -2 + 5 * u),
    "bimodal": Dist("bimodal: 0.7N(-1,.5)+0.3N(2,.7)", _q_bimodal),
    "normal02": Dist("normal02: N(-0.2,1) (public engine 'b')", lambda u: -0.2 + _ndtri(u)),
    "pareto3": Dist("pareto3: numpy pareto(3) = Lomax, (1-u)^(-1/3) - 1", lambda u: (1 - u) ** (-1 / 3) - 1),
    "exp10": Dist("exp10: Exp(mean 10)", lambda u: -10 * np.log1p(-u)),
}


# ----------------------------------------------------------------------------- dynamic programs
def pmf_on_grid(samples, h, weights=None, tail=1e-9):
    """p[j] = P(round(r/h) = j) for j = 1..J (index 0 holds the mass of j <= 0, never accepted)."""
    s = np.asarray(samples, float)
    w = np.ones_like(s) if weights is None else np.asarray(weights, float)
    j = np.rint(s / h).astype(np.int64)
    pos = j >= 1
    J = int(j[pos].max()) if pos.any() else 1
    p = np.bincount(np.where(pos, j, 0), weights=w, minlength=J + 1) / w.sum()
    # drop the far tail that cannot move W by more than `tail`
    c = np.cumsum(p[::-1])[::-1]
    J = max(1, int(np.nonzero(c > tail)[0].max())) if (c > tail).any() else 1
    p = p[:J + 1].copy()
    p[0] = 1.0 - p[1:].sum()
    return p


def target_table(p, D, T=T, L=L):
    """W (T+1, D+1) float32: W[n, i] = max P(score gain > i*h) with n rounds left."""
    J = len(p) - 1
    pj = p[1:][::-1]                                   # column m of the window is j = J - m
    tailP = np.concatenate((np.cumsum(p[1:][::-1])[::-1], [0.0]))   # tailP[j-1] = P(j' >= j)
    W = np.zeros((T + 1, D + 1), np.float32)
    W[0, 0] = 1.0
    idx = np.arange(D + 1)
    rows = idx
    ones = np.ones(J)
    for n in range(1, T + 1):
        S = W[n - 1].astype(np.float64)
        A = W[max(n - 1 - L, 0)].astype(np.float64)
        win = sliding_window_view(np.concatenate((ones, A[:-1])), J)  # win[i, m] = A(i - (J - m))
        C = np.cumsum(win * pj, axis=1)                # C[i, m] = sum_{j >= J - m} p_j A(i - j)
        cnt = np.searchsorted(-A, -S, side="left")     # #{k : A(k) > S(i)}, A non-increasing
        jstar = np.maximum(idx - cnt + 1, 1)
        ok = jstar <= J
        tail_v = np.where(ok, C[rows, np.clip(J - jstar, 0, J - 1)], 0.0)
        tail_p = tailP[np.minimum(jstar, J) - 1] * ok
        Wn = S * (1.0 - tail_p) + tail_v
        Wn[0] = 1.0
        W[n] = np.minimum.accumulate(np.minimum(Wn, 1.0))   # keep monotone against fp noise
    return W


def ev_gap_sorted(r_sorted, T=T, L=L):
    """Expected-value DP threshold per rounds-left n for an equally weighted sample (solution.py)."""
    r = np.asarray(r_sorted, float)
    m = len(r)
    top = np.concatenate(([0.0], np.cumsum(r[::-1])))
    V = np.zeros(T + 1)
    gap = np.zeros(T + 1)
    for n in range(1, T + 1):
        skip = V[n - 1]
        acc = V[n - 1 - L] if n - 1 - L > 0 else 0.0
        g = max(skip - acc, 0.0)
        gap[n] = g
        k = m - np.searchsorted(r, g, side="left")
        V[n] = (top[k] + k * acc + (m - k) * skip) / m
    return gap


def target_decide(W, h, n, d, r, ev_accept, L=L):
    """Vectorised target-DP decision. d = B - score, r = reward, ev_accept = EV-DP fallback.

    The half-step offset maps d in [0, h) to i = 1, so i = 0 means strictly above B (on a coarse
    grid, counting ties as reached costs most of the gain).
    """
    D = W.shape[1] - 1
    d = d + 0.5 * h
    i = np.clip(np.rint(d / h).astype(np.int64), 0, D)
    Sv = W[n - 1, i]
    ia = np.rint((d - r) / h).astype(np.int64)
    Av = np.where(ia <= 0, 1.0, W[max(n - 1 - L, 0), np.clip(ia, 0, D)])
    decided = (i > 0) & (np.maximum(Av, Sv) > 1e-9)
    return np.where(decided, (Av > Sv + 1e-12) & (r > 0), ev_accept)


# ----------------------------------------------------------------------------- simulation
def simulate(rw, policy):
    """rw (N, T) rewards, policy(n, score, r) -> bool array. Returns final scores (N,)."""
    N, TT = rw.shape
    score = np.zeros(N)
    lock = np.zeros(N, np.int64)
    for t in range(1, TT + 1):
        r = rw[:, t - 1]
        a = policy(TT - t + 1, score, r) & (t > lock)
        score += np.where(a, r, 0.0)
        lock = np.where(a, t + L, lock)
    return score


def grid_step(samples, B_max, work=1.5e5):
    """h ~ sd / 10 (on N(0,1) within MC noise of sd / 20 at a quarter of the cost), but coarse
    enough that D * J = B_max * r_max / h^2 stays under `work` (about 1 s for T = 2000)."""
    s = np.asarray(samples, float)
    r_max = max(float(np.quantile(s, 1 - 1e-4)), 1e-9)
    return max(float(np.std(s)) / 10.0, math.sqrt(B_max * r_max / work))


def validate(name, N=20000, seed=0):
    dist = DISTS[name]
    g = np.random.default_rng(seed)
    q = dist.quantiles(40000)
    gap = ev_gap_sorted(q)
    rw = dist.sample(g, (N, T))                 # no run shift here: the table assumes a known law
    ev = simulate(rw, lambda n, s, r: r >= gap[n])
    B_list = [np.quantile(ev, x) for x in (0.9, 0.99)]
    h = grid_step(q, max(B_list) * 1.2)
    D = int(max(B_list) * 1.2 / h) + 2
    t0 = time.time()
    p = pmf_on_grid(q, h)
    W = target_table(p, D)
    tb = time.time() - t0
    print(f"[{name}] EV-DP mean {ev.mean():.2f} sd {ev.std():.2f}   h={h:.4f} D={D} J={len(p)-1} "
          f"table {tb:.2f}s", flush=True)
    for B in B_list:
        sc = simulate(rw, lambda n, s, r: target_decide(W, h, n, B - s, r, r >= gap[n]))
        pe, pt = (ev > B).mean(), (sc > B).mean()
        print(f"   B={B:8.2f}: W predicted {W[T, int(np.rint(B / h + 0.5))]:.4f}  MC target {pt:.4f} "
              f"(+-{math.sqrt(pt * (1 - pt) / N):.4f})  MC EV-DP {pe:.4f}  x{pt / max(pe, 1e-9):.2f}  "
              f"mean {sc.mean():.2f}", flush=True)


if __name__ == "__main__":
    names = sys.argv[1:] or ["eth", "normal", "lognormal", "uniform", "bimodal", "exp10"]
    for nm in names:
        validate(nm)
