# Finite-horizon dynamic-programming strategy for the accept/skip lockout game.
#
# With n rounds left (including the current one) and a lockout of L:
#     V(n) = E_r[ max( accept value, V(n-1) ) ]
#     accept value = q(r) * r + [q(r) + (1 - q(r)) * lf] * V(n-1-L) + (1 - q(r)) * (1 - lf) * V(n-1)
# where q(r) is the probability that an accepted reward is actually credited and lf is the
# probability that a failed claim still locks you out. Under the standard rules q = 1, and a constant
# q never changes the optimal threshold, so the rule reduces to: accept iff r >= V(n-1) - V(n-1-L).
#
# Reward distribution: exponential (mean estimated online, or EXP_MEAN if announced) or the
# empirical distribution of every reward seen so far (rewards are observed while locked too).
# MODE = "auto" starts exponential and switches to empirical if a KS check rejects the fit.
#
# Exclusive claims: other teams' score and lock changes reveal who claimed each round and what
# fraction of the reward each claimant received. From that we learn
#     p(r)  - probability a free rival claims a reward of a given percentile
#     w(k)  - average fraction received per claimant when k teams claim together
# and q(r) = sum_k Binom(k; F, p(r)) * w(k + 1), with F the number of rivals free this round.
# Under the standard rules w == 1, so this machinery switches itself off.

import math
from bisect import bisect_left, insort

import numpy as np

MODE = "auto"      # "exp" (rounds 1, 3), "empirical" (round 2), or "auto"
EXP_MEAN = None    # set to the announced exponential mean if known, else estimated online
WARMUP = 8         # rounds observed before the first possible accept when the mean is unknown
REFIT_EVERY = 50   # rounds between DP refits (also refit when inputs move materially)
GRID = 256         # quantile grid size for the contention-aware DP
N_BINS = 20        # reward-percentile bins for the rival claim model
DECAY = 0.995      # per-round decay of rival claim statistics (tracks time-varying thresholds)
F_EDGES = (0, 1, 2, 3, 5, 8, 12, 17)   # lower edges of free-rival buckets
JITTER = 0.08      # log-sd of the per-game threshold multiplier under contention


def _w_exp_closed_form(N, L):
    # Exponential(mean 1), q constant: W(n) = W(n-1) + exp(-(W(n-1) - W(n-1-L)))
    W = [0.0] * (N + 1)
    for n in range(1, N + 1):
        acc = W[n - 1 - L] if n - 1 - L > 0 else 0.0
        W[n] = W[n - 1] + math.exp(-(W[n - 1] - acc))
    return W


def _v_sorted(N, L, r_sorted):
    # Empirical distribution, constant q: O(log m) per step via sorted samples + suffix sums
    m = len(r_sorted)
    top = np.concatenate(([0.0], np.cumsum(r_sorted[::-1])))
    V = [0.0] * (N + 1)
    for n in range(1, N + 1):
        skip = V[n - 1]
        acc = V[n - 1 - L] if n - 1 - L > 0 else 0.0
        k = m - bisect_left(r_sorted, skip - acc)
        V[n] = (top[k] + k * acc + (m - k) * skip) / m
    return V


def _v_grid(N, L, r, q, lf):
    # General DP on an equally weighted grid of reward values with per-value q(r)
    V = np.zeros(N + 1)
    qr = q * r
    keep = q + (1.0 - q) * lf
    for n in range(1, N + 1):
        skip = V[n - 1]
        acc = V[n - 1 - L] if n - 1 - L > 0 else 0.0
        a = qr + keep * acc + (1.0 - keep) * skip
        V[n] = np.maximum(a, skip).mean()
    return V


class Solution:
    def __init__(self):
        self._w_cache = {}
        self._reset(None, None)

    # ------------------------------------------------------------------ state
    def _reset(self, T, L):
        self.T, self.L = T, L
        self.last_t = 0
        self.rw = {}                # round -> reward
        self.sorted_rw = []
        self.sum_rw = 0.0
        self.mode = MODE if MODE != "auto" else "exp"
        self.ks_checked_at = 0
        self.V = None
        self.V_key = None
        self.fit_t = -10**9
        self.fit_mu = None
        # own claims
        self.my_claim = None        # (round, score_before, reward)
        self.my_claims = {}         # round -> fraction received
        self.lf_lock = 1.0          # pseudo-counts: failed claim still locks
        self.lf_free = 0.0
        # rivals
        self.snaps = {}             # k -> (scores ndarray, locked ndarray), state at round k
        self.conv_votes = [0.0, 0.0]  # evidence for lock-flag shift s = 0 / 1
        self.free_cnt = np.full(N_BINS, 5.0)    # decayed free-rival counts per reward bin
        self.claim_cnt = np.full(N_BINS, 0.5)   # decayed rival claims per reward bin
        self.round_cnt = np.full(N_BINS, 1.0)   # decayed rounds per reward bin
        self.anygain_cnt = np.full(N_BINS, 0.1) # decayed rounds where some rival was credited
        self.losers_seen = 0                    # rivals locked without being credited
        self.gain_rounds = 0
        self.multi_gain_rounds = 0              # rounds with 2+ credited teams (standard rules)
        self.q_obs = self.q_pred = 0.0
        self.q_cal = 1.0
        # Counterfactual "would we have been credited" table by (free-rival bucket, reward bin)
        self.cf_sum = np.zeros((len(F_EDGES), N_BINS))
        self.cf_n = np.zeros((len(F_EDGES), N_BINS))
        # Per-game threshold jitter under contention, so identical strategies do not move in lockstep
        self.jitter = float(np.exp(np.random.default_rng().normal(0.0, JITTER)))
        self.processed = set()
        self.F_avg = None
        self.contention = False

    # --------------------------------------------------------- distribution
    def _mu(self):
        if EXP_MEAN is not None:
            return float(EXP_MEAN)
        n = len(self.sorted_rw)
        return self.sum_rw / n if n else 1.0

    def _maybe_ks(self):
        n = len(self.sorted_rw)
        if MODE != "auto" or n < 20 or n - self.ks_checked_at < (10 if n < 300 else 50):
            return
        self.ks_checked_at = n
        x = np.asarray(self.sorted_rw)
        mu = self._mu()
        if x[0] < 0 or mu <= 0:
            self.mode = "empirical"
            return
        cdf = 1.0 - np.exp(-x / mu)
        i = np.arange(1, n + 1)
        d = max(np.max(i / n - cdf), np.max(cdf - (i - 1) / n))
        # Lilliefors (estimated mean) 1% critical value for the exponential ~ 1.3 / sqrt(n)
        self.mode = "empirical" if d * math.sqrt(n) > 1.3 else "exp"

    def _grid(self):
        if self.mode == "exp":
            u = (np.arange(GRID) + 0.5) / GRID
            return -self._mu() * np.log1p(-u)
        x = np.asarray(self.sorted_rw)
        if len(x) <= GRID:
            return x
        idx = ((np.arange(GRID) + 0.5) * len(x) / GRID).astype(int)
        return x[idx]

    def _pct(self, r):
        n = len(self.sorted_rw)
        return bisect_left(self.sorted_rw, r) / n if n else 0.5

    # ------------------------------------------------------------ rivals
    def _p_claim(self):
        if self.losers_seen >= 3:
            # Failed claimants are locked and visible: count claims directly
            p = self.claim_cnt / self.free_cnt
        else:
            # Only winners are visible: invert P(some rival claims) = 1 - (1 - p)^F
            h = np.clip(self.anygain_cnt / self.round_cnt, 0.0, 0.999)
            Fb = np.maximum(self.free_cnt / self.round_cnt, 1.0)
            p = 1.0 - (1.0 - h) ** (1.0 / Fb)
        return np.maximum.accumulate(np.clip(p, 0.0, 1.0))  # claims are monotone in reward

    def _q_model(self, arr, F):
        p = self._p_claim()
        n = max(len(self.sorted_rw), 1)
        bins = np.minimum((np.searchsorted(self.sorted_rw, arr) / n * N_BINS).astype(int), N_BINS - 1)
        pk = np.maximum(p[bins], 1e-9)
        return (1.0 - (1.0 - pk) ** (F + 1)) / ((F + 1) * pk), bins

    def _q_of_r(self, r, F, marginal=False):
        """Probability our claim is credited.

        Binomial random-tie-break model, blended with the counterfactual table built from rounds
        where we could see how many rivals claimed (captures rivals that react to the free count).
        """
        arr = np.atleast_1d(np.asarray(r, float))
        if not self.contention or F <= 0:
            q = np.ones_like(arr)
        else:
            qm, bins = self._q_model(arr, F)
            if marginal:
                cs, cn = self.cf_sum.sum(0)[bins], self.cf_n.sum(0)[bins]
            else:
                fb = max(i for i, e in enumerate(F_EDGES) if F >= e)
                cs, cn = self.cf_sum[fb, bins], self.cf_n[fb, bins]
            q = (cs + 3.0 * qm) / (cn + 3.0)
            q = np.clip(q * self.q_cal, 1e-3, 1.0)
        return q if isinstance(r, np.ndarray) else float(q[0])

    def _observe_rivals(self, t, other_scores):
        if not other_scores:
            return
        sc = np.fromiter((s for s, _ in other_scores), float, len(other_scores))
        lk = np.fromiter((bool(l) for _, l in other_scores), bool, len(other_scores))
        k = t - 1
        self.snaps[k] = (sc, lk)
        self.snaps.pop(k - 4, None)
        prev = self.snaps.get(k - 1)
        if prev is None or len(prev[0]) != len(sc):
            return
        r = self.rw.get(k)
        # Lock-flag convention: winners of round k show a score jump of rw[k] in snapshot k.
        if r is not None and abs(r) > 1e-12:
            won = (sc - prev[0]) / r > 0.5
            if won.any():
                self.conv_votes[1] += float(np.sum(lk[won]))           # flag already set: s = 1
                self.conv_votes[0] += float(np.sum(~lk[won]))          # flag set one round later: s = 0
        s = 1 if self.conv_votes[1] >= self.conv_votes[0] else 0
        # Process round u whose claim set is known: claimed(u) = ~flag(u-s) & flag(u+1-s)
        u = k if s == 1 else k - 1
        if u in self.processed:
            return
        a, b, c = self.snaps.get(u - s), self.snaps.get(u + 1 - s), self.snaps.get(u - 1)
        ru = self.rw.get(u)
        cur = self.snaps.get(u)
        if a is None or b is None or c is None or cur is None or ru is None or abs(ru) < 1e-12:
            return
        self.processed.add(u)
        free = ~a[1]
        frac = (cur[0] - c[0]) / ru
        gain = frac > 0.05
        claimed = (free & b[1]) | gain
        losers = claimed & ~gain
        F, K, G = int(free.sum()), int(claimed.sum()), int(gain.sum())
        self.F_avg = F if self.F_avg is None else 0.98 * self.F_avg + 0.02 * F
        bin_ = min(int(self._pct(ru) * N_BINS), N_BINS - 1)
        for arr in (self.free_cnt, self.claim_cnt, self.round_cnt, self.anygain_cnt):
            arr *= DECAY
        self.free_cnt[bin_] += F
        self.claim_cnt[bin_] += K
        self.round_cnt[bin_] += 1
        self.anygain_cnt[bin_] += G > 0
        mine = self.my_claims.pop(u, None)
        if self.losers_seen >= 3:
            fb = max(i for i, e in enumerate(F_EDGES) if F >= e)
            self.cf_sum *= DECAY
            self.cf_n *= DECAY
            self.cf_sum[fb, bin_] += 1.0 / (K + 1.0)
            self.cf_n[fb, bin_] += 1.0
        if G > 0 and (losers.any() or mine is not None and mine < 0.5):
            self.losers_seen += int(losers.sum())
        if G >= 1 or (mine is not None and mine > 0.5):
            self.gain_rounds += 1
            if G + (1 if mine is not None and mine > 0.5 else 0) >= 2:
                self.multi_gain_rounds += 1
        if mine is not None:
            # calibrate our credited rate against the model's prediction
            self.q_obs += mine
            self.q_pred += self._q_of_r(ru, F) if self.contention else 1.0
        self.contention = self.gain_rounds >= 12 and self.multi_gain_rounds < 0.1 * self.gain_rounds
        if self.contention and self.q_pred >= 5:
            self.q_cal = float(np.clip((self.q_obs + 2.0) / (self.q_pred + 2.0), 0.5, 2.0))

    def _free_now(self, t):
        s = 1 if self.conv_votes[1] >= self.conv_votes[0] else 0
        snap = self.snaps.get(t - 1)
        if snap is None:
            return self.F_avg or 0.0
        if s == 1:
            return float((~snap[1]).sum())
        return self.F_avg if self.F_avg is not None else float((~snap[1]).sum())

    # ------------------------------------------------------------- decide
    def _threshold_gap(self, n):
        acc = self.V[n - 1 - self.L] if n - 1 - self.L > 0 else 0.0
        return self.V[n - 1] - acc

    def _refit(self, t):
        N = self.T - t + 1
        L = self.L
        mu = self._mu()
        lf = self.lf_lock / (self.lf_lock + self.lf_free)
        if self.contention:
            r = self._grid()
            q = self._q_of_r(r, self.F_avg or 0.0, marginal=True)
            self.V = _v_grid(N, L, r, q, lf)
        elif self.mode == "exp":
            key = (L,)
            W = self._w_cache.get(key)
            if W is None or len(W) < N + 1:
                W = _w_exp_closed_form(max(N, self.T or N), L)
                self._w_cache[key] = W
            self.V = W
        else:
            self.V = _v_sorted(N, L, np.asarray(self.sorted_rw))
        self.fit_t, self.fit_mu, self.fit_mode, self.fit_cont = t, mu, self.mode, self.contention

    def play(self, reward, lockout, t, T, your_score, other_scores, is_locked):
        reward = float(reward)
        if t == 1 or t <= self.last_t or T != self.T or lockout != self.L:
            self._reset(T, lockout)
        self.last_t = t

        # Outcome of our own previous claim
        if self.my_claim is not None and t == self.my_claim[0] + 1:
            u, s0, r0 = self.my_claim
            frac = (your_score - s0) / r0 if abs(r0) > 1e-12 else 1.0
            self.my_claims[u] = float(np.clip(frac, 0.0, 1.0))
            if frac < 0.5 and self.L > 0:
                if is_locked:
                    self.lf_lock += 1
                else:
                    self.lf_free += 1
            self.my_claim = None

        self._observe_rivals(t, other_scores)

        self.rw[t] = reward
        insort(self.sorted_rw, reward)
        self.sum_rw += reward
        self.rw.pop(t - 6, None)
        self._maybe_ks()

        if is_locked or t > T:
            return False
        n = T - t + 1
        if self.L <= 0:
            return reward > 0
        if EXP_MEAN is None and t <= WARMUP and n > self.L + 1:
            return False

        mu = self._mu()
        stale = (self.V is None or self.fit_mode != self.mode or self.fit_cont != self.contention
                 or (self.mode == "empirical" and t - self.fit_t >= REFIT_EVERY)
                 or (self.contention and t - self.fit_t >= REFIT_EVERY))
        if stale:
            self._refit(t)
        # V is indexed by rounds remaining; the exponential table is shared and scaled by mu
        scale = mu if self.fit_mode == "exp" and not self.contention else 1.0
        gap = self._threshold_gap(n) * scale

        if self.contention:
            q = self._q_of_r(reward, self._free_now(t))
            lf = self.lf_lock / (self.lf_lock + self.lf_free)
            accept = q * reward >= (q + (1 - q) * lf) * gap * self.jitter
        else:
            accept = reward >= gap

        if accept:
            self.my_claim = (t, float(your_score), reward)
        return bool(accept)


play = Solution().play
