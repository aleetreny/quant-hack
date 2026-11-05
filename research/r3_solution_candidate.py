# Round 3 (exclusive claims, collision pays nobody) strategy. Order-invariant, adapts in-game.
#
# Rules assumed: rewards Exponential(mean 10), T = 2000, L = 15. If exactly one team accepts it is
# credited; if two or more accept nobody is credited and all of them are locked for L rounds.
#
# 1. Rival tracking without list positions. A team's score only changes when it wins, so the score
#    value itself is its identity. Each turn the multiset of (score, locked) is matched to the
#    previous one: the one score that moved is the winner, and inside each score group the number
#    of flags that turned on (correcting for locks that expire after L flags) is the number of new
#    claimants. Teams sharing a score (all zero-score teams, identical clones) are interchangeable.
# 2. Claim model. For every free rival we know how long it has been free. Greedy teams claim the
#    moment they unlock; pickier teams wait. We learn p_c(r) = P(free rival of free-age class c
#    claims reward r) with exponential forgetting, and predict for the current free set
#        S(r) = P(no rival claims) = prod_c (1 - p_c(r)) ** n_c.
# 3. Threshold. The value of the L rounds we give up comes from a DP on the realised no-claim rate
#    per reward bin (how often a claim would have been alone), stationary mid-game, exact near the
#    end, blended with L * (our own recent earnings per round). Accept iff S(r) * r >= gap * jitter.
# 4. Anti-collision. S is calibrated against what rivals actually did in the rounds where we
#    wanted to claim; acceptance is a smooth probability whose softness grows with the measured
#    correlation; after a collision we back off a random number of free rounds. Randomisation is
#    off when nobody contends (S ~ 1), so alone it is the exact round-1 DP.
# 5. Lock-flag convention (flag at t-1 pre- or post-action) is detected online.

import math
import random

import numpy as np

EXP_MEAN = 10.0     # announced exponential mean (None -> estimate online)


class Solution:
    # tunables (instance attributes so the arena can override them per player)
    BW = 1.0            # reward bin width (in units of mean/10)
    NB = 40             # number of reward bins (last one is the tail)
    DECAY = 0.997       # per-round forgetting of claim statistics
    PRIOR_P = 0.15      # prior claim probability of a free rival
    PRIOR_N = 2.0       # prior weight (trials)
    NEIGH_K = 3.0       # shrinkage of a bin towards its neighbours
    FA_EDGES = (1, 4)   # free-age classes: 0 | 1..3 | 4+
    REFIT = 25          # rounds between threshold refits
    TAIL = 160          # rounds handled by the exact finite-horizon DP
    JITTER = 0.08       # log-sd of the per-game threshold multiplier
    TAU = 0.15          # softness of the accept rule (relative to the gap); 0 = deterministic
    TAU_CORR = 1.5      # extra softness per unit of measured correlation with rivals
    P_MAX = 0.95        # max accept probability (random skips de-synchronise copies)
    CROWD = 2.0         # contention (1 - S) at which randomisation is fully on is 1 / CROWD
    CORR = True         # correct S for correlation with rivals that want the same rewards as us
    CORR_A = 10.0       # prior weight (rounds) of the correlation correction
    REGION = 5          # bins per correlation region
    BACKOFF = 4         # after a collision skip U{0..BACKOFF * 2^(streak-1)} free rounds (capped)
    BACKOFF_CAP = 16
    EMP_W = 0.6         # max weight of the empirical gap L * (recent own earnings per round)
    EMP_RAMP = 300.0    # rounds until the empirical gap gets half of EMP_W
    EMP_DECAY = 0.999   # forgetting of the own earnings rate
    WARMUP = 5          # rounds observed before the first possible accept

    def __init__(self):
        self.rng = random.Random()
        self._reset(None, None)

    # ------------------------------------------------------------------ state
    def _reset(self, T, L):
        self.T, self.L = T, L
        self.last_t = 0
        self.rw = {}
        self.rsum, self.rn = 0.0, 0
        nc = len(self.FA_EDGES) + 1
        self.cl_n = np.zeros((nc, self.NB))     # decayed free-rival trials
        self.cl_k = np.zeros((nc, self.NB))     # decayed claims
        self.z_n = np.zeros(self.NB)            # decayed rounds per bin
        self.z_k = np.zeros(self.NB)            # decayed rounds where no rival claimed
        nr = (self.NB + self.REGION - 1) // self.REGION
        self.w_n = np.zeros(nr)                 # rounds where we would claim: model-predicted S
        self.w_k = np.zeros(nr)                 # ... and realised "no rival claimed"
        self.w_hist = {}
        self.f_reg = np.ones(nr)
        # slots: parallel lists (score, lock age (0 = free), free age, won-flag)
        self.s_sc, self.s_la, self.s_fa, self.s_won = [], [], [], []
        self.valid_from = 10**9
        self.post_votes, self.pre_votes = 0.0, 0.0
        self.al = [0.0, 0.0, 0.0]               # new locks, and their summed reward under post / pre
        self.bad = 0
        self.gap_c = None
        self.gap_tail = None
        self.fit_t = -10**9
        self.jit = math.exp(self.rng.gauss(0.0, self.JITTER))
        self.my_claim = None
        self.own_n = self.own_w = 0
        self.streak = 0
        self.backoff = 0
        self.own_res = {}                       # round -> our claim was credited
        self.earn = 0.0                         # decayed own earnings, and decayed round count
        self.earn_n = 0.0
        self.prev_score = None
        self.own_pred = 0.0
        self.own_raw = 0.0                      # model S without the correlation correction

    def _mu(self):
        if EXP_MEAN is not None:
            return EXP_MEAN
        return self.rsum / self.rn if self.rn >= 30 else 10.0

    def _bin(self, r):
        b = int(r * 10.0 / (self._mu() * self.BW))
        return 0 if b < 0 else (b if b < self.NB else self.NB - 1)

    def _cls(self, fa_round):
        c = 0
        for e in self.FA_EDGES:
            if fa_round >= e:
                c += 1
        return c

    # ------------------------------------------------------------------ tracking
    def _init_slots(self, obs, fresh):
        self.s_sc = [s for s, _ in obs]
        self.s_la = [1 if f else 0 for _, f in obs]
        self.s_fa = [0 if f else (1 if fresh else 5) for _, f in obs]
        self.s_won = [0] * len(obs)

    def _observe(self, t, obs):
        """Match snapshot t-1 to the slots (state at t-2); update claim statistics."""
        k = t - 1
        if not obs:
            return
        if len(obs) != len(self.s_sc):
            self._init_slots(obs, fresh=(k <= 1))
            self.valid_from = k + 1 if k <= 1 else k + self.L
            return
        L = self.L
        # groups of the new snapshot
        new_g = {}
        for s, f in obs:
            g = new_g.get(s)
            if g is None:
                new_g[s] = [1, 1 if f else 0]
            else:
                g[0] += 1
                if f:
                    g[1] += 1
        old_g = {}
        for i, s in enumerate(self.s_sc):
            old_g.setdefault(s, []).append(i)
        departed, arrivals = [], []
        for s, ids in old_g.items():
            need = new_g[s][0] if s in new_g else 0
            if len(ids) > need:
                # pick the departing (winning) slots: eligible (free or expiring), most eager first
                el = sorted(ids, key=lambda i: (0 if self.s_la[i] == 0 or self.s_la[i] >= L else 1,
                                                0 if self.s_la[i] >= L else self.s_fa[i]))
                departed.extend(el[:len(ids) - need])
        for s, g in new_g.items():
            have = len(old_g.get(s, ()))
            if g[0] > have:
                arrivals.extend([s] * (g[0] - have))
        if len(departed) != len(arrivals) or len(departed) > 3:
            self.bad += 1
            self._init_slots(obs, fresh=False)
            self.valid_from = k + L
            return
        if departed:
            departed.sort(key=lambda i: self.s_sc[i])
            arrivals.sort()
            for i, s in zip(departed, arrivals):
                self.s_sc[i] = s
                self.s_won[i] = 2
        # per group: count new locks
        mem = {}
        for i, s in enumerate(self.s_sc):
            mem.setdefault(s, []).append(i)
        free_prev = [i for i in range(len(self.s_sc)) if self.s_la[i] == 0]
        newly = []
        for s, ids in mem.items():
            nl = new_g[s][1]
            cont = [i for i in ids if 0 < self.s_la[i] < L]
            cand = [i for i in ids if self.s_la[i] == 0]
            m = nl - len(cont)
            if m < 0 or m > len(cand):
                self.bad += 1
                m = max(0, min(m, len(cand)))
            if m:
                cand.sort(key=lambda i: (-self.s_won[i], self.s_fa[i]))
                newly.extend(cand[:m])
        # convention votes from winners moved this snapshot
        nset = set(newly)
        for i in departed:
            if i in nset:
                self.post_votes += 1
            else:
                self.pre_votes += 1
        # alignment evidence: claims depend on the reward, so under the right convention the rewards
        # matched to new locks are further from the mean than under the wrong one
        if newly:
            r0, r1 = self.rw.get(k), self.rw.get(k - 1)
            if r0 is not None and r1 is not None:
                self.al[0] += len(newly)
                self.al[1] += len(newly) * r0
                self.al[2] += len(newly) * r1
        post = self._post()
        # statistics for the round these claims belong to
        if k >= self.valid_from:
            u = k if post else k - 1
            r = self.rw.get(u)
            if r is not None:
                b = self._bin(r)
                d = self.DECAY
                self.cl_n *= d
                self.cl_k *= d
                self.z_n *= d
                self.z_k *= d
                self.w_n *= d
                self.w_k *= d
                for i in free_prev:
                    c = self._cls(self.s_fa[i] - 1)
                    self.cl_n[c, b] += 1.0
                    if i in nset:
                        self.cl_k[c, b] += 1.0
                self.z_n[b] += 1.0
                if not newly:
                    self.z_k[b] += 1.0
                w = self.w_hist.get(u)
                if w is not None:
                    g = b // self.REGION
                    self.w_n[g] += w
                    # our own outcome is the ground truth when we did claim (robust to flags)
                    alone = self.own_res.get(u, not newly)
                    if alone:
                        self.w_k[g] += 1.0
        # advance slot states
        for i in range(len(self.s_sc)):
            la = self.s_la[i]
            if i in nset:
                self.s_la[i], self.s_fa[i] = 1, 0
            elif la == 0:
                self.s_fa[i] += 1
            elif la >= L:
                self.s_la[i], self.s_fa[i] = 0, 1
            else:
                self.s_la[i] = la + 1
            if self.s_won[i]:
                self.s_won[i] -= 1

    def _post(self):
        """Lock-flag convention: True if a flag at t-1 already shows a claim made at t-1."""
        if self.post_votes != self.pre_votes:
            return self.post_votes > self.pre_votes      # winners are near-certain evidence
        n = self.al[0]
        if n < 5:
            return True
        mu = self.rsum / self.rn if self.rn else 10.0
        return abs(self.al[1] / n - mu) >= abs(self.al[2] / n - mu)

    # ------------------------------------------------------------------ model
    def _p(self, c, b):
        n, kk = self.cl_n[c], self.cl_k[c]
        lo, hi = max(b - 1, 0), min(b + 2, self.NB)
        pn = (kk[lo:hi].sum() + self.PRIOR_N * self.PRIOR_P) / (n[lo:hi].sum() + self.PRIOR_N)
        return (kk[b] + self.NEIGH_K * pn) / (n[b] + self.NEIGH_K)

    def _free_now(self, t):
        """Expected free-age class counts of rivals free in the current round.

        Post-action flags give the free set exactly. With pre-action flags the latest snapshot is
        one round old: a rival free then is still free now unless it claimed last round's reward,
        which we weight by the model's own claim probability."""
        nc = len(self.FA_EDGES) + 1
        cnt = [0.0] * nc
        post = self._post()
        if not post:
            r1 = self.rw.get(t - 1)
            b1 = self._bin(r1) if r1 is not None else None
            stay = [1.0 if b1 is None else 1.0 - self._p(c, b1) for c in range(nc)]
        for i in range(len(self.s_sc)):
            la = self.s_la[i]
            if post:
                if la == 0:
                    cnt[self._cls(self.s_fa[i] - 1)] += 1.0
            else:
                if la == 0 and self.s_won[i] == 0:
                    cnt[self._cls(self.s_fa[i])] += stay[self._cls(self.s_fa[i] - 1)]
                elif la >= self.L:
                    cnt[0] += 1.0
        return cnt

    def _S(self, r, t):
        if not self.s_sc:
            return 1.0
        b = self._bin(r)
        s = self.f_reg[b // self.REGION] if self.CORR else 1.0
        for c, n in enumerate(self._free_now(t)):
            if n:
                s *= (1.0 - min(self._p(c, b), 0.999)) ** n
        return s

    def _refit(self, t):
        mu = self._mu()
        G = 200
        u = (np.arange(G) + 0.5) / G
        r = -mu * np.log1p(-u)
        b = np.minimum((r * 10.0 / (mu * self.BW)).astype(int), self.NB - 1)
        if self.s_sc:
            # realised no-claim rate per bin, shrunk towards neighbours and a 0.5 prior
            zk, zn = self.z_k, self.z_n
            pad_k = np.convolve(zk, [1, 1, 1], "same") + 1.0
            pad_n = np.convolve(zn, [1, 1, 1], "same") + 2.0
            zb = (zk + 3.0 * pad_k / pad_n) / (zn + 3.0)
            if self.CORR:
                # f = P(no rival claims | we would claim) / P(no rival claims), per region
                R = self.REGION
                f = (self.w_k + self.CORR_A) / (self.w_n + self.CORR_A)
                self.f_reg = np.clip(f, 0.05, 1.2)
                zb = zb * np.repeat(self.f_reg, R)[:self.NB]
            S = zb[b]
        else:
            S = np.ones(G)
        x = S * r
        L = self.L
        lo, hi = 0.0, float(x.max())
        for _ in range(40):
            c = 0.5 * (lo + hi)
            if L * np.maximum(x - c, 0.0).mean() > c:
                lo = c
            else:
                hi = c
        self.gap_c = 0.5 * (lo + hi)
        H = min(self.TAIL, self.T)
        V = np.zeros(H + 1)
        gap = np.zeros(H + 1)
        for n in range(1, H + 1):
            acc = V[n - 1 - L] if n - 1 - L > 0 else 0.0
            gap[n] = V[n - 1] - acc
            V[n] = np.maximum(x + acc, V[n - 1]).mean()
        self.gap_tail = gap
        self.fit_t = t

    # ------------------------------------------------------------------ play
    def play(self, reward, lockout, t, T, your_score, other_scores, is_locked):
        reward = float(reward)
        if t == 1 or t <= self.last_t or T != self.T or lockout != self.L:
            self._reset(T, lockout)
        self.last_t = t
        if self.my_claim is not None:
            u, s0, r0 = self.my_claim
            if t == u + 1 and r0 > 0:
                self.own_n += 1
                self.own_res[u] = your_score - s0 > 0.5 * r0
                self.own_res.pop(u - 4, None)
                if self.own_res[u]:
                    self.own_w += 1
                    self.streak = 0
                else:
                    # collision: random backoff so that colliding copies unlock out of step
                    self.streak += 1
                    hi = min(self.BACKOFF * 2 ** (self.streak - 1), self.BACKOFF_CAP)
                    self.backoff = self.rng.randint(0, hi) if self.BACKOFF > 0 else 0
            self.my_claim = None
        try:
            self._observe(t, other_scores)
        except Exception:
            self._init_slots(other_scores or [], fresh=False)
            self.valid_from = t + (self.L or 0)
        if self.prev_score is not None:
            self.earn = self.earn * self.EMP_DECAY + max(your_score - self.prev_score, 0.0)
            self.earn_n = self.earn_n * self.EMP_DECAY + 1.0
        self.prev_score = your_score
        self.rw[t] = reward
        self.rw.pop(t - 4, None)
        self.rsum += reward
        self.rn += 1
        if t > T or self.L <= 0:
            return (not is_locked) and reward > 0
        n = T - t + 1
        if self.gap_c is None or t - self.fit_t >= self.REFIT:
            self._refit(t)
        gap = self.gap_tail[n] if n < len(self.gap_tail) else self.gap_c
        if self.EMP_W > 0 and self.earn_n > 0 and self.gap_c > 0:
            # policy-iteration check: the lockout is worth L rounds at the rate we actually earn
            w = self.EMP_W * t / (t + self.EMP_RAMP)
            emp = self.L * self.earn / self.earn_n
            gap *= (1.0 - w) + w * emp / self.gap_c
        S = self._S(reward, t)
        # if we would claim this round (deterministic rule, also while locked), remember the
        # model's own S (without f) to calibrate it against what rivals actually do
        if S * reward >= gap and self.s_sc and not is_locked:
            f = self.f_reg[self._bin(reward) // self.REGION] if self.CORR else 1.0
            self.w_hist[t] = S / f
        self.w_hist.pop(t - 4, None)
        if is_locked:
            return False
        if t <= self.WARMUP and n > self.L + 1:
            return False
        if self.backoff > 0 and n > self.L + 1:
            self.backoff -= 1
            return False
        # randomise only under contention: alone (S ~ 1) the plain DP rule is optimal
        crowd = min(1.0, (1.0 - S) * self.CROWD)
        m = S * reward - gap * (1.0 + (self.jit - 1.0) * crowd)
        # correlation: how much less often our claims succeed than the uncorrected model says
        corr = 1.0 - (self.own_w + 2.0) / (self.own_raw + 2.0)
        corr = 0.0 if corr < 0.0 else (1.0 if corr > 1.0 else corr)
        tau = (self.TAU + self.TAU_CORR * corr) * crowd
        if tau > 0 and gap > 0:
            z = m / (tau * gap)
            p = 1.0 / (1.0 + math.exp(-max(min(z, 50.0), -50.0)))
        else:
            p = 1.0 if m >= 0 else 0.0
        if n > 1:
            p *= 1.0 - (1.0 - self.P_MAX) * crowd
        accept = self.rng.random() < p
        if accept:
            self.my_claim = (t, float(your_score), reward)
            self.own_pred += S
            self.own_raw += self.w_hist.get(t, S)
        return bool(accept)


play = Solution().play
