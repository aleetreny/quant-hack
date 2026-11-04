import math
import random
from collections import deque

MEAN = 10.0
DEFAULT_LOCK = 15
DELAY = 2
WINDOW = 800
REPLAY_EVERY = 20
MIN_HIST = 120
SWITCH = 1.03
NEAR_BEST = 0.95
JITTER = 0.08
OPP_MIN_OBS = 15
OPP_WINDOW = 400
PRIOR_FIT = (20.0, 0.8, 0.1)

FLATS = (6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 27.0, 30.0, 35.0)
BANDS = tuple((lo, hi) for lo in (6.0, 8.0, 10.0, 12.0, 14.0, 16.0)
              for hi in (14.0, 16.0, 18.0, 20.0, 22.0, 25.0) if hi >= lo + 4.0)
KAPPAS = (4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 24.0)
RULES = ([("flat", c, 0.0) for c in FLATS] + [("band", lo, hi) for lo, hi in BANDS]
         + [("model", k, 0.0) for k in KAPPAS])
DEFAULT_RULE = ("model", 12.0, 0.0)


def _fit_threshold(obs):
	pts = sorted(obs)
	n = len(pts)
	total = sum(1 for _, a in pts if a)
	if total == 0:
		return float("inf"), 0.0, 0.5 / (n + 2)
	best_err, best_i, below = None, 0, 0
	for i in range(n + 1):
		err = below + (n - i) - (total - below)
		if best_err is None or err < best_err:
			best_err, best_i = err, i
		if i < n and pts[i][1]:
			below += 1
	if best_i == 0:
		theta = pts[0][0] - 1e-9
	elif best_i == n:
		theta = pts[-1][0] + 1e-9
	else:
		theta = 0.5 * (pts[best_i - 1][0] + pts[best_i][0])
	hi, lo = pts[best_i:], pts[:best_i]
	p_hi = (sum(1 for _, a in hi if a) + 0.5) / (len(hi) + 1)
	p_lo = (sum(1 for _, a in lo if a) + 0.1) / (len(lo) + 2)
	return theta, p_hi, p_lo


def _end_scale(G, H=400):
	V = [0.0] * (H + 1)
	c = [0.0] * (H + 1)
	for s in range(1, H + 1):
		cs = V[s - 1] - (V[s - G] if s >= G else 0.0)
		c[s] = cs
		V[s] = V[s - 1] + math.exp(-cs)
	flat = c[H] if c[H] > 0 else 1.0
	return [x / flat for x in c]


class Team:
	__slots__ = ("prev_score", "prev_flag", "clean", "flips", "pending", "last_final",
	             "obs", "n_obs", "fit", "n_fit", "rose_last")

	def __init__(self):
		self.prev_score = None
		self.prev_flag = False
		self.clean = set()
		self.flips = set()
		self.pending = None
		self.last_final = -10**9
		self.obs = deque(maxlen=OPP_WINDOW)
		self.n_obs = 0
		self.fit = None
		self.n_fit = 0
		self.rose_last = False


class Solution:
	def __init__(self, seed=None):
		self.rng = random.Random(seed)
		self.reset(None)

	def reset(self, T):
		self.T = T
		self.jit = 1.0 + JITTER * (2.0 * self.rng.random() - 1.0)
		self.last_t = 0
		self.G = DEFAULT_LOCK + 1
		self.scale = _end_scale(self.G)
		self.teams = []
		self.rewards = {}
		self.rall = [0.0]
		self.palone = {}
		self.lag_votes = [0, 0]
		self.flip_log = []
		self.hist = deque(maxlen=WINDOW)
		self.n_final = 0
		self.pool = None
		self.pool_at = -10**9
		self.rule = DEFAULT_RULE
		self.replay_at = -10**9
		self.replay_every = self.rng.randint(REPLAY_EVERY - 5, REPLAY_EVERY + 5)
		self.min_hist = MIN_HIST + self.rng.randint(0, 40)
		self.my_pending = None
		self.my_result = {}

	def lag(self):
		a, b = self.lag_votes
		if a + b >= 3:
			return 1 if b > a else 0
		if len(self.flip_log) >= 8:
			ks = self.flip_log[-200:]
			r0 = sum(self.rall[k] for k in ks if 0 < k < len(self.rall))
			r1 = sum(self.rall[k - 1] for k in ks if 1 < k <= len(self.rall))
			return 1 if r1 > r0 else 0
		return 0

	def observe(self, t, others):
		if len(others) != len(self.teams):
			self.teams = [Team() for _ in others]
		k = t - 1
		for tm, rep in zip(self.teams, others):
			try:
				score, flag = rep
			except Exception:
				score, flag = None, False
			flag = bool(flag)
			rose = False
			if isinstance(score, (int, float)) and tm.prev_score is not None:
				rose = abs(score - tm.prev_score) > 1e-9
			flip = flag and not tm.prev_flag
			if rose:
				tm.clean.add(k)
				if flip:
					self.lag_votes[0] += 1
					tm.pending = None
				else:
					tm.pending = k
			elif flip:
				if tm.pending == k - 1:
					self.lag_votes[1] += 1
				else:
					tm.flips.add(k)
					self.flip_log.append(k)
				tm.pending = None
			elif tm.pending is not None and tm.pending < k - 1:
				tm.pending = None
			tm.rose_last = rose
			tm.prev_score = score if isinstance(score, (int, float)) else None
			tm.prev_flag = flag

	def took(self, tm, r_turn, lag):
		return r_turn in tm.clean or (r_turn + lag) in tm.flips

	def finalise(self, turn):
		r = self.rewards.pop(turn, None)
		pa = self.palone.pop(turn, 1.0)
		if r is None:
			return
		lag = self.lag()
		taken = False
		for tm in self.teams:
			acc = self.took(tm, turn, lag)
			if turn - tm.last_final >= self.G:
				tm.obs.append((r, acc))
				tm.n_obs += 1
			if acc:
				tm.last_final = turn
				taken = True
		mine = self.my_result.pop(turn, None)
		if mine is not None:
			taken = not mine
		self.hist.append((r, taken, pa))
		self.n_final += 1

	def p_take(self, tm, r):
		if len(tm.obs) >= OPP_MIN_OBS:
			if tm.fit is None or tm.n_obs - tm.n_fit >= 10:
				tm.fit = _fit_threshold(list(tm.obs))
				tm.n_fit = tm.n_obs
			theta, hi, lo = tm.fit
		else:
			theta, hi, lo = self.pool or PRIOR_FIT
		return hi if r >= theta else lo

	def p_alone(self, t, r):
		p = 1.0
		for tm in self.teams:
			free = not tm.prev_flag and not tm.rose_last and t - tm.last_final >= self.G
			if free:
				p *= 1.0 - self.p_take(tm, r)
		return p

	@staticmethod
	def accepts(rule, r, pa, sc):
		kind, a, b = rule
		if kind == "flat":
			return r >= a * sc
		if kind == "band":
			return a * sc <= r < b * sc
		return r * pa >= a * sc

	def replay(self):
		G = self.G
		gains = {}
		for rule in RULES:
			gain, free_at = 0.0, 0
			for i, (r, taken, pa) in enumerate(self.hist):
				if i >= free_at and self.accepts(rule, r, pa, 1.0):
					if not taken:
						gain += r
					free_at = i + G
			gains[rule] = gain
		best_gain = max(gains.values())
		cur_gain = gains.get(self.rule)
		if cur_gain is not None and cur_gain >= best_gain / SWITCH:
			return
		near = [rule for rule, g in gains.items() if g >= NEAR_BEST * best_gain]
		self.rule = self.rng.choice(near)

	def play(self, reward, lockout, t, T, your_score, other_scores, is_locked):
		try:
			if t != self.last_t + 1 or T != self.T:
				self.reset(T)
			self.last_t = t
			G = (int(lockout) if lockout and lockout > 0 else DEFAULT_LOCK) + 1
			if G != self.G:
				self.G = G
				self.scale = _end_scale(G)
			others = list(other_scores or [])
			if t > 1:
				self.observe(t, others)
			if self.my_pending is not None:
				turn, before = self.my_pending
				self.my_result[turn] = your_score - before > 1e-9
				self.my_pending = None
			r = float(reward)
			self.rewards[t] = r
			self.rall.append(r)
			pa = self.p_alone(t, r) if self.teams else 1.0
			self.palone[t] = pa
			self.finalise(t - DELAY)
			if t - self.pool_at >= 25:
				pool = [o for tm in self.teams for o in list(tm.obs)[-100:]]
				if len(pool) >= 40:
					self.pool = _fit_threshold(pool)
				self.pool_at = t
			if self.n_final >= self.min_hist and t - self.replay_at >= self.replay_every:
				self.replay()
				self.replay_at = t
			if is_locked or not r > 0:
				return False
			s = max(T - t + 1, 1)
			sc = self.jit * self.scale[min(s, len(self.scale) - 1)]
			if self.accepts(self.rule, r, pa, sc):
				self.my_pending = (t, your_score)
				return True
			return False
		except Exception:
			return bool(not is_locked and reward > 20.0)


play = Solution().play
