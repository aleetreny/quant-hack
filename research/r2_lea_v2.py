import math
from bisect import bisect_right, insort
from collections import deque
from statistics import NormalDist

PRIOR_MU, PRIOR_SD = -0.2, 1.0
PRIOR_N = 200
FADE = 300
CHECK_Z = 4.0
SD_RANGE = (0.7, 1.4)
MIN_SAMPLES = 3
H_MAX = 600
DRIFT_W = 200
DRIFT_KS = 2.4

_ND = NormalDist()


def _pseudo(mu, sd, m):
	return [mu + sd * _ND.inv_cdf((i + 0.5) / m) for i in range(m)]


class Solution:
	def __init__(self):
		self.reset()

	def reset(self):
		self.xs = []
		self.sum = 0.0
		self.sum2 = 0.0
		self.prior_ok = True
		self.recent = deque(maxlen=DRIFT_W)
		self.drifts = 0
		self.cut = None
		self.fit_n = -10**9
		self.G = None
		self.last_t = 0

	def check_prior(self):
		n = len(self.xs)
		if n < 5:
			return
		mean = self.sum / n
		var = max(self.sum2 / n - mean * mean, 0.0)
		if abs(mean - PRIOR_MU) > CHECK_Z * PRIOR_SD / math.sqrt(n):
			self.prior_ok = False
		elif n >= 20 and not (SD_RANGE[0] <= math.sqrt(var) / PRIOR_SD <= SD_RANGE[1]):
			self.prior_ok = False

	def check_drift(self):
		n = len(self.xs)
		if n < 2 * DRIFT_W:
			return False
		rec = sorted(self.recent)
		k = n - DRIFT_W
		j = 0
		dist = 0.0
		for i, v in enumerate(self.xs):
			while j < DRIFT_W and rec[j] <= v:
				j += 1
			d = abs(j / DRIFT_W - (i + 1 - j) / k)
			if d > dist:
				dist = d
		if dist <= DRIFT_KS * math.sqrt(1.0 / DRIFT_W + 1.0 / k):
			return False
		self.xs = rec
		self.sum = sum(rec)
		self.sum2 = sum(x * x for x in rec)
		self.prior_ok = False
		self.drifts += 1
		return True

	def samples(self):
		n = len(self.xs)
		m = int(PRIOR_N * max(0.0, 1.0 - n / FADE))
		if m <= 0:
			return self.xs
		if self.prior_ok:
			pad = _pseudo(PRIOR_MU, PRIOR_SD, m)
		else:
			mean = self.sum / n
			sd = math.sqrt(max(self.sum2 / n - mean * mean, 0.0))
			pad = _pseudo(mean, sd, m) if sd > 0 else [mean] * m
		return sorted(self.xs + pad)

	def refit(self):
		xs = self.samples()
		n = len(xs)
		suf = [0.0] * (n + 1)
		for i in range(n - 1, -1, -1):
			suf[i] = suf[i + 1] + xs[i]
		G = self.G
		V = [0.0] * (H_MAX + 1)
		cut = [0.0] * (H_MAX + 1)
		for s in range(1, H_MAX + 1):
			cs = V[s - 1] - (V[s - G] if s >= G else 0.0)
			i = bisect_right(xs, cs)
			V[s] = V[s - 1] + (suf[i] - cs * (n - i)) / n
			cut[s] = cs
		self.cut = cut
		self.fit_n = len(self.xs)

	def play(self, reward, lockout, t, T, your_score, other_scores, is_locked):
		try:
			if t == 1 or t < self.last_t:
				self.reset()
			self.last_t = t
			G = max(int(lockout), 0) + 1
			if G != self.G:
				self.G = G
				self.cut = None
			r = float(reward)
			if r == r:
				insort(self.xs, r)
				self.sum += r
				self.sum2 += r * r
				self.recent.append(r)
			n = len(self.xs)
			was_ok = self.prior_ok
			if self.prior_ok:
				self.check_prior()
			if is_locked or not reward > 0 or (n < MIN_SAMPLES and not self.prior_ok):
				return False
			every = 1 if n < 50 else 5 if n < FADE else 10
			if self.cut is None or n - self.fit_n >= every or was_ok != self.prior_ok:
				self.check_drift()
				self.refit()
			s = min(max(T - t + 1, 1), H_MAX)
			return bool(reward >= self.cut[s])
		except Exception:
			return False


play = Solution().play
