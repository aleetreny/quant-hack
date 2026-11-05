import bisect


class Solution:
    def __init__(self):
        self.history = []
        self.values = [0.0]
        self.thresholds = [0.0]

        self.cached_T = None
        self.cached_lockout = None
        self.last_update = 0

        self.update_interval = 20
        self.min_samples = 30

    def expected_excess(self, threshold, samples, prefix):
        """Estimate E[max(X - threshold, 0)]."""

        n = len(samples)

        if n == 0:
            return 0.0

        idx = bisect.bisect_right(samples, threshold)
        count = n - idx

        if count == 0:
            return 0.0

        total = prefix[n] - prefix[idx]

        return max(0.0, (total - count * threshold) / n)

    def compute_values(self, T, lockout):
        """Compute optimal thresholds from observed rewards."""

        samples = sorted(self.history)
        prefix = [0.0]

        for x in samples:
            prefix.append(prefix[-1] + x)

        self.values = [0.0]
        self.thresholds = [0.0]

        for n in range(1, T + 1):
            skip_value = self.values[n - 1]

            future_index = max(0, n - lockout - 1)
            accept_future = self.values[future_index]

            threshold = skip_value - accept_future

            expected_gain = self.expected_excess(
                threshold, samples, prefix
            )

            self.values.append(skip_value + expected_gain)
            self.thresholds.append(threshold)

        self.cached_T = T
        self.cached_lockout = lockout
        self.last_update = len(self.history)

    def play(
        self,
        reward: float,
        lockout: int,
        t: int,
        T: int,
        your_score: float,
        other_scores: list[tuple[float, bool]],
        is_locked: bool
    ) -> bool:

        # Observe every reward, including locked rounds
        self.history.append(reward)

        if is_locked:
            return False

        remaining = T - t + 1

        # Cold-start phase
        if len(self.history) < self.min_samples:

            if remaining <= lockout + 1:
                return reward > 0

            if len(self.history) < 5:
                return False

            mean_reward = sum(self.history) / len(self.history)

            return reward > 1.5 * mean_reward

        # Update empirical dynamic programming
        needs_update = (
            self.cached_T != T
            or self.cached_lockout != lockout
            or len(self.history) - self.last_update >= self.update_interval
        )

        if needs_update:
            self.compute_values(T, lockout)

        threshold = self.thresholds[remaining]

        return reward > threshold


play = Solution().play
