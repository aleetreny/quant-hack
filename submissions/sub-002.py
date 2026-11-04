import math


class Solution:
    def __init__(self):
        self.history = []
        self.values = [0.0]
        self.cached_lockout = None

    def compute_values(self, T, lockout):
        """Calculate optimal expected future rewards."""

        self.values = [0.0]

        for n in range(1, T + 1):
            skip = self.values[n - 1]
            accept_future = self.values[max(0, n - lockout - 1)]

            threshold = skip - accept_future

            expected_gain = 10.0 * math.exp(-threshold / 10.0)

            self.values.append(skip + expected_gain)

        self.cached_lockout = lockout

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

        self.history.append(reward)

        if is_locked:
            return False

        if self.cached_lockout != lockout or len(self.values) <= T:
            self.compute_values(T, lockout)

        remaining = T - t + 1

        skip_value = self.values[max(0, remaining - 1)]
        accept_future = self.values[max(0, remaining - lockout - 1)]

        threshold = skip_value - accept_future

        # Competitive adjustment
        if other_scores and t > 200:

            leader_score = max(score for score, _ in other_scores)
            score_gap = leader_score - your_score

            # Estimate the expected remaining score
            expected_remaining = self.values[remaining]

            # Adjust aggressiveness based on relative position
            relative_gap = score_gap / max(expected_remaining, 1.0)

            if relative_gap > 0.15:
                # Significantly behind: increase selectivity
                threshold *= 1.12

            elif relative_gap > 0.05:
                # Slightly behind
                threshold *= 1.06

            elif relative_gap < -0.10:
                # Comfortably ahead: stay closer to optimal
                threshold *= 1.00

        return reward > threshold


play = Solution().play
