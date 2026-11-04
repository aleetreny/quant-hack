import math

class Solution:
    def __init__(self):
        self.history = []
        self.V = [0.0]

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

        # Precompute optimal expected rewards
        while len(self.V) <= T:
            n = len(self.V)

            skip_value = self.V[n - 1]
            future_value = self.V[max(0, n - lockout - 1)]

            threshold = skip_value - future_value

            expected_gain = 10 * math.exp(-threshold / 10)

            self.V.append(skip_value + expected_gain)

        # t starts at 1, not 0
        remaining = T - t + 1

        skip_value = self.V[max(0, remaining - 1)]
        future_value = self.V[max(0, remaining - lockout - 1)]

        threshold = skip_value - future_value

        return reward > threshold


play = Solution().play
