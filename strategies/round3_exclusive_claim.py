# Round 3 team system: one MAIN team plus up to N_HELPERS helper teams, never colliding with each other.
#
# Every team runs this same file with a different ROLE / HELPER_ID. All teams see the same reward every
# round and every rule below depends only on (t, reward) and on simulated lock states, so each copy can
# replay the whole system exactly:
#   MAIN claims iff it is free and reward >= C_SCALE * (round-1 optimal cutoff with n rounds left).
#   On every other round with reward >= S_MIN, the free helper with the lowest id claims it. A helper
#   claim collides with any rival claim of that round (collision pays nobody and locks every claimant),
#   so rivals lose those rewards and spend 15 rounds locked; our main team is never touched.
# Lock rule (server): a claim at round t blocks rounds t+1 .. t+L, free again at t+L+1.
# A missing helper only leaves its rounds unblocked; it never causes a collision between our teams.

import math

ROLE = "helper"  # "main" or "helper"
HELPER_ID = 3        # 0 .. N_HELPERS-1 for helpers
N_HELPERS = 12
C_SCALE = 0.0        # >0: main claims reward >= C_SCALE * round-1 optimal cutoff; 0: use the band below
MAIN_LO = 3.0        # band mode: main claims MAIN_LO <= reward <= MAIN_HI when free
MAIN_HI = 15.0
S_MIN = 0.0          # helpers ignore rewards below this
S_MAX = 1e9          # ... and above this (keeps helpers' own scores below the main team's)
MEAN = 10.0


def _gaps(T, L):
    W = [0.0] * (T + 1)
    for n in range(1, T + 1):
        acc = W[n - 1 - L] if n - 1 - L > 0 else 0.0
        W[n] = W[n - 1] + math.exp(-(W[n - 1] - acc))
    g = [0.0] * (T + 1)
    for n in range(1, T + 1):
        acc = W[n - 1 - L] if n - 1 - L > 0 else 0.0
        g[n] = MEAN * (W[n - 1] - acc)
    return g


class Solution:
    ROLE = ROLE
    HELPER_ID = HELPER_ID
    N_HELPERS = N_HELPERS
    C_SCALE = C_SCALE
    MAIN_LO = MAIN_LO
    MAIN_HI = MAIN_HI
    S_MIN = S_MIN
    S_MAX = S_MAX

    def __init__(self):
        self.T = self.L = None
        self.last_t = 0

    def _reset(self, T, L):
        self.T, self.L = T, L
        self.gap = _gaps(T, L)
        self.main_lock = 0
        self.h_lock = [0] * self.N_HELPERS
        self.last_t = 0

    def _step(self, t, r):
        """Advance the shared simulation by round t; returns (main claims, helper id claiming or -1)."""
        L = self.L
        if self.C_SCALE > 0:
            want = r > 0 and r >= self.C_SCALE * self.gap[self.T - t + 1]
        else:
            want = self.MAIN_LO <= r <= self.MAIN_HI or (self.T - t < self.L and r > 0)
        mc = t > self.main_lock and want
        if mc:
            self.main_lock = t + L
            return True, -1
        if self.S_MIN <= r <= self.S_MAX:
            for i in range(self.N_HELPERS):
                if t > self.h_lock[i]:
                    self.h_lock[i] = t + L
                    return False, i
        return False, -1

    def play(self, reward, lockout, t, T, your_score, other_scores, is_locked):
        try:
            L = int(lockout)
            if t == 1 or T != self.T or L != self.L or t != self.last_t + 1:
                self._reset(T, L)
                # rebuild from round 1 is impossible without past rewards; start fresh at this t
            self.last_t = t
            mc, h = self._step(t, float(reward))
            if is_locked:
                return False
            if self.ROLE == "main":
                return bool(mc)
            return bool(h == self.HELPER_ID)
        except Exception:
            return False


play = Solution().play
