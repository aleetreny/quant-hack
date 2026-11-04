# You can create any variables here to store state within a run (but not between runs)
history = []

def play(reward: float, lockout: int, t: int, T: int, your_score: float, other_scores: list[tuple[float, bool]], is_locked: bool) -> bool:
	"""
	Return True to accept the reward, False to skip.

	Called every round (including while locked out, so you can track history).
	If you are locked out (is_locked == True), returning True has no effect.

	Parameters:
	    reward       - the reward offered this round (float)
	    lockout      - rounds you will be locked out if you accept (int)
	    t            - current round number, starting at 1 (int)
	    T            - total number of rounds (int)
	    your_score   - your cumulative score so far at round t-1 (float)
	    other_scores - list of (score, is_locked) for other teams at round t-1 (list of tuples: [(float, bool), ...]). Empty [] on round 1.
	    is_locked    - bool, whether you are currently locked out this round

	Returns:
	    bool - True to accept, False to skip
	"""
	history.append(reward)
	return True
