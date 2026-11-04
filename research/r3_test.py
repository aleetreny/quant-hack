"""Client test (`test.py`) run against the round-3 candidate, plus a 23-rival timing check and an
exec-from-source check (submit.py sends the file as source). `uv run python research/r3_test.py`"""
import random, sys, timeit
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gr_tradinggame.coding.client import Client

SRC = (ROOT / "research" / "r3_solution_candidate.py").read_text()

# 1. exactly what the server does with a source submission: exec the file, call `play`
ns = {}
exec(SRC, ns)
play = ns["play"]
client = Client("YourTeamName", password="YourPassword")
client.test(play)
client.test(SRC)

# 2. full 2000-round game with 23 rivals, worst-case timing per call
ns = {}
exec(SRC, ns)
play = ns["play"]
rng = random.Random(1)
scores = [0.0] * 23
locks = [0] * 23
lock_until = 0
worst = tot = 0.0
my = 0.0
for t in range(1, 2001):
    r = rng.expovariate(0.1)
    others = [] if t == 1 else [(scores[j], locks[j] >= t) for j in range(23)]
    rng.shuffle(others)
    tic = timeit.default_timer()
    a = play(r, 15, t, 2000, my, others, t <= lock_until)
    dt = timeit.default_timer() - tic
    worst, tot = max(worst, dt), tot + dt
    assert isinstance(a, bool)
    claim = [j for j in range(23) if locks[j] < t and rng.random() < (0.5 if r > 12 else 0.05)]
    if a and t > lock_until:
        claim.append(-1)
        lock_until = t + 15
    for j in claim:
        if j >= 0:
            locks[j] = t + 15
    if len(claim) == 1:
        if claim[0] == -1:
            my += r
        else:
            scores[claim[0]] += r
print(f"2000-round game, 23 rivals: mean {tot / 2000 * 1000:.3f} ms/call, worst {worst * 1000:.2f} ms, "
      f"total {tot:.2f} s, our score {my:.0f}")
