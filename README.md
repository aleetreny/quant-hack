# G-Research Trading Game: our strategies

Every round the server offers a reward. Accepting it adds the reward to your score and locks you out
for the next 15 rounds; the game lasts 2000 rounds. The whole problem is deciding when a reward is
worth 15 rounds of waiting.

| Round | Rewards | Rules | Ranked on | Our script | Result |
|---|---|---|---|---|---|
| 1 | Exponential, mean 10 (announced) | Every accepter is paid | Best score | [`strategies/round1_exponential.py`](strategies/round1_exponential.py) | Best score 2862.17 (2nd) |
| 2 | Hidden distribution | Every accepter is paid | Best score | [`strategies/round2_hidden_distribution.py`](strategies/round2_hidden_distribution.py) | Best score 134.69 (6th) |
| 3 | Exponential, mean 10 | Exclusive claim: two or more accepters are paid nothing | Latest score | [`strategies/round3_exclusive_claim.py`](strategies/round3_exclusive_claim.py) | 298.93, **1st** |

## Round 1: exact optimal stopping

With `n` rounds left, the value of the game satisfies

```
V(n) = E[ max( r + V(n-1-L), V(n-1) ) ]        accept  iff  r >= V(n-1) - V(n-1-L)
```

For an exponential reward the expectation has a closed form, so the full 2000-round threshold
table is exact and takes milliseconds to build. Mid-game the threshold is about 20.1 (the top 13%
of rewards). It rises in the last few dozen rounds, when only one accept is left, and drops to 0
on the final round.

Because ranking used the *best* score over many runs, and every team that solved the DP exactly
tied to the cent on every run, we accepted slightly above the exact threshold (`KAPPA = 1.03`).
This de-correlates our score from that group, so on the luckiest run we can finish ahead of it
instead of tying.

## Round 2: hidden distribution

Same DP, but the distribution is unknown. The strategy:

- starts from a prior fitted to rewards mined from the public leaderboard in earlier runs (a normal
  distribution, mean about -0.2, sd 1);
- checks that prior against every reward it sees and falls back to the empirical distribution if
  the data disagree;
- never accepts a non-positive reward;
- accepts slightly above the risk-neutral threshold (`KAPPA = 1.07`), again because the best score
  counted.

The decision rule works on the exact DP of the fitted distribution, and only the distribution
itself is learned online.

## Round 3: exclusive claim

If exactly one team accepts, it is paid; if two or more accept, nobody is paid and all of them are
locked out. Thresholds from round 1 are useless here, because every team that uses them asks for
the same large rewards and collides.

The submitted script is a coordinated system rather than a single-team policy. Several of our
teams run the same file with a different role, and they never collide with each other:

- Every decision depends only on the round number and the reward, which all teams see. Each copy
  can therefore replay the whole group's schedule, including who is locked and when.
- One team in the group claims mid-size rewards (3 to 15).
- Every other reward is claimed by the free team of the group with the lowest id.
- In practice one of our teams claims nearly every round, so a rival claim usually collides and
  pays nobody, while rewards that rivals leave alone are paid to us.

Before settling on this, we built an order-invariant adaptive learner, kept in `research/` (see
[Research](#research) below). It tracks rivals by their score values, so it does not depend on the
order of the list. From that it learns the chance that nobody else claims each reward size, and it
randomises to avoid lockstep collisions.

## Repository layout

```
strategies/      the three scripts we used, one per round (each passes the client test)
solution.py      general DP strategy with an online contention model
research/        simulators, tuning experiments and result notes (R1/R2/R3_RESULTS.md)
src/             the organisers' client library
test.py          client test harness; submit.py submits solution.py
```

## Research

- `research/r1_*.py`, `research/target_dp*.py`: optimal stopping, target-score DP, best-of-runs
  contests.
- `research/r2_*.py`: hidden-distribution learners, priors, threshold multipliers.
- `research/r3_arena.py`: round-3 simulator where a collision pays nobody. It covers many fields
  (round-1 cutters, greedy teams, bands, adaptive bots, copies of a strategy), shuffled or stable
  `other_scores` order, and both lock-flag conventions.
- `research/r3_bench.py`, `research/r3_replica.py`: benchmarks, plus a replica of a real run
  fitted from the public leaderboard.

## Running

```bash
uv sync --python 3.11
uv run python test.py                     # client test on solution.py
uv run python research/r3_test.py         # client test and timing for the round-3 learner
uv run python research/r3_bench.py base 16 # round-3 benchmark
```

To submit, copy a script from `strategies/` to `solution.py`, fill in `config.py` and run
`uv run python submit.py`.
