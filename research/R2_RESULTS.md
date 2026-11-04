# Round 2 results (hidden distribution, ranked on best score)

## 1. Live data: what the round-2 distribution is

Mined by `research/miner.py`. It polls the leaderboard every 25 s and saves one file per run in
`_local/runs/`; raw pages go to `_local/raw/`. The unbiased samples are the rounds that
bot-greedy and bot-random accepted. Both bots take negative rewards, so they ignore the reward
value. On round 1 their pooled means were 10.46 and 10.09 against a true mean of 10.
`research/r2_prior.py` writes the fit, and `_local/r2_latest_check.txt` is refreshed
automatically after every new round-2 run.

First round-2 run (2026-11-04 19:05:20, label just "Round 2", only the bots ran it):

| | mean | sd | q90 | q95 | q99 | KS D*sqrt(n) |
|---|---|---|---|---|---|---|
| data (n = 234) | -0.219 | 1.042 | 1.004 | 1.387 | 2.027 | |
| N(-0.2, 1), public engine 'b' | -0.200 | 1.000 | 1.082 | 1.445 | 2.126 | **0.65** |
| ETH fit: -0.19 + 0.97 * light-tail(Z) | -0.201 | 0.953 | 1.009 | 1.340 | 1.960 | 0.79 |
| N(0, 1) | 0 | 1 | 1.282 | 1.645 | 2.326 | 1.70 (rejected at 1%) |
| numpy pareto(3) = Lomax | 0.500 | 0.852 | 1.154 | 1.714 | 3.641 | 8.37 (rejected) |

Reading: the data fit N(-0.2, 1) (5% critical value 1.36), and pareto(3) and N(0, 1) are ruled
out. As at ETH, the upper tail sits below the normal fit (q90/q95/q99 are 0.08/0.06/0.10 lower),
but with 234 samples that is within noise. Every accept decision depends on that tail, so keep
mining: from about 5 runs (around 1200 samples) the per-run mean spread and the tail can be told
apart. Watch the "between-run" sd line in `_local/r2_latest_check.txt`; ETH saw about 0.1.

## 2. Target DP for an arbitrary distribution (`research/target_dp_general.py`)

It maximises P(score > B) with the state (rounds left, d = B - score), on a grid with step h.
Rewards are rounded to the same grid. Each step is a truncated convolution with a threshold
j*(d) found by binary search, computed as a reversed cumsum over a sliding window: O(D J) numpy
work per step. The table does not depend on B, so one table per distribution answers every target.

Two implementation details matter:

- **Grid step.** h is about sd/10, and coarser when D*J would exceed 1.5e5. With h = 0.165 on
  N(0,1) the policy loses most of its gain (MC 0.102 against 0.110 at h = 0.05).
- **Half-step offset.** Index 0 must mean "strictly above B". Counting ties as wins on a coarse
  grid inflates the predicted W.

Validation: predicted W against Monte Carlo (20,000 runs, T = 2000, L = 15), at B = the EV-DP
q90 and q99:

| distribution | table time | B | W predicted | MC target-DP | MC EV-DP | gain |
|---|---|---|---|---|---|---|
| eth (no run shift) | 0.40 s | 124.00 | 0.1154 | 0.1085 | 0.1000 | x1.08 |
| | | 129.86 | 0.0122 | 0.0117 | 0.0100 | x1.17 |
| N(0,1) | 0.51 s | 151.92 | 0.1140 | 0.1096 | 0.1000 | x1.10 |
| | | 158.39 | 0.0137 | 0.0126 | 0.0100 | x1.26 |
| lognormal exp(N(0,.75)) - 1.3 | 1.46 s | 237.66 | 0.1144 | 0.1109 | 0.1000 | x1.11 |
| | | 259.91 | 0.0114 | 0.0112 | 0.0100 | x1.12 |
| U(-2, 3) | 0.37 s | 259.30 | 0.1211 | 0.1086 | 0.1000 | x1.09 |
| | | 265.34 | 0.0140 | 0.0120 | 0.0100 | x1.21 |
| bimodal | 0.41 s | 248.80 | 0.1159 | 0.1085 | 0.1000 | x1.08 |
| | | 256.18 | 0.0140 | 0.0126 | 0.0100 | x1.26 |
| Exp(10) | 1.14 s | 2857.37 | 0.1120 | 0.1064 | 0.1000 | x1.06 |
| | | 2996.48 | 0.0134 | 0.0132 | 0.0100 | x1.32 |

The prediction is slightly optimistic, by about 3-10% relative, because of grid rounding; the
uniform's hard edge is the worst case. On rewards that lie exactly on the grid, W matches MC to
noise (h = 0.05: 0.1066 predicted, 0.1067 MC). Every table builds in under 1.5 s with numpy only.

Update after 4 runs (19:18, n = 935): mean -0.185, sd 1.012, q90/95/99 = 1.043 / 1.548 / 2.134.
KS against N(-0.2, 1) gives 0.82, so it fits; N(0, 1) gives 2.41 and is rejected. Skew is -0.07
and excess kurtosis 0.01. The between-run mean spread (0.040) is **below** sampling noise
(0.066), so there is no per-run shift. The upper tail now matches the normal. Conclusion: the
law is N(-0.2, 1), the public engine 'b', and there is nothing left to learn online.

The live field does not tie. In the 19:08:59 run the 11 serious teams scored 130.3-135.7 with
only one tie. `research/r2_field_kappa.py` recovers each team's threshold relative to the exact
N(-0.2, 1) EV-DP. In the 19:13:22 run several teams are pickier than EV:

- k ~ 1.11-1.13: umn, qwe, Mite
- k ~ 1.33-1.39: wrt, imp, neelutkarshmaxim
- k ~ 1.43-1.53: gregress, alpha, wer

The rest are learners whose threshold moves during the run. Because the field is not a clone
group, the R1 "shadow-lock" has nothing to lock onto in round 2.

## 3. Best-of-K contest

All contests: 20 shared runs, with the field having played 5 before we join. "Win" means our
best strictly beats the field's best. Numbers are P(win) ± one standard error.

### 3a. Target DP against EV-DP baselines (`research/r2_contest.py`, 320 contests)

The field is the oracle EV-DP plus online empirical EV-DP with thresholds x 0.95 / 1 / 1.05 / 1.1.
Target = field best so far + delta, with delta in within-run sd units (5.9 on N(-0.2, 1), 5.5 on
eth).

| contender | N(-0.2,1): P(beat field) | P(beat oracle-EV clones) | mean score | eth (run shift sd 0.1): P(beat field) |
|---|---|---|---|---|
| O: target DP, oracle law, delta 0.25 / 0.5 / 1 sd | 0.169 / 0.178 / 0.172 | 0.33 / 0.29 / 0.32 | 124.9 | 0.138 / 0.181 / 0.178 |
| P: target DP, prior + online shift | 0.194 / 0.175 / 0.184 | 0.34 / 0.32 / 0.33 | 124.9 | 0.131 / 0.138 / 0.147 |
| E: target DP, no prior (learned online) | 0.081 / 0.084 / 0.081 | 0.20 / 0.21 / 0.20 | 123.4 | 0.053 / 0.044 / 0.072 |
| exact EV-DP x 1.00 | 0.000 (ties the oracle) | 0.000 | 126.2 | |
| exact EV-DP x 1.03 | **0.234** | **0.447** | 126.2 | |
| EV-DP, prior + online shift (candidate, TARGET=None) | 0.172 | 0.350 | 126.2 | |

The learning cost is large: the target DP without a prior wins less than half as often as with one.

### 3b. Which rule against which field (`research/r2_kappa_contest.py`, N(-0.2,1), 320 contests)

- **"sim" field:** the same field as in 3a.
- **"live" field:** mimics the 19:13 run, with exact EV-DP x {1.0, 1.12, 1.35, 1.45, 1.52},
  empirical EV x {1, 1.05, 1.1} and Lea's rule.

| contender | sim field | live field | mean score |
|---|---|---|---|
| exact EV-DP x 0.97 | 0.175 | 0.141 | 126.05 |
| x 1.00 | 0.000 | 0.000 | 126.10 |
| x 1.01 | 0.163 | 0.109 | 126.11 |
| x 1.02 | 0.206 | 0.134 | 126.11 |
| x 1.03 | 0.203 | 0.125 | 126.08 |
| x 1.05 | 0.234 | 0.147 | 125.99 |
| **x 1.08** | **0.256** | **0.153** | 125.71 |
| x 1.10 | 0.231 | 0.084 | 125.48 |
| x 1.15 | 0.166 | 0.097 | 124.64 |
| x 1.20 | 0.166 | 0.091 | 123.44 |
| x 1.30 | 0.053 | 0.031 | 120.01 |
| Lea (`solution_round2.py`: EV-DP, empirical + N(-0.2,1) prior fading at 300) | 0.188 | (in the field) | 125.97 |
| target DP P, delta 0.5 sd | 0.166 | 0.122 | 124.78 |

Standard errors are 0.016-0.024.

- **Lea is a sound mean maximiser.** Its 125.97 is within 0.1% of the exact DP. But for a
  best-of-runs ranking it is one more near-clone (0.188), below a plain exact DP x 1.05-1.08.
- **The target DP does not pay in round 2** (0.12-0.17). Its per-run gain is only x1.1, and it
  gives up about 1.3 points of mean. That suits a clone group, not a field of varied learners.
- **The best k depends on the field.** k = 1.00 ties anyone running the exact rule, and k ≥ 1.1
  collides with the teams already there. The sweet spot is k = 1.05-1.08.

## 4. Recommendation and candidate

Round 2: **exact EV-DP on N(-0.2, 1) with KAPPA = 1.06-1.08**, with an empirical fallback in case
the law changes; Ale3 already has that structure. Re-run `research/r2_field_kappa.py` after each
run and keep KAPPA in a gap that no other team occupies. Drop TARGET in round 2.

`research/r2_solution_candidate.py` (target DP with optional prior) passes the client tests
(`uv run python research/r2_test.py`, four modes) and takes about 1.5 s per game. Keep it as a
tool in case the field turns into a clone group, where a target pays off (0.29-0.34 against
oracle-EV clones).
