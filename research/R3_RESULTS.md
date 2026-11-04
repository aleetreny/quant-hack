# Round 3 results (exclusive claims, collision pays nobody)

Candidate: `research/r3_solution_candidate.py` (no edits to `solution.py`). Not submitted.

## Rules simulated (`research/r3_arena.py`)

Exp(10) rewards shared by all teams, T=2000, L=15. Exactly one accepter is credited; two or more
accepters get nothing and are all locked 15 rounds. Only teams, 24 per game (us + 23).
`other_scores` order: `shuffle` (new random order every turn, default), `fixed`, `stable`.
Lock flag at t-1: post-action (`fs=1`), pre-action (`fs=0`) or never set (`fs=-1`).

Fields: `solo`; `eth_early` (12 round-1 DP cutters, greedy, bands, followers); `bot_flood`
(15 greedy); `band_crowd`; `dp1_wall` (23 DP cutters); `adaptive` (bandits, winner-followers,
undercutters); `mid_switch` (teams change strategy at t=500-1200); `late_flood` (10 greedy from
t=1000); `public_band` (10 x band 9-16, the public ETH recommendation); `sol_py` (8 x current
`solution.py`); `smart_mix` (variants of this candidate, `solution.py`, bandits, followers);
`selfK` (K copies of the strategy under test); `self12_bandit`; `sparse` (5 teams).

## Strategy (summary)

- Order-invariant tracking: a team's score only changes when it wins, so the score value is its
  identity. Matching the multiset of (score, locked) between turns gives the winner and, per score
  group, the number of new locks (correcting expiries after L flags) = number of claimants. No list
  positions are used anywhere.
- Claim model per free rival by free-age class (0 / 1-3 / 4+ rounds free; greedy teams claim the
  moment they unlock), per reward bin of width 1, with forgetting (0.997/round):
  S(r) = P(no rival claims) for the current free set.
- Threshold: value of 15 lost rounds from a DP on the realised no-claim rate (stationary
  mid-game, exact over the last 160 rounds), blended with 15 x our own recent earnings per round.
  Accept iff S(r) r >= gap.
- Anti-collision: S calibrated on the rounds where we wanted to claim (own outcome when we
  claimed); smooth accept probability whose softness grows with measured correlation; random
  backoff after a collision; per-game jitter. All of it switches off when S ~ 1, so alone it is the
  exact round-1 DP (2676 = optimum).
- Flag convention (pre/post) detected online from winners and from which alignment makes new locks
  depend on the reward.

## Results: 16 games per field, seat 0, shuffled order, post flags

`mean (p10, mean rank among 24)`. Baselines in seat 0 against the same rivals and rewards; in
`selfK` the copies are copies of the baseline.

| field | cand | dp1 | band9_16 | solution.py |
|---|---|---|---|---|
| solo | 2676 (p10 2532, r1.0) | 2685 (p10 2512, r1.0) | 1221 (p10 1193, r1.0) | 2676 (p10 2535, r1.0) |
| eth_early | 963 (p10 794, r1.0) | 1 (p10 0, r8.4) | 132 (p10 77, r6.4) | 6 (p10 0, r8.4) |
| bot_flood | 2011 (p10 1936, r1.0) | 1 (p10 0, r3.1) | 1148 (p10 1112, r1.0) | 483 (p10 14, r1.4) |
| band_crowd | 1305 (p10 1080, r1.0) | 21 (p10 0, r2.6) | 0 (p10 0, r3.0) | 408 (p10 95, r1.2) |
| dp1_wall | 2063 (p10 1923, r1.0) | 0 (p10 0, r2.8) | 1221 (p10 1193, r1.0) | 571 (p10 59, r1.1) |
| adaptive | 933 (p10 728, r1.0) | 0 (p10 0, r13.8) | 273 (p10 120, r5.8) | 18 (p10 0, r13.0) |
| mid_switch | 1767 (p10 1629, r1.0) | 443 (p10 350, r1.0) | 228 (p10 152, r1.0) | 718 (p10 462, r1.0) |
| sol_py | 1297 (p10 1103, r1.1) | 15 (p10 0, r3.1) | 0 (p10 0, r3.2) | 179 (p10 0, r2.9) |
| sparse | 1417 (p10 1254, r2.0) | 189 (p10 0, r4.3) | 0 (p10 0, r4.0) | 182 (p10 0, r4.2) |
| self2 | 909 (p10 760, r1.4) | 0 (p10 0, r3.2) | 0 (p10 0, r3.2) | 34 (p10 0, r2.9) |
| self4 | 611 (p10 485, r2.6) | 0 (p10 0, r3.4) | 0 (p10 0, r3.4) | 5 (p10 0, r4.8) |
| self8 | 434 (p10 298, r5.0) | 0 (p10 0, r3.4) | 0 (p10 0, r3.4) | 18 (p10 0, r4.6) |
| self12 | 329 (p10 279, r9.0) | 0 (p10 0, r7.6) | 0 (p10 0, r6.6) | 0 (p10 0, r7.2) |
| public_band | 1300 (p10 1056, r1.0) | 1 (p10 0, r4.2) | 0 (p10 0, r4.4) | 117 (p10 46, r3.2) |
| late_flood | 1592 (p10 1445, r1.0) | 1 (p10 0, r3.2) | 0 (p10 0, r3.3) | 456 (p10 18, r1.4) |
| smart_mix | 374 (p10 272, r7.4) | 12 (p10 0, r15.7) | 288 (p10 164, r9.9) | 0 (p10 0, r16.5) |
| self24 | 258 (p10 171, r12.8) | 0 (p10 0, r1.0) | 0 (p10 0, r1.0) | 0 (p10 0, r1.0) |
| self12_bandit | 289 (p10 226, r9.5) | 0 (p10 0, r11.2) | 0 (p10 0, r11.1) | 0 (p10 0, r11.7) |

(The candidate column above is the version before the last fix of flag-convention detection; the
final candidate, same settings: avg 1149, worst field 281, avg p10 1019.)

| Strategy | avg over fields | worst field | avg p10 | avg rank |
|---|---|---|---|---|
| candidate (final) | 1149 | 281 | 1019 | 3.1 |
| round-1 DP (cut ~20) | 187 | 0 | 159 | 5.2 |
| band 9-16 | 251 | 0 | 223 | 4.0 |
| `solution.py` (index-based rival model) | 326 | 0 | 179 | 4.9 |

## Robustness of the candidate (final version, 16 games per field)

| Setting | avg | worst field | avg p10 | avg rank |
|---|---|---|---|---|
| shuffled order, post flags | 1149 | 281 | 1019 | 3.1 |
| shuffled order, pre flags | 1113 | 264 | 981 | 3.2 |
| shuffled order, no flags at all | 906 | 165 | 781 | 3.1 |
| stable order, post flags (pre-fix version) | 1145 | 253 | 1009 | 3.1 |
| fixed order, post flags (pre-fix version) | 1132 | 232 | 986 | 3.4 |

`solution.py` with a stable order (its best case) averages 291, worst field 0: its index logic is
not the only problem; with no randomisation its copies and the DP crowd wipe it out.

Per field, final candidate (post / pre / no flags):

|  field | cand | cand | cand  |
| ---|---|---|--- |
|  solo | 2676 (p10 2532, r1.0) | 2676 (p10 2532, r1.0) | 2676 (p10 2532, r1.0)  |
|  eth_early | 920 (p10 756, r1.0) | 901 (p10 741, r1.0) | 604 (p10 472, r2.2)  |
|  bot_flood | 1989 (p10 1890, r1.0) | 1980 (p10 1869, r1.0) | 1372 (p10 1287, r1.0)  |
|  band_crowd | 1256 (p10 1087, r1.0) | 1277 (p10 1110, r1.0) | 1013 (p10 843, r1.0)  |
|  dp1_wall | 2069 (p10 1920, r1.0) | 2023 (p10 1914, r1.0) | 1390 (p10 1291, r1.0)  |
|  adaptive | 1007 (p10 895, r1.0) | 864 (p10 699, r1.1) | 718 (p10 580, r1.1)  |
|  mid_switch | 1758 (p10 1633, r1.0) | 1717 (p10 1559, r1.0) | 1636 (p10 1438, r1.0)  |
|  sol_py | 1426 (p10 1129, r1.1) | 1296 (p10 1057, r1.0) | 931 (p10 815, r1.0)  |
|  sparse | 1394 (p10 1244, r2.1) | 1412 (p10 1189, r1.9) | 1303 (p10 1141, r2.0)  |
|  self2 | 895 (p10 763, r1.5) | 826 (p10 666, r1.7) | 641 (p10 482, r1.4)  |
|  self4 | 673 (p10 573, r2.4) | 597 (p10 498, r2.6) | 477 (p10 379, r2.1)  |
|  self8 | 432 (p10 344, r4.1) | 400 (p10 351, r4.9) | 289 (p10 206, r4.5)  |
|  self12 | 316 (p10 237, r9.8) | 316 (p10 253, r9.6) | 216 (p10 117, r9.1)  |
|  public_band | 1341 (p10 1197, r1.0) | 1277 (p10 1154, r1.0) | 991 (p10 840, r1.0)  |
|  late_flood | 1571 (p10 1388, r1.0) | 1572 (p10 1385, r1.0) | 1282 (p10 1109, r1.0)  |
|  smart_mix | 368 (p10 292, r7.8) | 358 (p10 281, r8.1) | 397 (p10 302, r4.2)  |
|  self24 | 281 (p10 205, r9.5) | 264 (p10 195, r10.0) | 165 (p10 88, r11.3)  |
|  self12_bandit | 307 (p10 251, r8.6) | 285 (p10 210, r8.8) | 212 (p10 138, r10.1)  |

## What mattered (ablations, 8-16 games per field)

- Determinism is fatal among copies: no randomisation scored 0 in every `selfK` field.
- Softer acceptance (TAU) is a trade-off: it doubles self-play / learner-field scores but costs
  10-20% against bots. Making softness proportional to the measured correlation keeps both
  (worst field 109 -> 250).
- Correlation calibration + collision backoff: avg p10 942 -> 977.
- Model gap vs realised: in learner-heavy fields the model gap (5-7) was ~3x the realised
  15 x earnings rate; blending them helps self-play (self12_bandit 250 -> 305).
- Flag-convention detection by winners alone failed in fields where rivals rarely win (dp1_wall pre
  flags 1765); adding the reward-alignment test fixed it (2023).
- DECAY 0.995-0.998, P_MAX 0.9-1.0, BACKOFF 0-8, prior 0.05-0.3, free-age classes: all within
  noise (avg 1122-1165). Defaults kept; `FA_EDGES=(1,3,8)` was +26 on one run, not verified.

## Weak spots / caveats

- Many copies of a best-responding learner end in an inefficient mixed equilibrium: 8-24 copies
  get 250-430 each and rank mid-pack. Committed strategies (bandits that settle on a band) beat
  adaptive learners in learner-heavy fields (`self12_bandit`, `smart_mix`: rank 7-9).
- If the server never sets the lock flags, it falls back on its own claim outcomes (-20%).
- The arena field is still a model: ETH showed simulators overstated live scores by ~2x.
  Check `other_scores` semantics on the server with an early submission if allowed.

## Reproduce

```bash
uv run python research/r3_test.py                       # client test + 23-rival timing
uv run python research/r3_bench.py base 16 shuffle all 1 # candidate vs baselines
uv run python research/r3_bench.py base 16 stable all 1  # stable other_scores order
uv run python research/r3_bench.py '{"cand":{}}' 16 shuffle all 0   # pre-action flags
```

Timing: 0.05-0.13 ms per call on average with 23 rivals (limit 15 ms), worst call < 1 ms after
warm-up; full game < 0.3 s.

## Claude/ETH-style fields (final candidate, 16 games per field, shuffled order, post flags)

Rivals: band 9-16 / 8-16 claiming with p=0.75, `rps` = contrarian `r * P(solo) >= 8` with p=0.75 and
random backoff (order-free lock counts), copies of the candidate, round-1 DP cutters (~20), greedy.
`claude_band` = 8 band9-16 p.75 + 4 band8-16 p.75 + 4 rps + 4 DP + 3 greedy; `band_sat` = 15 band
9-16 + 8 DP; `claude_rps` = 12 rps + 5 band + 3 candidate copies + 3 DP; `claude_mix` = 6 candidate
copies + 6 band + 6 rps + 5 DP.

| field | candidate | band 9-16 | band 9-16 p=0.75 | round-1 DP |
|---|---|---|---|---|
| claude_band | 763 (p10 622, rank 1.1) | 1 (0) | 65 (24) | 2 (0) |
| band_sat | 1644 (p10 1508, rank 1.0) | 0 (0) | 142 (88) | 1 (0) |
| claude_rps | 420 (p10 349, rank 2.4) | 0 (0) | 104 (56) | 0 (0) |
| claude_mix | 362 (p10 287, rank 4.7) | 0 (0) | 54 (23) | 0 (0) |
| public_band | 1260 (p10 1129, rank 1.0) | 0 (0) | 294 (231) | 1 (0) |
| self8 | 414 (p10 339, rank 4.6) | 0 (0) | 81 (54) | 0 (0) |
| dp1_wall | 2061 (p10 1915, rank 1.0) | 1221 (1193) | 1134 (1099) | 0 (0) |
| bot_flood | 2013 (p10 1922, rank 1.0) | 1148 (1112) | 1073 (1019) | 1 (0) |
| eth_early | 920 (p10 809, rank 1.0) | 132 (77) | 259 (190) | 1 (0) |

Summary: candidate avg 1095, worst field 362, avg rank 2.0; band 9-16 avg 278, worst 0; band p=0.75
avg 356, worst 54; round-1 DP avg 1, worst 0. Frozen copy: `_local/submitted/Ale5.py` (sha256 af49a799...).

## Live tuning (edit the class attributes at the top of `Solution`)

- Many collisions (our score barely moves while we keep getting locked): raise `TAU_CORR` (1.5 -> 2.5)
  and `BACKOFF` (4 -> 8, `BACKOFF_CAP` 32); lowering `P_MAX` (0.95 -> 0.8) also spreads claims.
- Almost no accepts: lower `PRIOR_P` (0.15 -> 0.05) and `EMP_W` stays; or reduce `TAU_CORR` (-> 0.5)
  if the field looks bot-like. `WARMUP` is only 5 rounds, so it is not the cause.
