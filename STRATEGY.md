# Strategy notes

## The game

| Round | Rewards | Rules | Opponents | Ranked on |
|---|---|---|---|---|
| 1 | Known exponential | Standard | Reference bots | Best score |
| 2 | Hidden distribution | Standard | Reference bots | Best score |
| 3 | Known exponential | Exclusive claim | Other teams only | Hidden (latest) score |

Accepting at round `t` locks you for rounds `t+1 .. t+L` (L = 15), so each accept costs 15 rounds of
opportunities. The right threshold is the value of those 15 rounds, which a finite-horizon DP gives exactly.

## Policy (`solution.py`)

With `n` rounds left: accept iff `q(r) * r >= [q + (1 - q) * lf] * (V(n-1) - V(n-1-L))`.

- Standard rules: `q` is constant and drops out, so the rule is `r >= V(n-1) - V(n-1-L)`.
  For an exponential with mean mu that is about `2.01 * mu` (top ~13% of rewards) for most of the game,
  falling to 0 over the last ~15 rounds. With mean 10: accept rewards of about 20.1 or more.
- Exponential mode: closed-form DP, mean estimated online (or fixed via `EXP_MEAN`).
- Empirical mode: DP on the empirical distribution of every reward seen (rewards are visible while locked).
- Exclusive claims: the policy reads rivals' score and lock changes to learn how often free rivals claim
  rewards of each size, then estimates our chance of being credited `q(r)` given how many rivals are free
  right now. It also learns whether failed claims still lock (`lf`). This switches on automatically only
  when rounds never credit more than one team.

## Settings per round

Edit the constants at the top of `solution.py` before submitting:

| Round | `MODE` | `EXP_MEAN` |
|---|---|---|
| 1 | `"exp"` | announced mean if known, else `None` |
| 2 | `"empirical"` | `None` |
| 3 | `"exp"` | announced mean if known, else `None` |

`"auto"` (default) also works: it starts exponential and switches to empirical if a KS test rejects the fit.

## Research tools (`research/`)

- `solo.py`: standard rules, many distributions, compared with hindsight-optimal fixed thresholds.
- `arena.py`: multi-team server-like loop (standard / exclusive, losers locked or not, both lock-flag conventions).
- `smart_field.py`: exclusive games against fields containing several copies of our policy.
