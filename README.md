# Trading Game Client

## Setup

Use Python 3.11 or newer and `uv`.

```bash
git clone <this-repo>
cd gr_tradinggame_client
uv sync --python 3.11
```

## Usage

### 1. Edit `solution.py`

Write your strategy in the `play` function or class method:

```python
class Solution:
    def __init__(self):
        self.history = []

    def play(self, reward: float, lockout: int, t: int, T: int, your_score: float, other_scores: list[tuple[float, bool]], is_locked: bool) -> bool:
        self.history.append(reward)
        if is_locked:
            return False
        return reward > 14.0

play = Solution().play
```

Your function is called every round (including while locked out). Returning `True` while locked out has no effect but you still observe the reward and the state of the game.

### 2. Test locally

```bash
uv run python test.py
```

Checks your function handles edge cases and runs fast enough, including a game-like check (2000 rounds, 23 other teams) that must average under 15 ms per round. On the server, a team that uses more than 60 s in one game is benched for the rest of that game.

The server has numpy, pandas and the Python standard library. Other packages are not available there, even if you install them locally.

### 3. Submit

Edit `config.py` with your team name, password, and server address, then:

```bash
uv run python submit.py
```

You can resubmit as many times as you want. The server uses your latest submission.

### Password

Your password is set on your first submission. All future submissions for that team name must use the same password. Choose anything you like, just don't forget it.

## Parameters

| Parameter | Type | Description |
|---|---|---|
| `reward` | `float` | The reward offered this round |
| `lockout` | `int` | Rounds locked out if you accept |
| `t` | `int` | Current round number, 1 to T |
| `T` | `int` | Total rounds |
| `your_score` | `float` | Your cumulative score so far at round $t-1$ |
| `other_scores` | `list[tuple[float, bool]]` | List of `(score, is_locked)` tuples for other teams at round $t-1$ (empty `[]` on round 1) |
| `is_locked` | `bool` | Whether you are currently locked out this round |
