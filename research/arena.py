"""Multi-team arena mirroring the server loop. `python research/arena.py`

rules: "standard" (every claimant credited) or "exclusive" (one random claimant credited).
loser_locked: whether failed exclusive claimants are still locked out.
flag_shift: 1 -> other_scores lock flag is post-action (locked next round); 0 -> pre-action.
"""
import importlib.util, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_solution(path=ROOT / "solution.py", **overrides):
    spec = importlib.util.spec_from_file_location(f"sol{time.time_ns()}", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    for k, v in overrides.items():
        setattr(mod, k, v)
    return mod.Solution().play


def thresh(c):
    return lambda r, L, t, T, s, o, lk: r > c


def always(r, L, t, T, s, o, lk):
    return True


def run_game(players, T=2000, L=15, dist=None, rules="standard", loser_locked=True,
             flag_shift=1, seed=0):
    g = np.random.default_rng(seed)
    rewards = dist(g, T) if dist else g.exponential(10.0, T)
    n = len(players)
    score = np.zeros(n); lock_until = np.zeros(n, int)
    snap = None  # (scores, flags) for round t-1
    for t in range(1, T + 1):
        r = float(rewards[t - 1])
        locked = t <= lock_until
        claims = []
        for i, p in enumerate(players):
            others = [] if snap is None else [(float(snap[0][j]), bool(snap[1][j])) for j in range(n) if j != i]
            a = p(r, L, t, T, float(score[i]), others, bool(locked[i]))
            if a and not locked[i]:
                claims.append(i)
        pre_flags = locked.copy()
        if claims:
            if rules == "standard":
                winners = claims
            else:
                winners = [claims[g.integers(len(claims))]]
            for i in claims:
                if i in winners:
                    score[i] += r; lock_until[i] = t + L
                elif loser_locked:
                    lock_until[i] = t + L
        post_flags = (t + 1) <= lock_until
        snap = (score.copy(), post_flags if flag_shift == 1 else pre_flags)
    return score


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=6)
    args = ap.parse_args()
    field_desc = "8x plainDP, 4x thr20, 4x thr14, 3x thr25, 2x thr30, 2x always"
    for rules, ll, fs in [("standard", True, 1), ("exclusive", True, 1), ("exclusive", True, 0),
                          ("exclusive", False, 1)]:
        res = {"ours": [], "plainDP": [], "thr20": [], "thr14": [], "thr25": [], "thr30": [], "always": []}
        for s in range(args.games):
            ours = load_solution()
            plain = [load_solution(MODE="exp") for _ in range(8)]
            for p in plain:  # disable contention modelling for the "plain" rivals
                p.__self__._observe_rivals = lambda *a, **k: None
            players = [ours] + plain + [thresh(20)] * 4 + [thresh(14)] * 4 + [thresh(25)] * 3 + [thresh(30)] * 2 + [always] * 2
            labels = ["ours"] + ["plainDP"] * 8 + ["thr20"] * 4 + ["thr14"] * 4 + ["thr25"] * 3 + ["thr30"] * 2 + ["always"] * 2
            sc = run_game(players, rules=rules, loser_locked=ll, flag_shift=fs, seed=s)
            for lab, v in zip(labels, sc):
                res[lab].append(v)
        print(f"{rules:9s} loser_locked={ll!s:5s} flag_shift={fs}: " +
              "  ".join(f"{k}={np.mean(v):7.1f}" for k, v in res.items()))
