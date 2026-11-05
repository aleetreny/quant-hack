"""Estimate each team's acceptance rule in mined round-2 runs, relative to the exact N(-0.2, 1) EV-DP.

For every team we rebuild its lock state from its own accepts. Rewards are known on every round some
team accepted. kappa_hi = min r / gap(n) over its accepts (it took these); kappa_lo = max r / gap(n)
over known rounds where it was free and declined. A team on the exact rule x k has
kappa_lo < k <= kappa_hi.
`uv run python research/r2_field_kappa.py`
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from target_dp_general import DISTS, L, T, ev_gap_sorted  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_local", "runs")
gap = ev_gap_sorted(DISTS["normal02"].quantiles(40000))

for f in sorted(glob.glob(os.path.join(ROOT, "*Round_2*.json"))):
    run = json.load(open(f))
    rw = {int(k): v for k, v in run["rewards"].items()}
    print(f"== {run['last_run']}  known rewards {len(rw)}")
    rows = []
    for team, acc in run["accepts"].items():
        if not acc or team.startswith("bot-greedy") or team == "bot-random":
            continue
        acc = sorted(acc)
        accs = set(acc)
        lock, lo, hi, nlo = 0, 0.0, np.inf, 0
        for t in range(1, T + 1):
            n = T - t + 1
            if t in accs:
                if gap[n] > 0:
                    hi = min(hi, rw[t] / gap[n])
                lock = t + L
            elif t > lock and t in rw and gap[n] > 0 and n > 2 * L:
                lo = max(lo, rw[t] / gap[n])
                nlo += 1
        rows.append((run["finals"][team], team, len(acc), lo, hi, nlo))
    for sc, team, na, lo, hi, nlo in sorted(rows, reverse=True):
        print(f"  {team:24s} {sc:9.3f} accepts {na:3d}  kappa in ({lo:.3f}, {hi:.3f}]  "
              f"({nlo} declined known rounds)")
