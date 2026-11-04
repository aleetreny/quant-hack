"""Round 2: turn mined runs (_local/runs, from research/miner.py) into PRIOR_Q and a TARGET.

Unbiased samples are the rounds bot-random and bot-greedy accepted (their accept decision does not
depend on the reward, checked on round 1: pooled means 10.09 and 10.46 for Exp(10)). If bot-greedy
never takes a negative reward while bot-random does, greedy is dropped (it would be truncated at 0).
Each run is centred on its own mean before pooling, so PRIOR_Q is the within-run shape; the location
is re-learned online in every run (r2_solution_candidate.py, SHIFT_N0).

`uv run python research/r2_prior.py [label substring, default "Round 2"] [delta in sd, default 0.5]`
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from target_dp_general import DISTS, T, ev_gap_sorted, simulate  # noqa: E402

REFS = ("normal02", "eth", "normal", "pareto3", "exp10")


def ks(x, dist):
    """KS distance of sample x to a reference law (via its quantile table) and D * sqrt(n)."""
    q = DISTS[dist].quantiles(200000)
    xs = np.sort(x)
    F = np.searchsorted(q, xs, side="right") / len(q)
    n = len(xs)
    i = np.arange(1, n + 1)
    d = max(np.max(i / n - F), np.max(F - (i - 1) / n))
    return d, d * np.sqrt(n)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_local", "runs")
BOTS = ("bot-random", "bot-greedy")


def load(label):
    runs = []
    for f in sorted(glob.glob(os.path.join(ROOT, "*.json"))):
        with open(f) as fh:
            r = json.load(fh)
        if label.lower() in r["label"].lower():
            runs.append(r)
    return runs


def field_best(run):
    out = []
    for row in run["table"]:
        try:
            out.append(float(row[2]))
        except ValueError:
            pass
    return max(out) if out else float("nan")


def bot_samples(run, bots):
    rw = run["rewards"]
    rounds = sorted({k for b in bots for k in run["accepts"].get(b, [])})
    return np.array([rw[str(k)] for k in rounds])


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "Round 2"
    delta = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    runs = load(label)
    if not runs:
        sys.exit(f"no mined runs matching {label!r} in {ROOT}")
    neg = {b: sum(sum(1 for k in r["accepts"].get(b, []) if r["rewards"][str(k)] < 0) for r in runs)
           for b in BOTS}
    bots = BOTS if neg["bot-greedy"] > 0 or neg["bot-random"] == 0 else ("bot-random",)
    print(f"{len(runs)} runs '{runs[0]['label']}'; negative rewards taken: {neg}; using {bots}")
    pooled, means = [], []
    for r in runs:
        x = bot_samples(r, bots)
        if len(x) < 30:
            continue
        means.append(x.mean())
        pooled.append(x - x.mean())
        q = np.quantile(x, [0.5, 0.9, 0.95, 0.99])
        print(f"  {r['last_run']}  n={len(x):4d} mean {x.mean():7.3f} sd {x.std():6.3f}  "
              f"q50/90/95/99 {np.round(q, 3)}  field best {field_best(r):.2f}")
    x = np.concatenate(pooled) + np.mean(means)
    m, s = x.mean(), x.std()
    se_mean = s / np.sqrt(np.mean([len(p) for p in pooled]))
    run_sd = np.sqrt(max(np.var(means) - se_mean ** 2, 0.0))
    z = (x - m) / s
    print(f"pooled n={len(x)} mean {m:.3f} sd {s:.3f} skew {np.mean(z ** 3):.3f} "
          f"exkurt {np.mean(z ** 4) - 3:.3f}  run-mean sd {np.std(means):.3f} "
          f"(sampling {se_mean:.3f} -> between-run {run_sd:.3f})")
    for p, zq in ((0.9, 1.2816), (0.95, 1.6449), (0.99, 2.3263)):
        print(f"  q{p}: data {np.quantile(x, p):.3f}  normal fit {m + zq * s:.3f}")
    raw = np.concatenate([bot_samples(r, bots) for r in runs])
    print(f"reference laws on the raw pooled samples (n={len(raw)}; KS D*sqrt(n) > 1.36 rejects at 5%, "
          f"> 1.63 at 1%):")
    for ref in REFS:
        rq = DISTS[ref].quantiles(200000)
        d, z = ks(raw, ref)
        print(f"  {DISTS[ref].name:55s} mean {rq.mean():7.3f} sd {rq.std():6.3f} q90/95/99 "
              f"{np.round(np.quantile(rq, [0.9, 0.95, 0.99]), 3)}  KS D={d:.4f} D*sqrt(n)={z:.2f}")
    print(f"  {'data':55s} mean {raw.mean():7.3f} sd {raw.std():6.3f} q90/95/99 "
          f"{np.round(np.quantile(raw, [0.9, 0.95, 0.99]), 3)}")
    PRIOR_Q = np.quantile(x, (np.arange(400) + 0.5) / 400)
    # EV-DP score spread on this shape (within a run) to size the target margin
    g = np.random.default_rng(0)
    gap = ev_gap_sorted(PRIOR_Q)
    sc = simulate(g.choice(x, (4000, T)), lambda n, s_, r: r >= gap[n])
    best = max(field_best(r) for r in runs)
    print(f"EV-DP on the pooled shape: mean {sc.mean():.2f} sd {sc.std():.2f}; field best so far {best:.2f}")
    print(f"suggested TARGET = field best + {delta} sd = {best + delta * sc.std():.2f}")
    print("PRIOR_Q = [" + ", ".join(f"{v:.4f}" for v in PRIOR_Q) + "]")
