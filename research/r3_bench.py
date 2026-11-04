"""Benchmark round-3 strategies across fields. `uv run python research/r3_bench.py [set] [seeds]`

Every strategy sits in seat 0 against the same 23 rivals and reward sequences; in the selfK fields
K-1 rivals are copies of the strategy under test. Prints mean (p10, mean rank) per field and a
robustness summary (worst-field mean, mean of per-field p10).
"""
import sys, time
sys.path.insert(0, __import__("os").path.dirname(__file__))
import numpy as np
from r3_arena import ROOT, evaluate, summarize, fields

CAND = str(ROOT / "research" / "r3_solution_candidate.py")


def cand(**kw):
    return ("file", CAND, kw)


BASE = {
    "dp1": ("bot", "dp1", {}),
    "band9_16": ("bot", "band", {"lo": 9, "hi": 16}),
    "solution.py": ("file", str(ROOT / "solution.py"), {}),
}

SETS = {
    "claude": {"cand": cand(), "band9_16": BASE["band9_16"], "band9_16p75": ("bot", "band", {"lo": 9, "hi": 16, "p": 0.75}),
               "dp1": BASE["dp1"]},
    "base": dict(cand=cand(), **BASE),
}


def robust(out):
    rows = []
    for cn, per in out.items():
        means = [np.mean([r["score"] for r in v]) for v in per.values()]
        p10s = [np.percentile([r["score"] for r in v], 10) for v in per.values()]
        ranks = [np.mean([r["rank"] for r in v]) for v in per.values()]
        rows.append(f"{cn:28s} avg {np.mean(means):6.0f}  worst-field {np.min(means):6.0f}  "
                    f"avg-p10 {np.mean(p10s):6.0f}  avg-rank {np.mean(ranks):4.1f}")
    return "\n".join(rows)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "base"
    if which.startswith("{"):
        # ad-hoc sweep: {"name": {param: value}, ...} of candidate overrides
        SETS[which] = {k: cand(**v) for k, v in eval(which).items()}
    nseed = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    order = sys.argv[3] if len(sys.argv) > 3 else "shuffle"
    fns = sys.argv[4].split(",") if len(sys.argv) > 4 and sys.argv[4] != "all" else None
    fs = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    t0 = time.time()
    out = evaluate(SETS[which], fns, seeds=range(nseed), order=order, flag_shift=fs)
    print(summarize(out, fns))
    print()
    print(robust(out))
    print(f"\n{time.time() - t0:.0f}s")
