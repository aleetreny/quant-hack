"""Local simulator. `python sim.py`"""
import numpy as np
from policy import LockoutPolicy, dp_exponential

N, LOCK = 2000, 15

def play(rewards, fn, q=1.0, g=None):
    score, lu = 0.0, -1
    for t, r in enumerate(rewards):
        free = t > lu
        if fn(r, score, not free) and free:
            if g is None or g.random() < q: score += r
            lu = t + LOCK
    return score

def fixed_q(qt):
    seen = []
    def f(r, s, l):
        seen.append(r); return (not l) and len(seen) > 20 and r >= np.quantile(seen, qt)
    return f

GENS = {"exp(1)": lambda g: g.exponential(1, N), "uniform": lambda g: g.uniform(0, 1, N),
        "lognormal": lambda g: g.lognormal(0, 1, N), "normal": lambda g: g.normal(0, 1, N),
        "pareto": lambda g: g.pareto(2.5, N)}

if __name__ == "__main__":
    print("theoretical V(2000) exp(1), q=1:", round(dp_exponential(N, LOCK, 1.0)[-1], 2))
    for name, gen in GENS.items():
        res = {}
        for s in range(30):
            rw = gen(np.random.default_rng(s))
            c = {"always": lambda *a: True, "q90": fixed_q(.9), "q95": fixed_q(.95),
                 "dp_empirical": LockoutPolicy(N, LOCK).decide}
            if name == "exp(1)": c["dp_known_exp"] = LockoutPolicy(N, LOCK, exp_mean=1.0).decide
            for k, f in c.items(): res.setdefault(k, []).append(play(rw, f))
        print(f"{name:10s}", "  ".join(f"{k}={np.mean(v):7.2f}±{np.std(v)/np.sqrt(len(v)):.1f}" for k, v in res.items()))
