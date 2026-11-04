"""Single-player (standard rules) benchmark across reward distributions. `python research/solo.py`"""
import sys, time
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
from arena import load_solution, thresh

T, L = 2000, 15
DISTS = {
    "exp(10)": lambda g: g.exponential(10, T),
    "uniform(0,40)": lambda g: g.uniform(0, 40, T),
    "lognormal": lambda g: g.lognormal(2, 1, T),
    "normal(10,5)": lambda g: g.normal(10, 5, T),
    "pareto": lambda g: 10 * g.pareto(2.5, T),
    "gamma(3,3)": lambda g: g.gamma(3, 3, T),
    "bimodal": lambda g: np.where(g.random(T) < 0.9, g.normal(5, 1, T), g.normal(40, 5, T)),
    "shifted_exp": lambda g: 20 + g.exponential(5, T),
}

def play_solo(fn, rewards):
    score, lu = 0.0, 0
    for t in range(1, T + 1):
        r = float(rewards[t - 1]); locked = t <= lu
        if fn(r, L, t, T, score, [], locked) and not locked:
            score += r; lu = t + L
    return score

if __name__ == "__main__":
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    for name, d in DISTS.items():
        res = {}
        t0 = time.time(); calls = 0
        for s in range(games):
            rw = d(np.random.default_rng(s))
            q = np.quantile(rw, [0.8, 0.87, 0.9])
            cands = {"ours": load_solution(), "ours_emp": load_solution(MODE="empirical"),
                     "oracle_q87": thresh(q[1]), "oracle_q80": thresh(q[0]), "oracle_q90": thresh(q[2])}
            for k, f in cands.items():
                res.setdefault(k, []).append(play_solo(f, rw))
        print(f"{name:14s} " + "  ".join(f"{k}={np.mean(v):7.1f}±{np.std(v)/np.sqrt(games):4.1f}" for k, v in res.items()))
