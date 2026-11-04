"""Exclusive-claim games against fields containing many copies of our strategy."""
import sys, time
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
import os
SOL = os.environ.get("SOL", None) or __import__("arena").ROOT / "solution.py"
from arena import load_solution, thresh, always, run_game

def plain():
    p = load_solution(SOL, MODE="exp"); p.__self__._observe_rivals = lambda *a, **k: None; return p

for n_smart in [int(x) for x in os.environ.get("NS", "1,6,12,24").split(",")]:
    for ll in [True, False]:
        res = {}
        for s in range(3):
            labels, players = [], []
            for _ in range(n_smart): labels.append("ours"); players.append(load_solution(SOL, MODE="exp"))
            rest = 24 - n_smart
            mix = [("plainDP", plain), ("thr14", lambda: thresh(14)), ("thr20", lambda: thresh(20)), ("always", lambda: always)]
            for i in range(rest):
                lab, mk = mix[i % len(mix)]; labels.append(lab); players.append(mk())
            t0 = time.time()
            sc = run_game(players, rules="exclusive", loser_locked=ll, seed=100 + s)
            dt = time.time() - t0
            for lab, v in zip(labels, sc): res.setdefault(lab, []).append(v)
        print(f"smart={n_smart:2d} loser_locked={ll!s:5s} ({dt:4.1f}s/game) " + "  ".join(f"{k}={np.mean(v):6.1f}" for k, v in res.items()))
