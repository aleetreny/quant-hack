"""Replica of a real round-3 run, fitted from the leaderboard's won-turn traces, and a search for
the variant of our team (plus optional blocker teams) that maximises P(our team is 1st).

`uv run python research/r3_replica.py _local/runs/<run>.json [games]`

Fit per rival (only wins are visible):
  * >= 80 wins and median gap 16 -> phase greedy (claims the moment it unlocks, random backoff after
    a collision, which is how three greedy teams end up on distinct phases);
  * other teams with wins -> band [q05, q95] of the rewards they won (cut at q05 if q95 is high);
  * teams with 0 wins in the run -> round-1 DP cutters (~20), which collide with each other.
Our own team (OURS) is removed and replaced by the variant under test.
"""
import json, math, sys
sys.path.insert(0, __import__("os").path.dirname(__file__))
import numpy as np
from r3_arena import ROOT, make_player, run_game, BOTS, _DP1

OURS = __import__("os").environ.get("R3_OURS", "")   # our team name in the leaderboard data
CANDF = str(ROOT / "research" / "r3_solution_candidate.py")


def bot_phase_greedy(rng=None, back=6):
    st = {}

    def play(r, L, t, T, s, o, lk):
        if t == 1:
            st.update(mine=None, wait=0)
        if st["mine"] is not None:
            if s - st["mine"] < 1e-9:
                st["wait"] = int(rng.integers(0, back + 1))
            st["mine"] = None
        if lk:
            return False
        if st["wait"] > 0:
            st["wait"] -= 1
            return False
        st["mine"] = s
        return True
    return play


def bot_set(lo=0.0, hi=1e9, p=1.0, back=0, rng=None):
    """Claims rewards in [lo, hi) with prob p; optional random backoff after a collision."""
    st = {}

    def play(r, L, t, T, s, o, lk):
        if t == 1:
            st.update(mine=None, wait=0)
        if st["mine"] is not None:
            if s - st["mine"] < 1e-9 and back:
                st["wait"] = int(rng.integers(0, back + 1))
            st["mine"] = None
        if lk or not (lo <= r < hi):
            return False
        if st["wait"] > 0:
            st["wait"] -= 1
            return False
        if p < 1.0 and rng.random() >= p:
            return False
        st["mine"] = s
        return True
    return play


def bot_blocker(lo=8.0, p=0.85, rng=None):
    """Blocker: claims every reward BELOW lo (the complement of the main team's set A=[lo, inf))."""
    return bot_set(lo=0.0, hi=lo, p=p, back=0, rng=rng)


BOTS.update(phase_greedy=bot_phase_greedy, set=bot_set, blocker=bot_blocker)


def fit_field(path):
    d = json.load(open(path))
    R = {int(k): v for k, v in d["rewards"].items()}
    field = []
    for row in d["table"]:
        name = row[1]
        if name == OURS or row[3] in ("&mdash;", "—"):
            continue
        acc = sorted(d["accepts"].get(name, []))
        if not acc:
            zero = __import__("os").environ.get("ZERO", "dp1")
            if zero == "blockmid":   # alternative fit: zero-score teams block mid/high rewards
                field.append((name, ("bot", "set", {"lo": 7.0, "p": 0.7})))
            else:
                field.append((name, ("bot", "dp1", {"jitter": 0.03})))
            continue
        r = np.array([R[x] for x in acc])
        gaps = np.diff(acc)
        if len(acc) >= 80 and len(gaps) and np.median(gaps) <= 16:
            field.append((name, ("bot", "phase_greedy", {})))
        else:
            lo, hi = float(np.quantile(r, 0.05)), float(np.quantile(r, 0.95))
            if hi > 25 or len(acc) < 8:
                hi = 1e9
            field.append((name, ("bot", "set", {"lo": lo * 0.95, "hi": hi * 1.05})))
    return field


def run(field, ours, blockers, seed):
    rng = np.random.default_rng(seed * 7 + 3)
    players = [make_player(ours, rng)] + [make_player(sp, rng) for _, sp in field] + \
              [make_player(b, rng) for b in blockers]
    sc = run_game(players, seed=seed)
    rivals = sc[1:1 + len(field)]
    return sc[0], 1 + sum(1 for x in rivals if x > sc[0]), max(rivals)


def job(a):
    field, name, ours, blockers, seed = a
    return name, run(field, ours, blockers, seed)


VARIANTS = {
    "Ale5": ("file", CANDF, {}),
    "Ale5_bo0_pm1": ("file", CANDF, {"BACKOFF": 0, "P_MAX": 1.0}),
    "Ale6": ("file", CANDF, {"TAU_CORR": 0.0, "BACKOFF": 0, "P_MAX": 1.0}),
    "greedy_bo6": ("bot", "phase_greedy", {}),
    "greedy_bo3": ("bot", "phase_greedy", {"back": 3}),
    "lt8_bo4": ("bot", "set", {"hi": 8.0, "back": 4}),
    "lt12_bo4": ("bot", "set", {"hi": 12.0, "back": 4}),
    "lt16_bo4": ("bot", "set", {"hi": 16.0, "back": 4}),
}

if __name__ == "__main__":
    from multiprocessing import Pool
    path = sys.argv[1]
    G = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    mode = sys.argv[3] if len(sys.argv) > 3 else "variants"
    field = fit_field(path)
    print("rivals:", len(field), {k: sum(1 for _, s in field if s[1] == k) for k in ("phase_greedy", "set", "dp1")})
    jobs = []
    if mode == "variants":
        for name, v in VARIANTS.items():
            jobs += [(field, name, v, [], s) for s in range(G)]
    else:
        # main team claims A = [lo, inf) (Ale5 restricted, or plain set); k blockers claim r < lo
        for lo in (6.0, 8.0, 10.0):
            for k in (0, 2, 4, 6, 10):
                for mn, main in (("set", ("bot", "set", {"lo": lo, "back": 4})),
                                 ("greedy", ("bot", "phase_greedy", {}))):
                    bl = [("bot", "blocker", {"lo": lo, "p": 0.85})] * k
                    jobs += [(field, f"{mn} A>={lo:g} k={k}", main, bl, s) for s in range(G)]
    with Pool(16) as pool:
        res = pool.map(job, jobs, chunksize=2)
    agg = {}
    for name, (sc, rk, best) in res:
        agg.setdefault(name, []).append((sc, rk, best))
    for name, v in agg.items():
        a = np.array(v)
        print(f"{name:24s} mean {a[:,0].mean():6.0f} p10 {np.percentile(a[:,0],10):6.0f} "
              f"P(1st) {np.mean(a[:,1]==1):5.2f} mean rank {a[:,1].mean():4.1f} best rival {a[:,2].mean():6.0f}")
