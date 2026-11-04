"""Round-3 arena: exclusive claims where a collision pays NOBODY.

Rules simulated (ETH Oct 2026 confirmed version):
  * all teams see the same reward sequence (Exponential(mean 10), T=2000, L=15);
  * exactly one accepter -> credited with the reward and locked for L rounds;
  * two or more accepters -> nobody is credited and all of them are locked for L rounds;
  * no bots: only teams.
other_scores is rebuilt for every team every turn; `order` controls its list order:
  "shuffle" (new random order every turn and team), "fixed" (one random order per game), "stable".
`flag_shift`: 1 -> lock flag at t-1 means "locked next round" (post-action), 0 -> pre-action,
  -1 -> the server never sets the flags (robustness check).

Players are built from picklable specs so games can run in a process pool:
  ("file", path, {overrides})  -> fresh Solution().play from that file
  ("bot", name, {kwargs})       -> one of the BOTS below
  "SELF"                         -> inside a field: a copy of the strategy under test
"""
import importlib.util, math, os, random, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
T_DEF, L_DEF, MU = 2000, 15, 10.0


# ------------------------------------------------------------------------------ helpers
_MOD_CACHE = {}


def _load_module(path):
    path = str(path)
    if path not in _MOD_CACHE:
        spec = importlib.util.spec_from_file_location(f"m{abs(hash(path))}_{time.time_ns()}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MOD_CACHE[path] = mod
    return _MOD_CACHE[path]


def dp1_gaps(T=T_DEF, L=L_DEF, mu=MU):
    """Round-1 optimal gap by rounds remaining n (solo exponential DP). gap[n] ~ 20.1 mid-game."""
    W = [0.0] * (T + 1)
    for n in range(1, T + 1):
        acc = W[n - 1 - L] if n - 1 - L > 0 else 0.0
        W[n] = W[n - 1] + math.exp(-(W[n - 1] - acc))
    g = [0.0] * (T + 1)
    for n in range(1, T + 1):
        acc = W[n - 1 - L] if n - 1 - L > 0 else 0.0
        g[n] = mu * (W[n - 1] - acc)
    return g


_DP1 = dp1_gaps()


# ------------------------------------------------------------------------------ bots
# Every bot is a closure play(r, L, t, T, s, others, locked) -> bool. Each gets a private rng.

def bot_dp1(scale=1.0, jitter=0.0, rng=None):
    """Round-1 optimal DP (cut ~20.1). Most teams arrive in round 3 with this."""
    m = scale * (math.exp(rng.normal(0, jitter)) if jitter else 1.0)
    return lambda r, L, t, T, s, o, lk: (not lk) and r >= _DP1[T - t + 1] * m


def bot_cut(c=20.0, rng=None):
    return lambda r, L, t, T, s, o, lk: (not lk) and r >= c


def bot_greedy(rng=None):
    return lambda r, L, t, T, s, o, lk: not lk


def bot_band(lo=9.0, hi=16.0, p=1.0, rng=None):
    if p >= 1.0:
        return lambda r, L, t, T, s, o, lk: (not lk) and lo <= r <= hi
    return lambda r, L, t, T, s, o, lk: (not lk) and lo <= r <= hi and rng.random() < p


def bot_random(lo=0.0, p=0.3, rng=None):
    return lambda r, L, t, T, s, o, lk: (not lk) and r >= lo and rng.random() < p


def bot_switch(first=None, second=None, at=1000, rng=None):
    """Behaves as bot `first` until round `at`, then as `second` (the field changing mid-run)."""
    a, b = make_bot(first, rng), make_bot(second, rng)
    return lambda r, L, t, T, s, o, lk: (a if t < at else b)(r, L, t, T, s, o, lk)


def bot_bandit(rng=None, epoch=80):
    """Order-invariant adaptive team: epsilon-greedy over bands, scored by own earnings per round."""
    arms = [(20, 1e9), (14, 1e9), (9, 16), (12, 20), (6, 12), (0, 8), (16, 30), (25, 1e9)]
    st = {"arm": int(rng.integers(len(arms))), "start": 1, "s0": 0.0, "val": [None] * len(arms), "n": [0] * len(arms)}

    def play(r, L, t, T, s, o, lk):
        if t == 1:
            st.update(arm=int(rng.integers(len(arms))), start=1, s0=0.0, val=[None] * len(arms), n=[0] * len(arms))
        if t - st["start"] >= epoch:
            a = st["arm"]; v = (s - st["s0"]) / (t - st["start"])
            st["n"][a] += 1
            st["val"][a] = v if st["val"][a] is None else 0.6 * st["val"][a] + 0.4 * v
            if rng.random() < 0.2 or all(x is None for x in st["val"]):
                st["arm"] = int(rng.integers(len(arms)))
            else:
                st["arm"] = max(range(len(arms)), key=lambda i: -1 if st["val"][i] is None else st["val"][i])
            st["start"], st["s0"] = t, s
        lo, hi = arms[st["arm"]]
        return (not lk) and lo <= r <= hi
    return play


def bot_follow(rng=None, window=60):
    """Order-invariant adaptive team: claims rewards inside the inter-quartile range of rewards that
    rivals recently won (detected from the rise in the sum of other_scores)."""
    st = {"prev": None, "prev_r": None, "won": []}
    jit = math.exp(rng.normal(0, 0.1))

    def play(r, L, t, T, s, o, lk):
        if t == 1:
            st.update(prev=None, prev_r=None, won=[])
        tot = sum(x for x, _ in o) if o else 0.0
        if st["prev"] is not None and st["prev_r"] and tot - st["prev"] > 0.5 * st["prev_r"]:
            st["won"].append(st["prev_r"])
            st["won"] = st["won"][-window:]
        st["prev"], st["prev_r"] = tot, r
        if len(st["won"]) < 8:
            return (not lk) and r >= 20
        w = sorted(st["won"])
        lo, hi = w[len(w) // 4] * jit, w[(3 * len(w)) // 4] * jit
        return (not lk) and lo <= r <= hi
    return play


def bot_undercut(rng=None):
    """Order-invariant: tracks the median won reward and claims just below it (undercuts the crowd)."""
    st = {"prev": None, "prev_r": None, "won": []}

    def play(r, L, t, T, s, o, lk):
        if t == 1:
            st.update(prev=None, prev_r=None, won=[])
        tot = sum(x for x, _ in o) if o else 0.0
        if st["prev"] is not None and st["prev_r"] and tot - st["prev"] > 0.5 * st["prev_r"]:
            st["won"].append(st["prev_r"]); st["won"] = st["won"][-80:]
        st["prev"], st["prev_r"] = tot, r
        if len(st["won"]) < 8:
            return (not lk) and 9 <= r <= 16
        m = sorted(st["won"])[len(st["won"]) // 2]
        return (not lk) and 0.6 * m <= r <= 0.95 * m
    return play


def bot_rps(rng=None, c=8.0, p=0.75, bw=2.0):
    """ETH-style contrarian: claim iff r * P(solo) >= c, P(solo) per reward bin from the order-free
    count of new locks; claims with prob p, random backoff after a failed claim."""
    st = {}
    cj = c * math.exp(rng.normal(0, 0.1))

    def play(r, L, t, T, s, o, lk):
        if t == 1:
            st.update(n=[2.0] * 30, k=[1.0] * 30, prev=None, prev_r=None, mine=None, back=0)
        nl = sum(1 for _, f in o if f)
        if st["prev"] is not None and st["prev_r"] is not None:
            b = min(int(st["prev_r"] / bw), 29)
            st["n"][b] = 0.99 * st["n"][b] + 1.0
            st["k"][b] = 0.99 * st["k"][b] + (1.0 if nl <= st["prev"] else 0.0)
        if st["mine"] is not None:
            if s - st["mine"][1] < 0.5 * st["mine"][0]:
                st["back"] = int(rng.integers(0, 8))
            st["mine"] = None
        st["prev"], st["prev_r"] = nl, r
        if lk:
            return False
        if st["back"] > 0:
            st["back"] -= 1
            return False
        b = min(int(r / bw), 29)
        if r * st["k"][b] / st["n"][b] >= cj and rng.random() < p:
            st["mine"] = (r, s)
            return True
        return False
    return play


BOTS = {"rps": bot_rps, "dp1": bot_dp1, "cut": bot_cut, "greedy": bot_greedy, "band": bot_band, "random": bot_random,
        "switch": bot_switch, "bandit": bot_bandit, "follow": bot_follow, "undercut": bot_undercut}


def make_bot(spec, rng):
    name, kw = spec
    return BOTS[name](rng=rng, **kw)


def make_player(spec, rng):
    kind = spec[0]
    if kind == "bot":
        return make_bot((spec[1], spec[2] if len(spec) > 2 else {}), rng)
    if kind == "file":
        mod = _load_module(spec[1])
        sol = mod.Solution()
        for k, v in (spec[2] if len(spec) > 2 else {}).items():
            setattr(sol, k, v)          # instance-level parameter overrides (candidate reads self.X)
        return sol.play
    raise ValueError(spec)


# ------------------------------------------------------------------------------ engine
def run_game(players, seed=0, T=T_DEF, L=L_DEF, order="shuffle", flag_shift=1, rewards=None, collect=False):
    g = np.random.default_rng(seed)
    rw = g.exponential(MU, T) if rewards is None else rewards
    prng = random.Random(seed * 7919 + 1)
    n = len(players)
    score = [0.0] * n
    lock_until = [0] * n
    snap = None
    perms = [list(range(n)) for _ in range(n)]
    if order == "fixed":
        for p in perms:
            prng.shuffle(p)
    won_by = [] if collect else None
    for t in range(1, T + 1):
        r = float(rw[t - 1])
        claims = []
        for i in range(n):
            locked = t <= lock_until[i]
            if snap is None:
                others = []
            else:
                if order == "shuffle":
                    idx = [j for j in range(n) if j != i]
                    prng.shuffle(idx)
                else:
                    idx = [j for j in perms[i] if j != i]
                others = [snap[j] for j in idx]
            a = players[i](r, L, t, T, score[i], others, locked)
            if a and not locked:
                claims.append(i)
        pre = [t <= lock_until[j] for j in range(n)]
        for i in claims:
            lock_until[i] = t + L
        if len(claims) == 1:
            score[claims[0]] += r
        if collect:
            won_by.append((r, claims[0] if len(claims) == 1 else -1, len(claims)))
        if flag_shift == 1:
            snap = [(score[j], t + 1 <= lock_until[j]) for j in range(n)]
        elif flag_shift == 0:
            snap = [(score[j], pre[j]) for j in range(n)]
        else:
            snap = [(score[j], False) for j in range(n)]
    return (score, won_by) if collect else score


# ------------------------------------------------------------------------------ fields
# Each field: list of (label, spec) for the 23 rivals (the strategy under test sits in seat 0).
def _rep(k, label, spec):
    return [(label, spec)] * k


def _fill(lst, n=23):
    """Pad with a generic mix up to n rivals."""
    mix = [("dp1", ("bot", "dp1", {"jitter": 0.03})), ("band", ("bot", "band", {"lo": 9, "hi": 16})),
           ("greedy", ("bot", "greedy")), ("cut14", ("bot", "cut", {"c": 14}))]
    i = 0
    while len(lst) < n:
        lst.append(mix[i % len(mix)]); i += 1
    return lst[:n]


DP1 = ("bot", "dp1", {"jitter": 0.03})
SOL = ("file", str(ROOT / "solution.py"), {})


def fields():
    F = {}
    F["solo"] = []
    F["eth_early"] = (_rep(12, "dp1", DP1) + _rep(3, "greedy", ("bot", "greedy")) +
                      [("band", ("bot", "band", {"lo": 10, "hi": 16})), ("band", ("bot", "band", {"lo": 12, "hi": 18}))] +
                      _rep(2, "cut14", ("bot", "cut", {"c": 14})) + _rep(2, "random", ("bot", "random", {"lo": 5, "p": 0.3})) +
                      _rep(2, "follow", ("bot", "follow")))
    F["bot_flood"] = _rep(15, "greedy", ("bot", "greedy")) + _rep(8, "dp1", DP1)
    F["band_crowd"] = (_rep(6, "band9_16", ("bot", "band", {"lo": 9, "hi": 16})) + _rep(4, "band8_14", ("bot", "band", {"lo": 8, "hi": 14})) +
                       _rep(3, "band12_20", ("bot", "band", {"lo": 12, "hi": 20})) + _rep(4, "small", ("bot", "band", {"lo": 0, "hi": 8})) +
                       _rep(4, "dp1", DP1) + _rep(2, "greedy", ("bot", "greedy")))
    F["dp1_wall"] = _rep(23, "dp1", DP1)
    F["adaptive"] = (_rep(8, "bandit", ("bot", "bandit")) + _rep(6, "follow", ("bot", "follow")) +
                     _rep(4, "undercut", ("bot", "undercut")) + _rep(5, "dp1", DP1))
    F["mid_switch"] = (_rep(8, "dp1->greedy", ("bot", "switch", {"first": ("dp1", {}), "second": ("greedy", {}), "at": 900})) +
                       _rep(6, "dp1->band", ("bot", "switch", {"first": ("dp1", {}), "second": ("band", {"lo": 8, "hi": 15}), "at": 700})) +
                       _rep(5, "band->dp1", ("bot", "switch", {"first": ("band", {"lo": 9, "hi": 16}), "second": ("dp1", {}), "at": 1200})) +
                       _rep(4, "greedy->off", ("bot", "switch", {"first": ("greedy", {}), "second": ("cut", {"c": 1e9}), "at": 500})))
    F["sol_py"] = _rep(8, "solution.py", SOL) + _fill([], 15)
    F["sparse"] = [("dp1", DP1), ("band", ("bot", "band", {"lo": 9, "hi": 16})), ("greedy", ("bot", "greedy")), ("follow", ("bot", "follow"))]
    for k in (2, 4, 8, 12):
        F[f"self{k}"] = _rep(k - 1, "SELF", "SELF") + _fill(_rep(6, "dp1", DP1) + _rep(2, "follow", ("bot", "follow")), 23 - (k - 1))
    F["public_band"] = (_rep(10, "band9_16", ("bot", "band", {"lo": 9, "hi": 16})) + _rep(6, "dp1", DP1) +
                        _rep(4, "greedy", ("bot", "greedy")) + _rep(3, "follow", ("bot", "follow")))
    F["late_flood"] = (_rep(10, "off->greedy", ("bot", "switch", {"first": ("cut", {"c": 1e9}), "second": ("greedy", {}), "at": 1000})) +
                       _rep(8, "dp1", DP1) + _rep(5, "band", ("bot", "band", {"lo": 9, "hi": 16})))
    CANDF = str(ROOT / "research" / "r3_solution_candidate.py")
    F["smart_mix"] = (_rep(4, "candA", ("file", CANDF, {"TAU": 0.3, "DECAY": 0.99, "JITTER": 0.2})) +
                      _rep(4, "candB", ("file", CANDF, {"CORR": False, "BACKOFF": 0, "P_MAX": 0.8})) +
                      _rep(5, "solution.py", SOL) + _rep(4, "follow", ("bot", "follow")) +
                      _rep(3, "bandit", ("bot", "bandit")) + _rep(3, "dp1", DP1))
    B75 = ("bot", "band", {"lo": 9, "hi": 16, "p": 0.75})
    B816 = ("bot", "band", {"lo": 8, "hi": 16, "p": 0.75})
    RPS = ("bot", "rps")
    F["herd_band"] = (_rep(8, "band9_16p75", B75) + _rep(4, "band8_16p75", B816) + _rep(4, "rps", RPS) +
                        _rep(4, "dp1", DP1) + _rep(3, "greedy", ("bot", "greedy")))
    F["band_sat"] = (_rep(8, "band9_16", ("bot", "band", {"lo": 9, "hi": 16})) + _rep(7, "band9_16p75", B75) +
                     _rep(8, "dp1", DP1))
    F["herd_rps"] = _rep(12, "rps", RPS) + _rep(5, "band9_16p75", B75) + _rep(3, "SELF", "SELF") + _rep(3, "dp1", DP1)
    F["herd_mix"] = _rep(6, "SELF", "SELF") + _rep(6, "band9_16p75", B75) + _rep(6, "rps", RPS) + _rep(5, "dp1", DP1)
    F["self24"] = _rep(23, "SELF", "SELF")
    F["self12_bandit"] = _rep(11, "SELF", "SELF") + _rep(6, "bandit", ("bot", "bandit")) + _rep(6, "follow", ("bot", "follow"))
    return F


# ------------------------------------------------------------------------------ evaluation
def play_one(args):
    """(cand_spec, field_name, field, seed, order, flag_shift) -> dict with seat-0 score and ranks."""
    cand, fname, field, seed, order, fs = args
    rng = np.random.default_rng(seed * 1000003 + 17)
    players, labels = [make_player(cand, rng)], ["CAND"]
    for lab, spec in field:
        players.append(make_player(cand if spec == "SELF" else spec, rng))
        labels.append("CAND" if spec == "SELF" else lab)
    sc = run_game(players, seed=seed, order=order, flag_shift=fs)
    s0 = sc[0]
    rank = 1 + sum(1 for x in sc[1:] if x > s0)
    cand_mean = float(np.mean([s for s, l in zip(sc, labels) if l == "CAND"]))
    by = {}
    for s, l in zip(sc, labels):
        by.setdefault(l, []).append(s)
    return {"field": fname, "seed": seed, "score": s0, "rank": rank, "n": len(sc), "cand_mean": cand_mean,
            "by": {k: float(np.mean(v)) for k, v in by.items()}}


def evaluate(cands, field_names=None, seeds=range(8), order="shuffle", flag_shift=1, procs=None):
    """cands: {name: spec}. Returns {name: {field: [result dicts]}} (same seeds -> same rewards)."""
    from multiprocessing import Pool
    F = fields()
    field_names = field_names or list(F)
    jobs, keys = [], []
    for cn, cs in cands.items():
        for fn in field_names:
            for s in seeds:
                jobs.append((cs, fn, F[fn], s, order, flag_shift)); keys.append(cn)
    with Pool(procs or max(1, (os.cpu_count() or 4) - 2)) as pool:
        res = pool.map(play_one, jobs, chunksize=1)
    out = {}
    for k, r in zip(keys, res):
        out.setdefault(k, {}).setdefault(r["field"], []).append(r)
    return out


def summarize(out, field_names=None):
    lines = []
    names = list(out)
    fns = field_names or list(next(iter(out.values())))
    lines.append("| field | " + " | ".join(names) + " |")
    lines.append("|---|" + "---|" * len(names))
    for fn in fns:
        cells = []
        for cn in names:
            v = np.array([r["score"] for r in out[cn][fn]])
            rk = np.mean([r["rank"] for r in out[cn][fn]])
            cells.append(f"{v.mean():.0f} (p10 {np.percentile(v, 10):.0f}, r{rk:.1f})")
        lines.append(f"| {fn} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
