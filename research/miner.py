"""Leaderboard miner: reconstruct each run's reward sequence from the public score chart.

Every team sees the same rewards in a run, so each jump in any team's cumulative-score trace is the
exact reward of that round (to the 3 decimals the chart prints). bot-greedy accepts every free round
and bot-random accepts at random, so their accepted rounds are unbiased samples of the distribution;
the union over all teams is biased towards large rewards (DP teams only take the top ~13%).

Polls every POLL seconds and writes one file per run (key = "Last run" timestamp + round label):
    _local/runs/<stamp>__<round>.json  {label, last_run, rewards {round: r}, accepts {team: [rounds]},
                                         finals {team: score}, table [[rank, team, best, latest]]}
    _local/raw/<stamp>__<round>.html.gz  raw page, for re-parsing later
`uv run python research/miner.py [--once]`
"""
import gzip
import html
import json
import os
import re
import sys
import time
import urllib.request

URL = "http://ec2-54-217-141-255.eu-west-1.compute.amazonaws.com:8080"
POLL = 25
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_local")
TRACE_RE = re.compile(r"\{x:\[(.*?)\],y:\[(.*?)\],mode:'lines',name:\"(.*?)\"", re.S)
STAMP_RE = re.compile(r"Last run:\s*([0-9-]+ [0-9:]+)\s*<span[^>]*>(.*?)</span>", re.S)
ROW_RE = re.compile(r"<tr[^>]*><td>(\d+)</td><td>(.*?)</td><td>(.*?)</td><td>(.*?)</td>", re.S)


def fetch():
    with urllib.request.urlopen(URL, timeout=20) as f:
        return f.read().decode("utf-8", "replace")


def parse(page):
    m = STAMP_RE.search(page)
    if not m:
        return None
    last_run, label = m.group(1), html.unescape(m.group(2)).strip()
    rewards, accepts, finals, conflicts = {}, {}, {}, 0
    for xs, ys, name in TRACE_RE.findall(page):
        name = json.loads('"' + name + '"')
        x = [int(float(v)) for v in xs.split(",")]
        y = [float(v) for v in ys.split(",")]
        finals[name] = y[-1]
        acc = []
        for k in range(1, len(y)):
            inc = round(y[k] - y[k - 1], 3)
            if inc != 0.0:
                rnd = x[k]  # y[k] is the score after round x[k]
                acc.append(rnd)
                if rnd in rewards and abs(rewards[rnd] - inc) > 2e-3:
                    conflicts += 1
                rewards.setdefault(rnd, inc)
        accepts[name] = acc
    table = [[int(r), html.unescape(re.sub(r"<[^>]+>", "", t)).strip(), b, l]
             for r, t, b, l in ROW_RE.findall(page)]
    return dict(label=label, last_run=last_run, n_known=len(rewards), conflicts=conflicts,
                rewards={str(k): v for k, v in sorted(rewards.items())}, accepts=accepts,
                finals=finals, table=table, fetched=time.strftime("%Y-%m-%d %H:%M:%S"))


def key_of(run):
    lab = re.sub(r"[^A-Za-z0-9]+", "_", run["label"]).strip("_")[:40]
    return run["last_run"].replace(" ", "T").replace(":", "") + "__" + lab


def save(run, page):
    k = key_of(run)
    path = os.path.join(ROOT, "runs", k + ".json")
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        if old["n_known"] >= run["n_known"] and len(old["accepts"]) >= len(run["accepts"]):
            return None
    os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "raw"), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(run, f)
    os.replace(tmp, path)
    with gzip.open(os.path.join(ROOT, "raw", k + ".html.gz"), "wt") as f:
        f.write(page)
    return k


def main():
    once = "--once" in sys.argv
    while True:
        try:
            page = fetch()
            run = parse(page)
            if run is None:
                print(time.strftime("%H:%M:%S"), "no run on page", flush=True)
            else:
                k = save(run, page)
                if k:
                    print(time.strftime("%H:%M:%S"), "saved", k, f"known={run['n_known']}",
                          f"teams={len(run['accepts'])} conflicts={run['conflicts']}", flush=True)
        except Exception as e:  # keep polling through network hiccups
            print(time.strftime("%H:%M:%S"), "error", repr(e), flush=True)
        if once:
            break
        time.sleep(POLL)


if __name__ == "__main__":
    main()
