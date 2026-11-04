"""Submit solution.py and record the submission.

Usage:
    uv run python submit_tracked.py -n "Mateo 3" -m "short description of what changed"

Each successful submission gets an id (sub-001, sub-002, ...) and:
  - solution.py is committed if it has uncommitted changes
  - a snapshot is saved to submissions/sub-NNN.py
  - a row is appended to submissions/log.csv
  - an annotated git tag sub-NNN is created on the submitted commit,
    plus an alias tag from the name (e.g. "Mateo 3" -> mateo-3)
  - main and the tag are pushed (skip with --no-push)

Compare two submissions:   git diff mateo-1 mateo-2 -- solution.py
Restore an old one:        git checkout mateo-1 -- solution.py
"""
import argparse
import csv
import datetime
import os
import re
import shutil
import subprocess
import sys

from config import TEAM_NAME, PASSWORD, SERVER
from gr_tradinggame.coding.client import Client

ROOT = os.path.dirname(os.path.abspath(__file__))
SUB_DIR = os.path.join(ROOT, "submissions")
LOG = os.path.join(SUB_DIR, "log.csv")
FIELDS = ["id", "name", "time", "commit", "description"]


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def next_id():
    nums = [int(t[4:]) for t in git("tag", "-l", "sub-*").split() if t[4:].isdigit()]
    return f"sub-{max(nums, default=0) + 1:03d}"


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-n", "--name", required=True, help='human name for this submission, e.g. "Mateo 3"')
    p.add_argument("-m", "--message", required=True, help="what this submission changes")
    p.add_argument("--no-push", action="store_true")
    args = p.parse_args()

    if not TEAM_NAME or not PASSWORD or not SERVER:
        sys.exit("ERROR: Edit config.py with your team name, password, and server")

    sub_id = next_id()
    alias = slug(args.name)
    if git("tag", "-l", alias):
        sys.exit(f"ERROR: tag {alias} already exists; pick another name")
    source = open(os.path.join(ROOT, "solution.py")).read()

    # Run local tests before touching git, so a failing solution leaves no trace.
    client = Client(TEAM_NAME, password=PASSWORD, server=SERVER)
    ns = {}
    exec(source, ns)
    client.test(ns["play"])

    if git("status", "--porcelain", "solution.py"):
        git("add", "solution.py")
        git("commit", "-m", f"{args.name} ({sub_id}): {args.message}")
    commit = git("rev-parse", "--short", "HEAD")

    client.submit(source)  # raises on failure
    now = datetime.datetime.now().isoformat(timespec="seconds")

    os.makedirs(SUB_DIR, exist_ok=True)
    shutil.copyfile(os.path.join(ROOT, "solution.py"), os.path.join(SUB_DIR, f"{sub_id}.py"))
    new_log = not os.path.exists(LOG)
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_log:
            w.writeheader()
        w.writerow({"id": sub_id, "name": args.name, "time": now, "commit": commit, "description": args.message})

    git("tag", "-a", sub_id, commit, "-m", f"{args.name} {now} {args.message}")
    git("tag", "-a", alias, commit, "-m", f"{sub_id} {now} {args.message}")
    git("add", "submissions")
    git("commit", "-m", f"Record {args.name} ({sub_id})")

    if not args.no_push:
        git("push", "origin", "HEAD")
        git("push", "origin", sub_id, alias)

    print(f"Recorded {args.name} = {sub_id} (commit {commit}): {args.message}")


if __name__ == "__main__":
    main()
