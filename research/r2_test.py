"""test.py pointed at research/r2_solution_candidate.py, in each of its modes.

`uv run python research/r2_test.py`
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gr_tradinggame.coding.client import Client  # noqa: E402

import r2_solution_candidate as cand  # noqa: E402

client = Client('YourTeamName', password='YourPassword')
z = np.sort(np.random.default_rng(0).normal(size=400))
for name, target, prior in (("default (EV, no prior)", None, None),
                            ("target, no prior", 150.0, None),
                            ("target, N(0,1) prior", 150.0, list(z)),
                            ("target, prior on exp data (prior dropped)", 3000.0, list(z))):
    cand.TARGET, cand.PRIOR_Q = target, prior
    print(f"--- {name}")
    client.test(cand.Solution().play)
