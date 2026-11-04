import base64
import datetime
import random
import requests
from requests.auth import HTTPBasicAuth
import cloudpickle as pickle
import numpy as np
import timeit

from .blackbox import Plackbox, Blackbox


class Client:
    # game_timeout: average seconds per call allowed in a realistic game. The server benches a team
    # that uses over 60 s in a 2,000-round game on a CPU slower than a typical laptop, so 15 ms here
    # leaves a margin under that.
    def __init__(self, team_name, password, server=None, timeout=0.1, game_timeout=0.015):
        self.team_name = team_name
        self.password = password
        if server:
            self.url = server.rstrip('/')
            if not self.url.startswith('http'):
                self.url = 'http://' + self.url
        else:
            self.url = None
        self.timeout = timeout
        self.game_timeout = game_timeout

    def test(self, source_or_function, verbose=False, strict=False, allow_state=True):
        return self._submit(source_or_function, strict=strict, submit=False, allow_state=allow_state, verbose=verbose)

    def submit(self, source_or_function, allow_state=True):
        return self._submit(source_or_function, strict=False, submit=True, allow_state=allow_state, verbose=False)

    def _submit(self, source_or_function, strict, submit, allow_state, verbose):
        if submit and not self.url:
            print('No server configured. Pass server= to Client().')
            return
        test_cases = [
            ((1.5, 5, 1, 10, 0.0, [], False), "round 1 (empty other_scores)", None),
            ((1.5, 5, 2, 10, 1.3, [(1.1, False), (2.3, True)], False), "round > 1 with other_scores and locked status", None),
            ((1.5, 5, 3, 10, 1.3, [], False), "playing by yourself", None),
            ((1.5, 5, 4, 10, -1.0, [(-1.1, False), (2.3, False)], True), "negative score & locked out", None),
        ]
        if isinstance(source_or_function, str):
            output = source_or_function
            foo = Blackbox(output)
        else:
            output = pickle.dumps(source_or_function)
            foo = Plackbox(output)
        outputs = dict()
        for j in range(3):
            for i, (args, err, hint) in enumerate(test_cases):
                test_str = f'Test {i} ("{err}"): f{args}'
                if j == 0 and verbose:
                    print(test_str)
                    test_str = ''
                try:
                    a = foo(*args)
                    if j == 0:
                        outputs[i] = a
                    else:
                        if a != outputs[i] and not allow_state:
                            print(test_str)
                            raise Exception(f"""You previously returned `{outputs[i]}` and are now returning `{a}`. If this is intentional, set `allow_state=True`""")
                        elif not isinstance(a, (bool, np.bool_)):
                            print(test_str)
                            raise Exception(f"""You should return `False` or `True` but you returned `{a}` of type {type(a)}""")
                        elif strict and hint is not None and a != hint[0]:
                            print(test_str)
                            raise Exception(hint[1])
                except Exception:
                    print(test_str)
                    raise
                if j == 0 and verbose:
                    print(f'\tSuccess -- returned {a}')
        # Speed in conditions like a real game: consecutive rounds, T=2000, lockout 15, 23 other teams.
        n_game_rounds = 300
        locked_until = 0
        tic = timeit.default_timer()
        for t in range(1, n_game_rounds + 1):
            scores = [] if t == 1 else [(10 * t * random.random(), random.random() < 0.3) for _ in range(23)]
            args = (random.expovariate(0.1), 15, t, 2000, 10 * t / 15, scores, t <= locked_until)
            try:
                a = foo(*args)
            except Exception:
                print(f'Game-like input {args}')
                raise
            if a and t > locked_until:
                locked_until = t + 15
            if timeit.default_timer() - tic > self.game_timeout * n_game_rounds:
                break
        per_call = (timeit.default_timer() - tic) / t
        if per_call > self.game_timeout:
            raise Exception(f"""Your code is too slow in a game-like test (2000 rounds, 23 other teams): it should take under {self.game_timeout * 1000:.0f} ms per round on average but took {per_call * 1000:.1f} ms. The server stops calling a team for the rest of a game once it has used 60 s.""")
        tic = timeit.default_timer()
        for j in range(1_000):
            T = random.randint(1, 5000)  # games are 2000 rounds; huge T only slowed down code that sets up per T
            t = int(random.random() * T) + 1
            lockout = int(10 * random.random())
            reward = 100 * random.random()
            score = reward * T * random.random()
            if t == 1:
                scores = []
            else:
                scores = [(reward * T * random.random(), random.random() > 0.5) for _ in range(random.randint(0, 5))]
            is_locked = random.random() > 0.5
            args = (reward, lockout, t, T, score, scores, is_locked)
            try:
                a = foo(*args)
            except Exception:
                print(f'Random input {args}')
                raise
            else:
                if not isinstance(a, (bool, np.bool_)):
                    print(f'Random input {args}')
                    raise Exception(f"""You should return `False` or `True` but returned `{a}` of type {type(a)}""")
            toc = timeit.default_timer()
            if toc - tic > max(1, (j + 1) * self.timeout):
                raise Exception(f"""Your code is too slow. Should take {self.timeout}s per round but took {(toc - tic) / (j + 1):.3f}s""")
        if submit:
            self._do_submit(source_or_function)
        else:
            print("All tests passed. You're ready to call `submit`\n")

    def _do_submit(self, output):
        if isinstance(output, str):
            submission = 's' + base64.b64encode(output.encode('utf-8')).decode('utf-8')
        else:
            submission = 'p' + base64.b64encode(pickle.dumps(output)).decode('utf-8')
        j = {"team": self.team_name, "password": self.password, 'time': str(datetime.datetime.now()), 'submission': submission}
        response = requests.post(
            self.url + '/receive',
            json=j,
            auth=HTTPBasicAuth("colabuser", "secretcolab"),
        )
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            try:
                msg = response.json().get("error", "Unknown error")
            except Exception:
                msg = response.text
            raise Exception(f"Submission failed: {msg}")
        else:
            print('Submission successful:', j['team'], j['time'])
