"""Client dropout / jitter simulation utilities.

Contains:
- The S1..S5 distribution-shift settings (train/test drop probabilities per client).
- ``bernoulli_drop_rids`` for per-client independent Bernoulli dropout.
- ``TestMaskPlanner`` for deterministic, fixed test-time missing masks.
"""
from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple, Union

# Kept for backward compatibility with the original ``train_function.py``.
maxnum_of_clients = 8


def make_S1(p: float = 0.3):
    train = {rid: p for rid in range(maxnum_of_clients)}
    test = {rid: p for rid in range(maxnum_of_clients)}
    return train, test


def make_S2():
    train = {0: 0.8, 1: 0.8, 2: 0.8, 3: 0.8, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1}
    test = {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1}
    return train, test


def make_S3(p: float = 0.9):
    train = {0: p, 1: p, 2: p, 3: p, 4: 1 - p, 5: 1 - p, 6: 1 - p, 7: 1 - p}
    test = {0: 1 - p, 1: 1 - p, 2: 1 - p, 3: 1 - p, 4: p, 5: p, 6: p, 7: p}
    return train, test


def make_S3_1(p: float = 0.9):
    train = {0: p, 1: 1 - p, 2: p, 3: 1 - p, 4: p, 5: 1 - p, 6: p, 7: 1 - p}
    test = {0: 1 - p, 1: p, 2: 1 - p, 3: p, 4: 1 - p, 5: p, 6: 1 - p, 7: p}
    return train, test


def make_S3_2(p: float = 0.9):
    train = {0: p, 1: p, 2: 1 - p, 3: 1 - p, 4: 1 - p, 5: 1 - p, 6: 1 - p, 7: 1 - p}
    test = {0: 1 - p, 1: 1 - p, 2: p, 3: p, 4: p, 5: p, 6: p, 7: p}
    return train, test


def make_S3_3(p: float = 0.9):
    train = {0: 1 - p, 1: 1 - p, 2: 1 - p, 3: 1 - p, 4: p, 5: p, 6: p, 7: p}
    test = {0: p, 1: p, 2: p, 3: p, 4: 1 - p, 5: 1 - p, 6: 1 - p, 7: 1 - p}
    return train, test


def make_S4():
    train = {0: 0.05, 1: 0.1, 2: 0.2, 3: 0.35, 4: 0.5, 5: 0.65, 6: 0.8, 7: 0.9}
    test = {k: min(v + 0.1, 0.95) for k, v in train.items()}
    return train, test


def make_S5():
    train = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
    test = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
    return train, test


SETTINGS = [
    ("S1_p0.3",) + make_S1(0.3),
    ("S2_train缺_test全",) + make_S2(),
    ("S3_train_test互补",) + make_S3(),
    ("S4_longtail_shift",) + make_S4(),
]


def bernoulli_drop_rids(
    maxnum_of_clients: int,
    rng: random.Random,
    drop_probs: Union[float, Sequence[float], Dict[int, float]] = 0.3,
    ensure_at_least_one_present: bool = True,
) -> Tuple[List[int], List[int]]:
    """Independently drop each client via a Bernoulli trial.

    Returns ``(present_rids, missing_rids)``.

    ``drop_probs`` may be:
      - float: same drop probability for every client;
      - list/tuple: ``drop_probs[rid]`` per client;
      - dict: ``{rid: p_drop}``.
    """
    present, missing = [], []

    for rid in range(maxnum_of_clients):
        if isinstance(drop_probs, float):
            p = drop_probs
        elif isinstance(drop_probs, (list, tuple)):
            p = float(drop_probs[rid])
        else:  # dict
            p = float(drop_probs.get(rid, 0.0))

        if rng.random() < (1.0 - p):
            present.append(rid)
        else:
            missing.append(rid)

    if ensure_at_least_one_present and len(present) == 0:
        keep = rng.randrange(maxnum_of_clients)
        present = [keep]
        missing = [rid for rid in range(maxnum_of_clients) if rid != keep]

    return present, missing


class TestMaskPlanner:
    """Deterministic, fixed test-time missing masks.

    The first time a batch index is requested, a plan (present rids + shuffle
    order) is generated lazily using a fixed-seed RNG, then reused forever. This
    replaces the original module-level ``TEST_MASK_STATE`` global dict.
    """

    def __init__(self, sim_seed_test: int, maxnum_of_clients: int):
        self.sim_seed_test = int(sim_seed_test)
        self.maxnum_of_clients = int(maxnum_of_clients)
        self._rng = random.Random(self.sim_seed_test)
        self.plan: List[List[int]] = []
        self.shuffle_plan: List[List[int]] = []

    def get_present_rids(
        self,
        batch_idx: int,
        drop_probs_test: Union[float, Sequence[float], Dict[int, float]],
    ) -> Tuple[List[int], List[int]]:
        while len(self.plan) <= batch_idx:
            present, _ = bernoulli_drop_rids(
                maxnum_of_clients=self.maxnum_of_clients,
                rng=self._rng,
                drop_probs=drop_probs_test,
                ensure_at_least_one_present=True,
            )
            self.plan.append(present)

            indices = list(range(len(present)))
            self._rng.shuffle(indices)
            self.shuffle_plan.append(indices)

        return self.plan[batch_idx], self.shuffle_plan[batch_idx]
