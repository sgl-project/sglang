"""Compare optimized gate math with the straightforward pre-optimization reference."""

import copy
import random
import unittest
from itertools import chain
from types import SimpleNamespace

from sglang.srt.managers import lpm_opportunity as new
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
Req = SimpleNamespace

PREFIX_TOKENS = 4096
RECHECK_SECONDS = 1.0


def _lcp(a, b):
    for index, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return index
    return min(len(a), len(b))


def _union_size(sequences):
    ordered = sorted(sequences)
    if not ordered:
        return 0
    return len(ordered[0]) + sum(
        len(right) - _lcp(left, right) for left, right in zip(ordered, ordered[1:])
    )


def reference_competition_opportunity(requests, capacity):
    """Queue-only interference proxy, not predicted cache savings.

    Count long-prefix reuse pairs whose intervening unique input working set,
    including the target prefix, exceeds total KV capacity. Namespace boundaries
    prevent false sharing. Called only after the old gate, at most once/second.
    """
    import time
    from collections import defaultdict

    start = time.perf_counter()
    if capacity <= 0 or len(requests) > 64:
        return dict(allowed=False, reason="invalid_capacity_or_window")
    sequences = [tuple(chain(r.origin_input_ids, r.output_ids)) for r in requests]
    namespaces = [(r.extra_key, r.cache_salt) for r in requests]
    groups = defaultdict(list)
    for index, tokens in enumerate(sequences):
        if len(tokens) >= PREFIX_TOKENS:
            groups[(namespaces[index], tokens[:PREFIX_TOKENS])].append(index)
    risk = 0
    pairs = []
    for positions in groups.values():
        for left, right in zip(positions, positions[1:]):
            prefix = _lcp(sequences[left], sequences[right])
            intervening = defaultdict(list)
            for index in range(left + 1, right):
                intervening[namespaces[index]].append(sequences[index])
            footprint = sum(_union_size(v) for v in intervening.values())
            overlap = max(
                (
                    min(prefix, _lcp(sequences[left], v))
                    for v in intervening[namespaces[left]]
                ),
                default=0,
            )
            working_set = footprint + prefix - overlap
            threatened = working_set > capacity
            if threatened:
                risk += prefix
            pairs.append(
                dict(
                    left=left,
                    right=right,
                    prefix=prefix,
                    working_set=working_set,
                    threatened=threatened,
                )
            )
    return dict(
        allowed=2 * risk >= 3 * capacity,
        risk_tokens=risk,
        ratio=risk / capacity,
        threshold=1.5,
        capacity=capacity,
        pairs=pairs,
        elapsed_s=time.perf_counter() - start,
    )


old = SimpleNamespace(
    competition_opportunity=reference_competition_opportunity,
)


def req(tokens, ns=None, salt=None, split=None):
    split = len(tokens) if split is None else split
    return Req(
        origin_input_ids=tokens[:split],
        output_ids=tokens[split:],
        extra_key=ns,
        cache_salt=salt,
    )


def clean(result):
    return {k: v for k, v in result.items() if k != "elapsed_s"}


class Equivalence(unittest.TestCase):
    def check(self, rs, kv):
        before = copy.deepcopy(rs)
        expected = clean(old.competition_opportunity(rs, kv))
        self.assertEqual(expected, clean(new.competition_opportunity(rs, kv)))
        if "pairs" in expected:
            expected["pairs"] = []
        self.assertEqual(
            expected, clean(new.competition_opportunity(rs, kv, include_pairs=False))
        )
        self.assertEqual([vars(x) for x in before], [vars(x) for x in rs])

    def test_boundaries(self):
        for rs in [
            [],
            [req([])],
            [req([1] * 4095), req([1] * 4095)],
            [req([1] * 4096), req([2] * 4096), req([1] * 4096)],
            [req([1] * 4096)] * 65,
        ]:
            for kv in [-1, 0, 1, 2730, 2731, 8191, 8192, 8193]:
                self.check(rs, kv)

    def test_exact_thresholds(self):
        rs = [req([1] * 6144), req([2] * 8192), req([1] * 6144)]
        self.check(rs, 4096)
        self.assertTrue(new.competition_opportunity(rs, 4096)["allowed"])
        self.check(rs, 4097)
        self.assertFalse(new.competition_opportunity(rs, 4097)["allowed"])
        rs = [req([1] * 4096), req([1] * 4096), req([2] * 12288)]
        self.check(rs, 8192)
        rs[-1].output_ids.append(2)
        self.check(rs, 8192)

    def test_no_cross_call_cache(self):
        rs = [req([1] * 4096), req([2] * 8192), req([1] * 4096)]
        self.check(rs, 2048)
        rs[-1].origin_input_ids[0] = 3
        self.check(rs, 2048)
        self.assertEqual(new.competition_opportunity(rs, 2048)["risk_tokens"], 0)

    def test_namespaces_outputs_and_duplicates(self):
        self.check(
            [
                req([1] * 5000, "a", "x", 4000),
                req([1] * 5000, "b", "x"),
                req([2] * 7000),
                req([1] * 5000, "a", "x", 3000),
                req([1] * 5000, "a", "y"),
            ],
            4096,
        )

    def test_randomized(self):
        rng = random.Random(7)
        for _ in range(100):
            rs = []
            for i in range(rng.randrange(2, 30)):
                ts = [rng.randrange(4)] * rng.choice([0, 4095, 4096, 4200]) + [
                    rng.randrange(10) for _ in range(rng.randrange(20))
                ]
                rs.append(
                    req(
                        ts,
                        rng.choice([None, "a"]),
                        rng.choice([None, "salt"]),
                        rng.randrange(len(ts) + 1),
                    )
                )
            self.check(rs, rng.choice([2048, 4096, 8192, 32768, 36864]))


if __name__ == "__main__":
    unittest.main()
