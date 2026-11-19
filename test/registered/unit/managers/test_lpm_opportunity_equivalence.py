"""Compare optimized gate math with the straightforward pre-optimization reference."""

import copy
import random
import unittest
from array import array
from itertools import chain
from types import SimpleNamespace

import numpy as np

from sglang.srt.managers import lpm_opportunity as new
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
Req = SimpleNamespace

PREFIX_TOKENS = 4096


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
    prevent false sharing. Independent of the scheduling gate cadence.
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
        # The scheduler passes array('q') buffers, which take the NumPy path.
        try:
            buffers = [
                Req(
                    origin_input_ids=array("q", r.origin_input_ids),
                    output_ids=array("q", r.output_ids),
                    extra_key=r.extra_key,
                    cache_salt=r.cache_salt,
                )
                for r in rs
            ]
        except (OverflowError, TypeError):
            return
        arrays = [
            Req(
                origin_input_ids=np.array(r.origin_input_ids, dtype=np.int64),
                output_ids=np.array(r.output_ids, dtype=np.int64),
                extra_key=r.extra_key,
                cache_salt=r.cache_salt,
            )
            for r in buffers
        ]
        expected = clean(old.competition_opportunity(rs, kv))
        for converted in (buffers, arrays):
            self.assertEqual(
                expected, clean(new.competition_opportunity(converted, kv))
            )
        if "pairs" in expected:
            expected["pairs"] = []
        for converted in (buffers, arrays):
            self.assertEqual(
                expected,
                clean(new.competition_opportunity(converted, kv, include_pairs=False)),
            )

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

    def test_long_sequences(self):
        # Mismatches land on both sides of the block edges the comparison uses.
        rng = random.Random(11)
        # Mismatches land on both sides of every place the scan changes step:
        # the end of the Python probe and the ends of the growing NumPy blocks.
        edges, offset, block = [0, 1, 20000], new.PROBE_TOKENS, 256
        for _ in range(4):
            edges += [offset - 1, offset, offset + 1]
            offset += block
            block *= 4
        for _ in range(40):
            base = [rng.randrange(1000) for _ in range(PREFIX_TOKENS + 24000)]
            rs = []
            for _ in range(rng.randrange(2, 12)):
                ts = list(base[: PREFIX_TOKENS + rng.choice(edges)])
                cut = PREFIX_TOKENS + rng.choice(edges)
                if cut < len(ts):
                    ts[cut] += 1
                rs.append(req(ts, split=rng.randrange(len(ts) + 1)))
            self.check(rs, rng.choice([4096, 8192, 32768, 65536]))

    def test_non_int64_inputs_keep_tuple_semantics(self):
        # Values NumPy int64 would change or reject must still compare as Python objects.
        for head in ([1.1] * 4096, ["1"] * 4096, [2**63] * 4096, [-(2**63) - 1] * 4096):
            rs = [req(head), req([2] * 4096), req([1] * 4096)]
            self.check(rs, 2048)
        wide = np.array([2**64 - 1] * 4096, dtype=np.uint64)
        rs = [req([-1] * 4096), req([2] * 4096)]
        rs.append(
            Req(origin_input_ids=wide, output_ids=[], extra_key=None, cache_salt=None)
        )
        self.assertEqual(
            clean(old.competition_opportunity(rs, 2048)),
            clean(new.competition_opportunity(rs, 2048)),
        )

    def test_mixed_short_and_long_intervening(self):
        # Long endpoints keep these windows above the NumPy threshold.
        rng = random.Random(42)
        for length in (256, 257, 40000):
            seqs = (
                [[7] * 20000]
                + [[rng.randrange(10000)] for _ in range(61)]
                + [[9000] * length, [7] * 20000]
            )
            self.check([req(s) for s in seqs], 4096)

    def test_early_divergence(self):
        # Requests that differ at their first token, in a window above the threshold.
        ids = list(range(100, 162))
        random.Random(42).shuffle(ids)
        for n in (396, 397, 512):
            seqs = [[7] * 4096] + [[i] * n for i in ids] + [[7] * 4096]
            self.check([req(s) for s in seqs], 8192)
        self.check([req([100 + i % 32] * 4096) for i in range(64)], 8192)

    def test_long_windows_take_the_numpy_path(self):
        taken = []
        original = new._vectorizable

        def spy(requests):
            taken.append(original(requests))
            return taken[-1]

        new._vectorizable = spy
        try:
            self.test_long_sequences()
            self.test_mixed_short_and_long_intervening()
        finally:
            new._vectorizable = original
        self.assertTrue(any(taken))

    def test_mismatch_just_after_a_block(self):
        # A scan that skipped one token here would report a longer shared prefix.
        offset = PREFIX_TOKENS + new.PROBE_TOKENS + 256
        changed = [7] * 20000
        changed[offset] = 8
        self.check([req([7] * 20000), req([9] * 20000), req(changed)], 8192)

    def test_unsized_iterables_keep_tuple_semantics(self):
        rs = [
            Req(
                origin_input_ids=iter([7] * 6144),
                output_ids=iter(()),
                extra_key=None,
                cache_salt=None,
            ),
            Req(
                origin_input_ids=iter([9] * 20000),
                output_ids=iter(()),
                extra_key=None,
                cache_salt=None,
            ),
            Req(
                origin_input_ids=iter([7] * 6144),
                output_ids=iter(()),
                extra_key=None,
                cache_salt=None,
            ),
        ]
        result = clean(new.competition_opportunity(rs, 4096))
        self.assertTrue(result["allowed"])
        self.assertEqual(result["risk_tokens"], 6144)

    def test_order_with_wide_and_negative_tokens(self):
        # Order must follow token values, not their byte layout or sign bit.
        rng = random.Random(5)
        values = [1, 255, 256, 257, 2**40, 2**62, -1, -256, -(2**40), -(2**63)]
        for _ in range(30):
            middle = []
            for _ in range(rng.randrange(4, 20)):
                stem = [rng.choice(values) for _ in range(rng.randrange(0, 3))]
                middle.append(
                    stem + [rng.choice(values) for _ in range(rng.randrange(1, 4))]
                )
            seqs = [[3] * 20000] + middle + [[3] * 20000]
            self.check([req(s) for s in seqs], rng.choice([4096, 20000, 30000]))


if __name__ == "__main__":
    unittest.main()
