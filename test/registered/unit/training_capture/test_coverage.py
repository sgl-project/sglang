"""Exact KV rectangle coverage against an independent per-cell oracle."""

import itertools
import random
import unittest
from types import SimpleNamespace

import msgspec
from sglang.srt.training_capture.protocol import (
    ContractError,
    _coverage,
    canonical_bytes,
    decode_manifest,
    validate_manifest,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.training_capture_utils import make_snapshot

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def rectangles(regions):
    return [
        SimpleNamespace(token_range=(t0, t1), head_range=(h0, h1))
        for t0, t1, h0, h1 in regions
    ]


def cell_oracle(regions, n, heads):
    if not regions:
        return None
    end = max(t1 for _, t1, _, _ in regions)
    if end not in (n - 1, n):
        return None
    cells = [[0] * heads for _ in range(end)]
    for t0, t1, h0, h1 in regions:
        for token in range(t0, t1):
            for head in range(h0, h1):
                cells[token][head] += 1
    return end if all(value == 1 for row in cells for value in row) else None


class TestKVCoverage(CustomTestCase):
    def assert_coverage(self, regions, n, heads):
        expected = cell_oracle(regions, n, heads)
        if expected is None:
            with self.assertRaises(ContractError, msg=str((regions, n, heads))):
                _coverage(rectangles(regions), n, heads)
        else:
            self.assertEqual(_coverage(rectangles(regions), n, heads), expected)

    def test_exhaustive_small_rectangle_multisets(self):
        choices = [
            (t0, t1, h0, h1)
            for t0, t1 in itertools.combinations(range(3), 2)
            for h0, h1 in itertools.combinations(range(3), 2)
        ]
        for count in range(5):
            for regions in itertools.combinations_with_replacement(choices, count):
                for n in (2, 3, 4):
                    self.assert_coverage(regions, n, heads=2)

    def test_random_tilings_and_corruptions_match_cell_oracle(self):
        rng = random.Random(571)
        for _ in range(500):
            end, heads = rng.randrange(2, 12), rng.randrange(2, 9)
            regions = [(0, end, 0, heads)]
            for _ in range(20):
                index = rng.randrange(len(regions))
                t0, t1, h0, h1 = regions[index]
                split_token = rng.choice((False, True))
                if split_token and t1 - t0 > 1:
                    mid = rng.randrange(t0 + 1, t1)
                    split = [(t0, mid, h0, h1), (mid, t1, h0, h1)]
                elif not split_token and h1 - h0 > 1:
                    mid = rng.randrange(h0 + 1, h1)
                    split = [(t0, t1, h0, mid), (t0, t1, mid, h1)]
                else:
                    continue
                regions[index : index + 1] = split
            rng.shuffle(regions)
            for n in (end, end + 1):
                self.assert_coverage(regions, n, heads)
                self.assert_coverage(regions[:-1], n, heads)
                self.assert_coverage(regions + [rng.choice(regions)], n, heads)
            arbitrary = [
                (
                    *sorted(rng.sample(range(end + 1), 2)),
                    *sorted(rng.sample(range(heads + 1), 2)),
                )
                for _ in range(rng.randrange(1, 20))
            ]
            self.assert_coverage(arbitrary, end + 1, heads)

    def test_shared_endpoints_and_changed_head_partition(self):
        regions = [
            (0, 2, 0, 2),
            (0, 1, 2, 5),
            (1, 3, 2, 4),
            (1, 3, 4, 5),
            (2, 3, 0, 2),
        ]
        for permutation in itertools.permutations(regions):
            self.assert_coverage(permutation, n=4, heads=5)
        # Equal summed area does not prove coverage: these overlap and leave gaps.
        self.assert_coverage([(0, 2, 0, 2), (0, 2, 0, 2)], n=3, heads=4)
        self.assert_coverage([(0, 1, 0, 4), (2, 3, 0, 4)], n=4, heads=4)

    def test_manifest_roundtrip_and_missing_chunk(self):
        manifest, _ = make_snapshot(128)
        expected = manifest.sequence.total_length - 1
        shuffled = list(manifest.objects)
        random.Random(3).shuffle(shuffled)
        manifest = msgspec.structs.replace(manifest, objects=shuffled)
        self.assertEqual(validate_manifest(manifest), expected)
        self.assertEqual(
            validate_manifest(decode_manifest(canonical_bytes(manifest))), expected
        )
        missing = next(
            obj for obj in shuffled if obj.kind == "kv" and obj.token_range[0] > 0
        )
        broken = msgspec.structs.replace(
            manifest,
            objects=[obj for obj in shuffled if obj is not missing],
            total_tensor_bytes=manifest.total_tensor_bytes - missing.nbytes,
        )
        with self.assertRaises(ContractError):
            validate_manifest(broken)
        with self.assertRaises(ContractError):
            decode_manifest(canonical_bytes(broken))


if __name__ == "__main__":
    unittest.main()
