"""CPU-only tests for the contiguous-layout bucketed row allocation."""

import unittest

import torch

from sglang.srt.layers.moe.moe_runner.deep_gemm import (
    _bucket_rows,
    _empty_bucketed_rows,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

COLS = 64
# The boundary (1024), one row past it, an exact power of two, and one row past that.
ROW_CASES = (0, 1, 1024, 1025, 2048, 2049)


class TestBucketRows(CustomTestCase):
    def test_at_or_below_the_floor_is_exact(self):
        for rows in (0, 1, 7, 1023, 1024):
            self.assertEqual(_bucket_rows(rows), rows)

    def test_known_buckets(self):
        # step = 1 << (bit_length - 4), so eight buckets per octave.
        self.assertEqual(_bucket_rows(1025), 1152)
        self.assertEqual(_bucket_rows(2048), 2048)
        self.assertEqual(_bucket_rows(2049), 2304)
        self.assertEqual(_bucket_rows(4096), 4096)

    def test_never_shrinks_and_is_monotonic(self):
        previous = 0
        for rows in range(0, 5000):
            bucket = _bucket_rows(rows)
            self.assertGreaterEqual(bucket, rows)
            self.assertGreaterEqual(bucket, previous)
            previous = bucket

    def test_slack_stays_within_one_eighth(self):
        for rows in range(1025, 20000):
            self.assertLessEqual(_bucket_rows(rows), rows * 1.125)

    def test_buckets_are_few(self):
        # The point of bucketing: distinct allocation sizes, not one per row count.
        distinct = {_bucket_rows(rows) for rows in range(1025, 20000)}
        self.assertLess(len(distinct), 64)


class TestEmptyBucketedRows(CustomTestCase):
    def _check(self, tensor, rows, dtype):
        self.assertEqual(tensor.shape, (rows, COLS))
        self.assertEqual(tensor.dtype, dtype)
        self.assertTrue(tensor.is_contiguous())
        self.assertEqual(tensor.stride(), (COLS, 1))
        self.assertEqual(tensor.storage_offset(), 0)
        held = tensor.untyped_storage().size() // tensor.element_size()
        self.assertEqual(held, _bucket_rows(rows) * COLS)

    def test_shape_strides_and_retained_capacity(self):
        for dtype in (torch.bfloat16, torch.float32):
            for rows in ROW_CASES:
                with self.subTest(dtype=dtype, rows=rows):
                    out = _empty_bucketed_rows(rows, COLS, dtype=dtype, device="cpu")
                    self._check(out, rows, dtype)

    def test_zero_initializes_the_visible_rows(self):
        for rows in ROW_CASES:
            with self.subTest(rows=rows):
                out = _empty_bucketed_rows(
                    rows, COLS, dtype=torch.bfloat16, device="cpu", zero=True
                )
                self._check(out, rows, torch.bfloat16)
                self.assertTrue(torch.equal(out, torch.zeros_like(out)))


if __name__ == "__main__":
    unittest.main()
