"""Unit tests for the batched D2H copy helper in managers/utils.py.

``EmbeddingBatchResult.copy_to_cpu`` copies one embedding tensor per request to
pinned host memory. ``_batched_async_d2h`` collapses that into a single
concatenated copy when the tensors allow it. These tests pin the concatenability
rules (which is where an incorrect batch would silently corrupt results) and, on
CUDA, that the batched path is value-identical to the per-tensor path.
"""

import unittest

import torch

from sglang.srt.managers.utils import (
    _async_d2h,
    _batched_async_d2h,
    _can_batch_d2h,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _cuda(*shape, dtype=torch.float32):
    return torch.randn(*shape, dtype=dtype, device="cuda")


class TestCanBatchD2H(CustomTestCase):
    def test_cpu_tensors_never_batch(self):
        # A CPU torch.cat would add a copy the per-tensor path does not do.
        self.assertFalse(_can_batch_d2h([torch.randn(2, 4), torch.randn(3, 4)]))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_uniform_2d_batches(self):
        # The multi-item scoring shape: one [num_items, cols] tensor per request.
        self.assertTrue(_can_batch_d2h([_cuda(2, 4), _cuda(3, 4), _cuda(1, 4)]))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_uniform_1d_batches(self):
        self.assertTrue(_can_batch_d2h([_cuda(4), _cuda(4)]))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_ragged_last_dim_does_not_batch(self):
        # Per-request matryoshka truncation produces differing widths.
        self.assertFalse(_can_batch_d2h([_cuda(2, 4), _cuda(2, 8)]))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_mixed_rank_does_not_batch(self):
        self.assertFalse(_can_batch_d2h([_cuda(4), _cuda(2, 4)]))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_mixed_dtype_does_not_batch(self):
        # torch.cat would promote and silently change the output dtype.
        self.assertFalse(
            _can_batch_d2h([_cuda(2, 4), _cuda(2, 4, dtype=torch.float16)])
        )


class TestBatchedAsyncD2HFallback(CustomTestCase):
    def _assert_matches_per_tensor(self, tensors):
        expected = [_async_d2h(t) for t in tensors]
        actual = _batched_async_d2h(tensors)
        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertEqual(a.shape, e.shape)
            self.assertEqual(a.dtype, e.dtype)
            self.assertTrue(torch.equal(a, e))

    def test_uniform_2d(self):
        self._assert_matches_per_tensor(
            [torch.randn(2, 4), torch.randn(3, 4), torch.randn(1, 4)]
        )

    def test_ragged(self):
        self._assert_matches_per_tensor([torch.randn(2, 4), torch.randn(2, 8)])

    def test_single_tensor(self):
        self._assert_matches_per_tensor([torch.randn(3, 4)])

    def test_zero_row_tensor_in_batch(self):
        self._assert_matches_per_tensor(
            [torch.randn(0, 4), torch.randn(2, 4), torch.randn(0, 4)]
        )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestBatchedAsyncD2HCuda(CustomTestCase):
    def _assert_matches_per_tensor(self, tensors):
        expected = [_async_d2h(t) for t in tensors]
        actual = _batched_async_d2h(tensors)
        torch.cuda.synchronize()
        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertEqual(a.shape, e.shape)
            self.assertEqual(a.dtype, e.dtype)
            self.assertTrue(torch.equal(a, e))

    def test_uniform_2d_matches_per_tensor(self):
        self._assert_matches_per_tensor([_cuda(2, 4), _cuda(3, 4), _cuda(1, 4)])

    def test_uniform_1d_matches_per_tensor(self):
        self._assert_matches_per_tensor([_cuda(4), _cuda(4)])

    def test_zero_row_tensor_in_batch(self):
        self._assert_matches_per_tensor([_cuda(0, 4), _cuda(2, 4), _cuda(0, 4)])

    def test_large_fanout_preserves_order(self):
        tensors = [
            torch.full((1, 4), float(i), device="cuda", dtype=torch.float32)
            for i in range(128)
        ]
        actual = _batched_async_d2h(tensors)
        torch.cuda.synchronize()
        for i, t in enumerate(actual):
            self.assertTrue(torch.equal(t, torch.full((1, 4), float(i))))


if __name__ == "__main__":
    unittest.main()
