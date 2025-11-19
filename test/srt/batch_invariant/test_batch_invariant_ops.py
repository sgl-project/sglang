# Adapted from https://github.com/thinking-machines-lab/batch_invariant_ops/blob/main/test_batch_invariance.py
import math
import unittest

import torch

from sglang.srt.batch_invariant_ops import batch_invariant_ops
from sglang.srt.batch_invariant_ops.batch_invariant_ops import set_batch_invariant_mode
from sglang.test.test_utils import CustomTestCase

device_type = getattr(torch.accelerator.current_accelerator(), "type", "cpu")
torch.set_default_device(device_type)

# Just to get the logging out of the way
with set_batch_invariant_mode(True):
    pass


class TestBatchInvariantOps(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        batch_invariant_ops._ENABLE_MM_COMPARISON_TEST = True

    @classmethod
    def tearDownClass(cls):
        batch_invariant_ops._ENABLE_MM_COMPARISON_TEST = False

    def _test_batch_invariance(self, M, K, N, dtype):
        """
        Test that matrix operations produce identical results for:
        - Method 1: Matrix-vector multiplication (batch size 1)
        - Method 2: Matrix-matrix multiplication, then slice (full batch)
        """
        a = torch.linspace(-100, 100, M * K, dtype=dtype).reshape(M, K)

        # Create non-contiguous tensor
        b = torch.linspace(-100, 100, K * N, dtype=dtype).reshape(N, K)
        b = b.transpose(0, 1)

        # Method 1: Matrix-vector multiplication (batch size 1)
        out1 = torch.mm(a[:1], b)

        # Method 2: Matrix-matrix multiplication, then slice (full batch)
        out2_pre = torch.mm(a, b)
        out2 = out2_pre[:1]

        # Check if results are identical
        diff = (out1 - out2).abs().max()
        return diff.item()

    def _run_multiple_iterations(self, iters, M, K, N, dtype):
        """Run multiple iterations and collect diff statistics"""
        difflist = []
        for _ in range(iters):
            diff = self._test_batch_invariance(M, K, N, dtype)
            difflist.append(diff)
        return difflist

    def _assert_batch_invariant_results(self, difflist, dtype, test_name):
        """
        Assert that in batch-invariant mode:
        1. All diffs must not be NaN
        2. All diffs must be exactly 0
        3. Max, min, and diff of diffs must all be 0
        """
        max_diff = max(difflist)
        min_diff = min(difflist)
        diff_range = max_diff - min_diff

        # Check for NaN values
        self.assertFalse(
            math.isnan(max_diff), f"{test_name}: max_diff is NaN for {dtype}"
        )
        self.assertFalse(
            math.isnan(min_diff), f"{test_name}: min_diff is NaN for {dtype}"
        )
        self.assertFalse(
            math.isnan(diff_range), f"{test_name}: diff_range is NaN for {dtype}"
        )

        # Check that all diffs are exactly 0
        self.assertEqual(
            max_diff,
            0.0,
            f"{test_name}: max_diff must be 0 in batch-invariant mode, got {max_diff} for {dtype}",
        )
        self.assertEqual(
            min_diff,
            0.0,
            f"{test_name}: min_diff must be 0 in batch-invariant mode, got {min_diff} for {dtype}",
        )
        self.assertEqual(
            diff_range,
            0.0,
            f"{test_name}: diff_range must be 0 in batch-invariant mode, got {diff_range} for {dtype}",
        )

    def test_small_matrices(self):
        """Test batch invariance with small matrix sizes"""
        test_cases = [
            ("Small-1", 8, 64, 128),
            ("Small-2", 16, 128, 256),
            ("Small-3", 4, 32, 64),
        ]

        for name, M, K, N in test_cases:
            with self.subTest(name=name, M=M, K=K, N=N):
                for dtype in [torch.float32, torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        # Run with batch-invariant mode
                        with set_batch_invariant_mode(True):
                            difflist = self._run_multiple_iterations(
                                iters=5, M=M, K=K, N=N, dtype=dtype
                            )
                            self._assert_batch_invariant_results(difflist, dtype, name)

    def test_medium_matrices(self):
        """Test batch invariance with medium matrix sizes"""
        test_cases = [
            ("Medium-1", 32, 128, 1024),
            ("Medium-2", 64, 512, 2048),
            ("Medium-3", 24, 192, 768),
        ]

        for name, M, K, N in test_cases:
            with self.subTest(name=name, M=M, K=K, N=N):
                for dtype in [torch.float32, torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        # Run with batch-invariant mode
                        with set_batch_invariant_mode(True):
                            difflist = self._run_multiple_iterations(
                                iters=5, M=M, K=K, N=N, dtype=dtype
                            )
                            self._assert_batch_invariant_results(difflist, dtype, name)

    def test_large_matrices(self):
        """Test batch invariance with large matrix sizes"""
        test_cases = [
            ("Large-1", 128, 1024, 4096),
            ("Large-2", 256, 2048, 8192),
            ("Large-3", 96, 768, 3072),
        ]

        for name, M, K, N in test_cases:
            with self.subTest(name=name, M=M, K=K, N=N):
                for dtype in [torch.float32, torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        # Run with batch-invariant mode
                        with set_batch_invariant_mode(True):
                            difflist = self._run_multiple_iterations(
                                iters=5, M=M, K=K, N=N, dtype=dtype
                            )
                            self._assert_batch_invariant_results(difflist, dtype, name)

    def test_without_batch_invariant_mode(self):
        """
        Test that without batch-invariant mode, results may differ.
        This test demonstrates the difference batch-invariant mode makes.
        """
        M, K, N = 32, 128, 1024
        dtype = torch.float32

        # Run without batch-invariant mode
        with set_batch_invariant_mode(False):
            difflist = self._run_multiple_iterations(
                iters=5, M=M, K=K, N=N, dtype=dtype
            )
            print(f"Without batch-invariant mode, we get diffs: {difflist}")

    def _test_bmm_batch_invariance(self, B, M, K, N, dtype):
        """
        Test that BMM operations produce identical results for:
        - Method 1: BMM with subset of batches
        - Method 2: BMM with all batches, then slice
        """
        a = torch.linspace(-100, 100, B * M * K, dtype=dtype).reshape(B, M, K)
        b = torch.linspace(-100, 100, B * K * N, dtype=dtype).reshape(B, K, N)

        # Method 1: BMM with subset (first 2 batches)
        subset_size = min(2, B)
        out1 = torch.bmm(a[:subset_size], b[:subset_size])

        # Method 2: BMM with all batches, then slice
        out2_pre = torch.bmm(a, b)
        out2 = out2_pre[:subset_size]

        # Check if results are identical
        diff = (out1 - out2).abs().max()
        return diff.item()

    def _run_bmm_multiple_iterations(self, iters, B, M, K, N, dtype):
        """Run multiple BMM iterations and collect diff statistics"""
        difflist = []
        for _ in range(iters):
            diff = self._test_bmm_batch_invariance(B, M, K, N, dtype)
            difflist.append(diff)
        return difflist

    def test_bmm_small_matrices(self):
        """Test BMM batch invariance with small matrix sizes"""
        test_cases = [
            ("BMM-Small-1", 4, 8, 64, 128),
            ("BMM-Small-2", 8, 16, 128, 256),
            ("BMM-Small-3", 6, 4, 32, 64),
        ]

        for name, B, M, K, N in test_cases:
            with self.subTest(name=name, B=B, M=M, K=K, N=N):
                for dtype in [torch.float32, torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        # Run with batch-invariant mode
                        with set_batch_invariant_mode(True):
                            difflist = self._run_bmm_multiple_iterations(
                                iters=5, B=B, M=M, K=K, N=N, dtype=dtype
                            )
                            self._assert_batch_invariant_results(difflist, dtype, name)

    def test_bmm_medium_matrices(self):
        """Test BMM batch invariance with medium matrix sizes"""
        test_cases = [
            ("BMM-Medium-1", 8, 32, 128, 1024),
            ("BMM-Medium-2", 16, 64, 512, 2048),
            ("BMM-Medium-3", 12, 24, 192, 768),
        ]

        for name, B, M, K, N in test_cases:
            with self.subTest(name=name, B=B, M=M, K=K, N=N):
                for dtype in [torch.float32, torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        # Run with batch-invariant mode
                        with set_batch_invariant_mode(True):
                            difflist = self._run_bmm_multiple_iterations(
                                iters=5, B=B, M=M, K=K, N=N, dtype=dtype
                            )
                            self._assert_batch_invariant_results(difflist, dtype, name)

    def test_bmm_large_matrices(self):
        """Test BMM batch invariance with large matrix sizes"""
        test_cases = [
            ("BMM-Large-1", 16, 128, 1024, 4096),
            ("BMM-Large-2", 32, 256, 2048, 8192),
            ("BMM-Large-3", 24, 96, 768, 3072),
        ]

        for name, B, M, K, N in test_cases:
            with self.subTest(name=name, B=B, M=M, K=K, N=N):
                for dtype in [torch.float32, torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        # Run with batch-invariant mode
                        with set_batch_invariant_mode(True):
                            difflist = self._run_bmm_multiple_iterations(
                                iters=5, B=B, M=M, K=K, N=N, dtype=dtype
                            )
                            self._assert_batch_invariant_results(difflist, dtype, name)


if __name__ == "__main__":
    unittest.main()
