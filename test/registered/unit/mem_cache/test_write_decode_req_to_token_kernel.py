"""
Unit tests for the _write_decode_req_to_token_kernel Triton kernel.

This module tests the Triton kernel that replaces the previous
req_to_token_pool.write() call in alloc_for_decode, ensuring correctness
of writing decode cache locations to the req_to_token pool.

Test Coverage:
- Basic correctness: single request write
- Multiple requests: batch write
- Edge cases: different stride values, boundary positions

Usage:
    python test_write_decode_req_to_token_kernel.py
    python -m pytest test_write_decode_req_to_token_kernel.py -v
"""

import unittest

import torch

from sglang.srt.mem_cache.common import _write_decode_req_to_token_kernel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="stage-b-test-1-gpu-small")


class TestWriteDecodeReqToTokenKernel(unittest.TestCase):
    """Test cases for _write_decode_req_to_token_kernel."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")

    def test_single_request_write(self):
        """Test writing a single decode entry to req_to_token pool."""
        max_batch = 4
        max_context_len = 16
        req_to_token = torch.zeros(
            (max_batch, max_context_len), dtype=torch.int32, device="cuda"
        )
        req_pool_indices = torch.tensor([2], dtype=torch.int64, device="cuda")
        seq_lens = torch.tensor([5], dtype=torch.int64, device="cuda")
        out_cache_loc = torch.tensor([100], dtype=torch.int64, device="cuda")

        _write_decode_req_to_token_kernel[(1,)](
            req_to_token,
            req_pool_indices,
            seq_lens,
            out_cache_loc,
            max_context_len,
        )

        # Verify the value was written at the correct position
        self.assertEqual(req_to_token[2, 5].item(), 100)
        # Verify no other positions were modified
        req_to_token[2, 5] = 0
        self.assertTrue(torch.all(req_to_token == 0).item())

    def test_batch_write(self):
        """Test writing multiple decode entries in a batch."""
        bs = 4
        max_batch = 8
        max_context_len = 32
        req_to_token = torch.zeros(
            (max_batch, max_context_len), dtype=torch.int32, device="cuda"
        )
        req_pool_indices = torch.tensor([0, 2, 5, 7], dtype=torch.int64, device="cuda")
        seq_lens = torch.tensor([3, 10, 1, 20], dtype=torch.int64, device="cuda")
        out_cache_loc = torch.tensor([50, 60, 70, 80], dtype=torch.int64, device="cuda")

        _write_decode_req_to_token_kernel[(bs,)](
            req_to_token,
            req_pool_indices,
            seq_lens,
            out_cache_loc,
            max_context_len,
        )

        # Verify each request wrote to the correct position
        self.assertEqual(req_to_token[0, 3].item(), 50)
        self.assertEqual(req_to_token[2, 10].item(), 60)
        self.assertEqual(req_to_token[5, 1].item(), 70)
        self.assertEqual(req_to_token[7, 20].item(), 80)

        # Verify no other positions were modified
        expected_positions = {(0, 3), (2, 10), (5, 1), (7, 20)}
        for i in range(max_batch):
            for j in range(max_context_len):
                if (i, j) in expected_positions:
                    self.assertNotEqual(req_to_token[i, j].item(), 0)
                else:
                    self.assertEqual(req_to_token[i, j].item(), 0)

    def test_overwrite_existing_value(self):
        """Test that the kernel correctly overwrites existing values."""
        max_batch = 2
        max_context_len = 8
        req_to_token = torch.full(
            (max_batch, max_context_len), 999, dtype=torch.int32, device="cuda"
        )
        req_pool_indices = torch.tensor([1], dtype=torch.int64, device="cuda")
        seq_lens = torch.tensor([4], dtype=torch.int64, device="cuda")
        out_cache_loc = torch.tensor([42], dtype=torch.int64, device="cuda")

        _write_decode_req_to_token_kernel[(1,)](
            req_to_token,
            req_pool_indices,
            seq_lens,
            out_cache_loc,
            max_context_len,
        )

        # Verify the value was overwritten
        self.assertEqual(req_to_token[1, 4].item(), 42)
        # Other positions should still have the old value
        self.assertEqual(req_to_token[1, 3].item(), 999)
        self.assertEqual(req_to_token[0, 4].item(), 999)

    def test_large_stride(self):
        """Test with a large stride (max_context_len) value."""
        max_batch = 2
        max_context_len = 1024
        req_to_token = torch.zeros(
            (max_batch, max_context_len), dtype=torch.int32, device="cuda"
        )
        req_pool_indices = torch.tensor([1], dtype=torch.int64, device="cuda")
        seq_lens = torch.tensor([512], dtype=torch.int64, device="cuda")
        out_cache_loc = torch.tensor([12345], dtype=torch.int64, device="cuda")

        _write_decode_req_to_token_kernel[(1,)](
            req_to_token,
            req_pool_indices,
            seq_lens,
            out_cache_loc,
            max_context_len,
        )

        self.assertEqual(req_to_token[1, 512].item(), 12345)

    def test_write_at_position_zero(self):
        """Test writing at seq_len=0 (first position)."""
        max_batch = 2
        max_context_len = 8
        req_to_token = torch.zeros(
            (max_batch, max_context_len), dtype=torch.int32, device="cuda"
        )
        req_pool_indices = torch.tensor([0], dtype=torch.int64, device="cuda")
        seq_lens = torch.tensor([0], dtype=torch.int64, device="cuda")
        out_cache_loc = torch.tensor([7], dtype=torch.int64, device="cuda")

        _write_decode_req_to_token_kernel[(1,)](
            req_to_token,
            req_pool_indices,
            seq_lens,
            out_cache_loc,
            max_context_len,
        )

        self.assertEqual(req_to_token[0, 0].item(), 7)

    def test_matches_python_reference(self):
        """Test that the Triton kernel produces the same result as a Python reference."""
        bs = 8
        max_batch = 16
        max_context_len = 64

        req_pool_indices = torch.randint(
            0, max_batch, (bs,), dtype=torch.int64, device="cuda"
        )
        seq_lens = torch.randint(
            0, max_context_len, (bs,), dtype=torch.int64, device="cuda"
        )
        out_cache_loc = torch.randint(0, 10000, (bs,), dtype=torch.int64, device="cuda")

        # Triton kernel result
        req_to_token_triton = torch.zeros(
            (max_batch, max_context_len), dtype=torch.int32, device="cuda"
        )
        _write_decode_req_to_token_kernel[(bs,)](
            req_to_token_triton,
            req_pool_indices,
            seq_lens,
            out_cache_loc,
            max_context_len,
        )

        # Python reference result
        req_to_token_ref = torch.zeros(
            (max_batch, max_context_len), dtype=torch.int32, device="cuda"
        )
        for i in range(bs):
            req_idx = req_pool_indices[i].item()
            col_idx = seq_lens[i].item()
            value = out_cache_loc[i].item()
            req_to_token_ref[req_idx, col_idx] = value

        self.assertTrue(
            torch.equal(req_to_token_triton, req_to_token_ref),
            "Triton kernel result does not match Python reference",
        )


if __name__ == "__main__":
    unittest.main()
