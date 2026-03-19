"""
Unit tests for the fused Triton kernel in normal_decode_set_metadata.

This test suite verifies:
1. Correctness against reference PyTorch implementation
2. Different page sizes (1, 16, 64)
3. With and without Sliding Window Attention (SWA)
4. Various batch sizes and sequence lengths
5. Edge cases
"""

import unittest

import torch

from sglang.srt.layers.attention.flashattention_backend import (
    normal_decode_set_metadata,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# Register this test for CUDA CI in stage-b (fast attention/kernel tests)
register_cuda_ci(est_time=25, suite="stage-b-test-large-1-gpu")


def reference_normal_decode_set_metadata(
    cache_seqlens_int32: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    strided_indices: torch.Tensor,
    max_seq_pages: int,
    seq_lens: torch.Tensor,
    seq_len_delta: int,
    page_size: int,
    swa_page_table: torch.Tensor = None,
    token_to_kv_pool=None,
):
    """
    Reference implementation using original PyTorch operations.
    This is the pre-Triton version for correctness comparison.
    """
    cache_seqlens_int32.copy_(seq_lens + seq_len_delta)
    cu_seqlens_k[1:].copy_(torch.cumsum(cache_seqlens_int32, dim=0, dtype=torch.int32))
    page_indices = req_to_token[
        req_pool_indices[:, None],
        strided_indices[:max_seq_pages][None, :],
    ]
    page_table[:, :max_seq_pages].copy_(page_indices // page_size)

    if swa_page_table is not None and token_to_kv_pool is not None:
        swa_page_indices = token_to_kv_pool.translate_loc_from_full_to_swa(page_indices)
        swa_page_table[:, :max_seq_pages].copy_(swa_page_indices // page_size)


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestNormalDecodeSetMetadata(CustomTestCase):
    """Test fused Triton kernel in normal_decode_set_metadata."""

    def setUp(self):
        self.device = "cuda"
        self.dtype = torch.int32

    def _create_test_data(
        self,
        batch_size: int,
        max_seq_len: int,
        page_size: int,
        has_swa: bool = False,
        seq_len_delta: int = 0,
    ):
        """Create test data for normal_decode_set_metadata."""
        # Random sequence lengths for each batch
        seq_lens = torch.randint(
            max_seq_len // 2,
            max_seq_len + 1,
            (batch_size,),
            dtype=torch.int64,
            device=self.device,
        )

        # Calculate max_seq_pages
        max_len = seq_lens.max().item()
        max_seq_pages = (max_len + seq_len_delta + page_size - 1) // page_size

        # Create req_pool_indices (maps batch index to pool index)
        req_pool_indices = torch.arange(
            batch_size, dtype=torch.int32, device=self.device
        )

        # Create strided_indices for page table indexing
        if page_size == 1:
            strided_indices = torch.arange(
                max_seq_len * 2, dtype=torch.int32, device=self.device
            )
        else:
            strided_indices = torch.arange(
                0, max_seq_len * 2, page_size, dtype=torch.int32, device=self.device
            )

        # Create req_to_token pool (simulates token locations in KV cache)
        pool_size = batch_size
        max_tokens = max_seq_len * 2
        req_to_token = torch.randint(
            0, 10000, (pool_size, max_tokens), dtype=torch.int32, device=self.device
        )

        # Output tensors (to be filled by the function)
        cache_seqlens_int32 = torch.zeros(
            batch_size, dtype=torch.int32, device=self.device
        )
        cu_seqlens_k = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=self.device
        )
        page_table = torch.zeros(
            (batch_size, max_seq_pages + 10), dtype=torch.int32, device=self.device
        )

        # SWA setup if needed
        swa_page_table = None
        token_to_kv_pool = None
        if has_swa:
            swa_page_table = torch.zeros(
                (batch_size, max_seq_pages + 10), dtype=torch.int32, device=self.device
            )
            # Create a simple SWA KV pool for testing
            token_to_kv_pool = self._create_swa_kv_pool(10000, page_size)

        return {
            "cache_seqlens_int32": cache_seqlens_int32,
            "cu_seqlens_k": cu_seqlens_k,
            "page_table": page_table,
            "req_to_token": req_to_token,
            "req_pool_indices": req_pool_indices,
            "strided_indices": strided_indices,
            "max_seq_pages": max_seq_pages,
            "seq_lens": seq_lens,
            "seq_len_delta": seq_len_delta,
            "page_size": page_size,
            "swa_page_table": swa_page_table,
            "token_to_kv_pool": token_to_kv_pool,
        }

    def _create_swa_kv_pool(self, size: int, page_size: int):
        """Create a mock SWA KV pool for testing that inherits from SWAKVPool."""

        # Create a minimal mock that inherits from SWAKVPool to pass isinstance check
        class MinimalSWAKVPool(SWAKVPool):
            def __init__(self, size, device):
                # Don't call super().__init__() to avoid complex initialization
                # Just set the minimal attributes needed for the test
                self.full_to_swa_index_mapping = torch.arange(
                    size, dtype=torch.int32, device=device
                )
                # Add some randomness to simulate real SWA mapping
                self.full_to_swa_index_mapping = (
                    self.full_to_swa_index_mapping
                    + torch.randint(0, 100, (size,), device=device)
                ) % size
                self.device = device

            def translate_loc_from_full_to_swa(self, page_indices):
                """Mock translation method."""
                return self.full_to_swa_index_mapping[page_indices]

        return MinimalSWAKVPool(size, self.device)

    def _run_test(
        self,
        batch_size: int,
        max_seq_len: int,
        page_size: int,
        has_swa: bool = False,
        seq_len_delta: int = 0,
    ):
        """Run a single test configuration."""
        # Create test data
        test_data = self._create_test_data(
            batch_size, max_seq_len, page_size, has_swa, seq_len_delta
        )

        # Clone data for reference implementation
        ref_data = {
            "cache_seqlens_int32": test_data["cache_seqlens_int32"].clone(),
            "cu_seqlens_k": test_data["cu_seqlens_k"].clone(),
            "page_table": test_data["page_table"].clone(),
            "swa_page_table": test_data["swa_page_table"].clone() if has_swa else None,
        }

        # Run reference implementation
        reference_normal_decode_set_metadata(
            ref_data["cache_seqlens_int32"],
            ref_data["cu_seqlens_k"],
            ref_data["page_table"],
            test_data["req_to_token"],
            test_data["req_pool_indices"],
            test_data["strided_indices"],
            test_data["max_seq_pages"],
            test_data["seq_lens"],
            test_data["seq_len_delta"],
            test_data["page_size"],
            ref_data["swa_page_table"],
            test_data["token_to_kv_pool"],
        )

        # Run fused Triton implementation
        normal_decode_set_metadata(
            test_data["cache_seqlens_int32"],
            test_data["cu_seqlens_k"],
            test_data["page_table"],
            test_data["req_to_token"],
            test_data["req_pool_indices"],
            test_data["strided_indices"],
            test_data["max_seq_pages"],
            test_data["seq_lens"],
            test_data["seq_len_delta"],
            test_data["page_size"],
            test_data["swa_page_table"],
            test_data["token_to_kv_pool"],
        )

        # Compare results
        self.assertTrue(
            torch.equal(
                test_data["cache_seqlens_int32"], ref_data["cache_seqlens_int32"]
            ),
            f"cache_seqlens_int32 mismatch. Expected:\n{ref_data['cache_seqlens_int32']}\nGot:\n{test_data['cache_seqlens_int32']}",
        )

        self.assertTrue(
            torch.equal(test_data["cu_seqlens_k"], ref_data["cu_seqlens_k"]),
            f"cu_seqlens_k mismatch. Expected:\n{ref_data['cu_seqlens_k']}\nGot:\n{test_data['cu_seqlens_k']}",
        )

        self.assertTrue(
            torch.equal(test_data["page_table"], ref_data["page_table"]),
            f"page_table mismatch at bs={batch_size}, page_size={page_size}",
        )

        if has_swa:
            self.assertTrue(
                torch.equal(test_data["swa_page_table"], ref_data["swa_page_table"]),
                f"swa_page_table mismatch at bs={batch_size}, page_size={page_size}",
            )

    # Test cases for page_size=1 (uses specialized kernel _fused_metadata_kernel_ps1_no_swa)
    def test_page_size_1_small_batch(self):
        """Test with page_size=1, small batch."""
        self._run_test(batch_size=2, max_seq_len=128, page_size=1, has_swa=False)

    def test_page_size_1_medium_batch(self):
        """Test with page_size=1, medium batch."""
        self._run_test(batch_size=16, max_seq_len=256, page_size=1, has_swa=False)

    def test_page_size_1_large_batch(self):
        """Test with page_size=1, large batch."""
        self._run_test(batch_size=64, max_seq_len=512, page_size=1, has_swa=False)

    def test_page_size_1_with_seq_len_delta(self):
        """Test with page_size=1 and seq_len_delta > 0."""
        self._run_test(
            batch_size=8, max_seq_len=200, page_size=1, has_swa=False, seq_len_delta=5
        )

    # Test cases for page_size > 1 (uses general kernel _fused_metadata_kernel_general)
    def test_page_size_16_small_batch(self):
        """Test with page_size=16, small batch."""
        self._run_test(batch_size=4, max_seq_len=256, page_size=16, has_swa=False)

    def test_page_size_16_medium_batch(self):
        """Test with page_size=16, medium batch."""
        self._run_test(batch_size=16, max_seq_len=512, page_size=16, has_swa=False)

    def test_page_size_64_small_batch(self):
        """Test with page_size=64, small batch."""
        self._run_test(batch_size=4, max_seq_len=512, page_size=64, has_swa=False)

    def test_page_size_64_medium_batch(self):
        """Test with page_size=64, medium batch."""
        self._run_test(batch_size=32, max_seq_len=1024, page_size=64, has_swa=False)

    def test_page_size_64_with_seq_len_delta(self):
        """Test with page_size=64 and seq_len_delta > 0."""
        self._run_test(
            batch_size=8, max_seq_len=512, page_size=64, has_swa=False, seq_len_delta=3
        )

    # Test cases with Sliding Window Attention (SWA)
    def test_page_size_16_with_swa(self):
        """Test with page_size=16 and SWA enabled."""
        self._run_test(batch_size=8, max_seq_len=256, page_size=16, has_swa=True)

    def test_page_size_64_with_swa(self):
        """Test with page_size=64 and SWA enabled."""
        self._run_test(batch_size=16, max_seq_len=512, page_size=64, has_swa=True)

    def test_page_size_64_with_swa_and_delta(self):
        """Test with page_size=64, SWA, and seq_len_delta."""
        self._run_test(
            batch_size=8, max_seq_len=400, page_size=64, has_swa=True, seq_len_delta=2
        )

    # Edge cases
    def test_batch_size_1(self):
        """Test with single batch."""
        self._run_test(batch_size=1, max_seq_len=128, page_size=1, has_swa=False)
        self._run_test(batch_size=1, max_seq_len=256, page_size=64, has_swa=False)

    def test_max_seq_pages_small(self):
        """Test edge case where max_seq_pages could be very small."""
        # This tests when sequences are very short
        test_data = self._create_test_data(
            batch_size=2, max_seq_len=10, page_size=64, has_swa=False
        )

        # Run fused implementation (should handle gracefully)
        normal_decode_set_metadata(
            test_data["cache_seqlens_int32"],
            test_data["cu_seqlens_k"],
            test_data["page_table"],
            test_data["req_to_token"],
            test_data["req_pool_indices"],
            test_data["strided_indices"],
            test_data["max_seq_pages"],
            test_data["seq_lens"],
            test_data["seq_len_delta"],
            test_data["page_size"],
            test_data["swa_page_table"],
            test_data["token_to_kv_pool"],
        )

        # Verify no crashes and basic properties
        self.assertEqual(
            test_data["cache_seqlens_int32"].sum().item(),
            test_data["seq_lens"].sum().item(),
        )

    def test_power_of_two_page_sizes(self):
        """Test various power-of-2 page sizes."""
        page_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        for page_size in page_sizes:
            with self.subTest(page_size=page_size):
                self._run_test(
                    batch_size=4, max_seq_len=256, page_size=page_size, has_swa=False
                )

    def test_varied_sequence_lengths(self):
        """Test with highly varied sequence lengths in the same batch."""
        batch_size = 8
        max_seq_len = 512
        page_size = 64

        test_data = self._create_test_data(
            batch_size, max_seq_len, page_size, has_swa=False
        )

        # Manually set varied sequence lengths
        test_data["seq_lens"] = torch.tensor(
            [10, 50, 100, 200, 300, 450, 500, 512],
            dtype=torch.int64,
            device=self.device,
        )
        test_data["max_seq_pages"] = (
            test_data["seq_lens"].max().item() + page_size - 1
        ) // page_size

        # Run both implementations
        ref_data = {
            "cache_seqlens_int32": test_data["cache_seqlens_int32"].clone(),
            "cu_seqlens_k": test_data["cu_seqlens_k"].clone(),
            "page_table": test_data["page_table"].clone(),
        }

        reference_normal_decode_set_metadata(
            ref_data["cache_seqlens_int32"],
            ref_data["cu_seqlens_k"],
            ref_data["page_table"],
            test_data["req_to_token"],
            test_data["req_pool_indices"],
            test_data["strided_indices"],
            test_data["max_seq_pages"],
            test_data["seq_lens"],
            0,
            page_size,
            None,
            None,
        )

        normal_decode_set_metadata(
            test_data["cache_seqlens_int32"],
            test_data["cu_seqlens_k"],
            test_data["page_table"],
            test_data["req_to_token"],
            test_data["req_pool_indices"],
            test_data["strided_indices"],
            test_data["max_seq_pages"],
            test_data["seq_lens"],
            0,
            page_size,
            None,
            None,
        )

        self.assertTrue(
            torch.equal(
                test_data["cache_seqlens_int32"], ref_data["cache_seqlens_int32"]
            )
        )
        self.assertTrue(
            torch.equal(test_data["cu_seqlens_k"], ref_data["cu_seqlens_k"])
        )
        self.assertTrue(torch.equal(test_data["page_table"], ref_data["page_table"]))


if __name__ == "__main__":
    unittest.main()
