"""
Test DCP interleaved storage for KV cache allocators.

This test verifies that tokens are correctly distributed across DCP ranks
based on token_idx % dcp_world_size == dcp_rank.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.allocator import (
    TokenToKVPoolAllocator,
    PagedTokenToKVPoolAllocator,
)


class MockKVCache:
    """Mock KVCache for testing."""

    def __init__(self, size, dtype, device):
        self.size = size
        self.dtype = dtype
        self.device = device

    def get_cpu_copy(self, indices):
        return None

    def load_cpu_copy(self, kv_cache_cpu, indices):
        pass


class TestDCPInterleavedStorage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.dtype = torch.bfloat16

    def setUp(self):
        """Set up test fixtures."""
        self.size = 100
        self.page_size = 16
        self.kvcache = MockKVCache(self.size, self.dtype, self.device)

    @patch("sglang.srt.mem_cache.allocator.get_dcp_world_size")
    @patch("sglang.srt.mem_cache.allocator.get_dcp_rank")
    def test_token_allocator_dcp_interleaved_page_size_1(self, mock_rank, mock_world_size):
        """Test TokenToKVPoolAllocator with DCP interleaved storage (page_size=1)."""
        # Setup: 2 DCP ranks, current rank is 0
        mock_world_size.return_value = 2
        mock_rank.return_value = 0

        allocator = TokenToKVPoolAllocator(
            size=self.size,
            dtype=self.dtype,
            device=self.device,
            kvcache=self.kvcache,
            need_sort=False,
        )

        # Test: allocate 5 tokens with positions [0, 1, 2, 3, 4]
        # Rank 0 should store: 0, 2, 4 (token_idx % 2 == 0)
        # Rank 1 should store: 1, 3 (token_idx % 2 == 1)
        token_positions = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=self.device)
        need_size = 5

        initial_available = allocator.available_size()
        indices = allocator.alloc(need_size, token_positions=token_positions)

        # Verify: indices for rank 1 tokens (positions 1, 3) should be 0 (placeholder)
        self.assertIsNotNone(indices)
        self.assertEqual(len(indices), need_size)
        self.assertEqual(indices[1].item(), 0)  # Position 1 -> rank 1 -> placeholder
        self.assertEqual(indices[3].item(), 0)  # Position 3 -> rank 1 -> placeholder
        self.assertNotEqual(indices[0].item(), 0)  # Position 0 -> rank 0 -> allocated
        self.assertNotEqual(indices[2].item(), 0)  # Position 2 -> rank 0 -> allocated
        self.assertNotEqual(indices[4].item(), 0)  # Position 4 -> rank 0 -> allocated

        # Verify: only 3 tokens were actually allocated (for rank 0)
        # The unused indices should be returned to free pool
        final_available = allocator.available_size()
        # Should have allocated 3 tokens (for rank 0), so available_size decreased by 3
        self.assertEqual(initial_available - final_available, 3)

    @patch("sglang.srt.mem_cache.allocator.get_dcp_world_size")
    @patch("sglang.srt.mem_cache.allocator.get_dcp_rank")
    def test_token_allocator_dcp_interleaved_rank_1(self, mock_rank, mock_world_size):
        """Test TokenToKVPoolAllocator with DCP interleaved storage, rank 1."""
        # Setup: 2 DCP ranks, current rank is 1
        mock_world_size.return_value = 2
        mock_rank.return_value = 1

        allocator = TokenToKVPoolAllocator(
            size=self.size,
            dtype=self.dtype,
            device=self.device,
            kvcache=self.kvcache,
            need_sort=False,
        )

        # Test: allocate 5 tokens with positions [0, 1, 2, 3, 4]
        token_positions = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=self.device)
        need_size = 5

        initial_available = allocator.available_size()
        indices = allocator.alloc(need_size, token_positions=token_positions)

        # Verify: indices for rank 0 tokens (positions 0, 2, 4) should be 0 (placeholder)
        self.assertIsNotNone(indices)
        self.assertEqual(len(indices), need_size)
        self.assertEqual(indices[0].item(), 0)  # Position 0 -> rank 0 -> placeholder
        self.assertEqual(indices[2].item(), 0)  # Position 2 -> rank 0 -> placeholder
        self.assertEqual(indices[4].item(), 0)  # Position 4 -> rank 0 -> placeholder
        self.assertNotEqual(indices[1].item(), 0)  # Position 1 -> rank 1 -> allocated
        self.assertNotEqual(indices[3].item(), 0)  # Position 3 -> rank 1 -> allocated

        # Verify: only 2 tokens were actually allocated (for rank 1)
        final_available = allocator.available_size()
        self.assertEqual(initial_available - final_available, 2)

    @patch("sglang.srt.mem_cache.allocator.get_dcp_world_size")
    @patch("sglang.srt.mem_cache.allocator.get_dcp_rank")
    def test_token_allocator_no_dcp(self, mock_rank, mock_world_size):
        """Test TokenToKVPoolAllocator without DCP (dcp_world_size=1)."""
        # Setup: DCP disabled (world_size=1)
        mock_world_size.return_value = 1
        mock_rank.return_value = 0

        allocator = TokenToKVPoolAllocator(
            size=self.size,
            dtype=self.dtype,
            device=self.device,
            kvcache=self.kvcache,
            need_sort=False,
        )

        token_positions = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=self.device)
        need_size = 5

        initial_available = allocator.available_size()
        indices = allocator.alloc(need_size, token_positions=token_positions)

        # Verify: all tokens should be allocated (no filtering)
        self.assertIsNotNone(indices)
        self.assertEqual(len(indices), need_size)
        self.assertTrue(torch.all(indices > 0))  # All indices should be allocated

        # Verify: all 5 tokens were allocated
        final_available = allocator.available_size()
        self.assertEqual(initial_available - final_available, 5)

    @patch("sglang.srt.mem_cache.allocator.get_dcp_world_size")
    @patch("sglang.srt.mem_cache.allocator.get_dcp_rank")
    def test_paged_allocator_extend_dcp_interleaved(self, mock_rank, mock_world_size):
        """Test PagedTokenToKVPoolAllocator.alloc_extend with DCP interleaved storage."""
        # Setup: 2 DCP ranks, current rank is 0
        mock_world_size.return_value = 2
        mock_rank.return_value = 0

        allocator = PagedTokenToKVPoolAllocator(
            size=self.size,
            page_size=self.page_size,
            dtype=self.dtype,
            device=self.device,
            kvcache=self.kvcache,
            need_sort=False,
        )

        # Test: extend with 5 tokens at positions [10, 11, 12, 13, 14]
        # Rank 0 should store: 10, 12, 14 (token_idx % 2 == 0)
        # Rank 1 should store: 11, 13 (token_idx % 2 == 1)
        bs = 1
        prefix_lens = torch.tensor([10], dtype=torch.int64, device=self.device)
        prefix_lens_cpu = torch.tensor([10], dtype=torch.int64)
        seq_lens = torch.tensor([15], dtype=torch.int64, device=self.device)
        seq_lens_cpu = torch.tensor([15], dtype=torch.int64)
        last_loc = torch.tensor([9], dtype=torch.int64, device=self.device)
        extend_num_tokens = 5
        token_positions = torch.tensor([10, 11, 12, 13, 14], dtype=torch.int64, device=self.device)

        initial_available = allocator.available_size()
        indices = allocator.alloc_extend(
            prefix_lens=prefix_lens,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            last_loc=last_loc,
            extend_num_tokens=extend_num_tokens,
            token_positions=token_positions,
        )

        # Verify: indices for rank 1 tokens (positions 11, 13) should be 0 (placeholder)
        self.assertIsNotNone(indices)
        self.assertEqual(len(indices), extend_num_tokens)
        self.assertEqual(indices[1].item(), 0)  # Position 11 -> rank 1 -> placeholder
        self.assertEqual(indices[3].item(), 0)  # Position 13 -> rank 1 -> placeholder
        self.assertNotEqual(indices[0].item(), 0)  # Position 10 -> rank 0 -> allocated
        self.assertNotEqual(indices[2].item(), 0)  # Position 12 -> rank 0 -> allocated
        self.assertNotEqual(indices[4].item(), 0)  # Position 14 -> rank 0 -> allocated

    @patch("sglang.srt.mem_cache.allocator.get_dcp_world_size")
    @patch("sglang.srt.mem_cache.allocator.get_dcp_rank")
    def test_paged_allocator_decode_dcp_interleaved(self, mock_rank, mock_world_size):
        """Test PagedTokenToKVPoolAllocator.alloc_decode with DCP interleaved storage."""
        # Setup: 2 DCP ranks, current rank is 0
        mock_world_size.return_value = 2
        mock_rank.return_value = 0

        allocator = PagedTokenToKVPoolAllocator(
            size=self.size,
            page_size=self.page_size,
            dtype=self.dtype,
            device=self.device,
            kvcache=self.kvcache,
            need_sort=False,
        )

        # Test: decode with 3 requests at positions [5, 6, 7]
        # Rank 0 should store: 5, 7 (token_idx % 2 == 0)
        # Rank 1 should store: 6 (token_idx % 2 == 1)
        bs = 3
        seq_lens = torch.tensor([6, 7, 8], dtype=torch.int64, device=self.device)
        seq_lens_cpu = torch.tensor([6, 7, 8], dtype=torch.int64)
        last_loc = torch.tensor([4, 5, 6], dtype=torch.int64, device=self.device)
        token_positions = torch.tensor([5, 6, 7], dtype=torch.int64, device=self.device)

        indices = allocator.alloc_decode(
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            last_loc=last_loc,
            token_positions=token_positions,
        )

        # Verify: indices for rank 1 token (position 6) should be 0 (placeholder)
        self.assertIsNotNone(indices)
        self.assertEqual(len(indices), bs)
        self.assertEqual(indices[1].item(), 0)  # Position 6 -> rank 1 -> placeholder
        self.assertNotEqual(indices[0].item(), 0)  # Position 5 -> rank 0 -> allocated
        self.assertNotEqual(indices[2].item(), 0)  # Position 7 -> rank 0 -> allocated

    @patch("sglang.srt.mem_cache.allocator.get_dcp_world_size")
    @patch("sglang.srt.mem_cache.allocator.get_dcp_rank")
    def test_paged_allocator_extend_no_token_positions(self, mock_rank, mock_world_size):
        """Test PagedTokenToKVPoolAllocator.alloc_extend without token_positions (backward compatibility)."""
        mock_world_size.return_value = 2
        mock_rank.return_value = 0

        allocator = PagedTokenToKVPoolAllocator(
            size=self.size,
            page_size=self.page_size,
            dtype=self.dtype,
            device=self.device,
            kvcache=self.kvcache,
            need_sort=False,
        )

        bs = 1
        prefix_lens = torch.tensor([10], dtype=torch.int64, device=self.device)
        prefix_lens_cpu = torch.tensor([10], dtype=torch.int64)
        seq_lens = torch.tensor([15], dtype=torch.int64, device=self.device)
        seq_lens_cpu = torch.tensor([15], dtype=torch.int64)
        last_loc = torch.tensor([9], dtype=torch.int64, device=self.device)
        extend_num_tokens = 5

        # Should work without token_positions (backward compatibility)
        indices = allocator.alloc_extend(
            prefix_lens=prefix_lens,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            last_loc=last_loc,
            extend_num_tokens=extend_num_tokens,
            token_positions=None,
        )

        # Verify: all tokens should be allocated (no filtering when token_positions is None)
        self.assertIsNotNone(indices)
        self.assertEqual(len(indices), extend_num_tokens)
        self.assertTrue(torch.all(indices > 0))  # All indices should be allocated

    @patch("sglang.srt.mem_cache.allocator.get_dcp_world_size")
    @patch("sglang.srt.mem_cache.allocator.get_dcp_rank")
    def test_dcp_3_ranks(self, mock_rank, mock_world_size):
        """Test with 3 DCP ranks."""
        # Setup: 3 DCP ranks, current rank is 1
        mock_world_size.return_value = 3
        mock_rank.return_value = 1

        allocator = TokenToKVPoolAllocator(
            size=self.size,
            dtype=self.dtype,
            device=self.device,
            kvcache=self.kvcache,
            need_sort=False,
        )

        # Test: allocate 6 tokens with positions [0, 1, 2, 3, 4, 5]
        # Rank 0 should store: 0, 3 (token_idx % 3 == 0)
        # Rank 1 should store: 1, 4 (token_idx % 3 == 1)
        # Rank 2 should store: 2, 5 (token_idx % 3 == 2)
        token_positions = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64, device=self.device)
        need_size = 6

        indices = allocator.alloc(need_size, token_positions=token_positions)

        # Verify: rank 1 should only store positions 1 and 4
        self.assertIsNotNone(indices)
        self.assertEqual(len(indices), need_size)
        self.assertEqual(indices[0].item(), 0)  # Position 0 -> rank 0 -> placeholder
        self.assertNotEqual(indices[1].item(), 0)  # Position 1 -> rank 1 -> allocated
        self.assertEqual(indices[2].item(), 0)  # Position 2 -> rank 2 -> placeholder
        self.assertEqual(indices[3].item(), 0)  # Position 3 -> rank 0 -> placeholder
        self.assertNotEqual(indices[4].item(), 0)  # Position 4 -> rank 1 -> allocated
        self.assertEqual(indices[5].item(), 0)  # Position 5 -> rank 2 -> placeholder


if __name__ == "__main__":
    unittest.main()

