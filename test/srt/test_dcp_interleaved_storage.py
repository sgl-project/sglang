"""
Test DCP interleaved storage for KV cache allocators.

This test verifies that tokens are correctly distributed across DCP ranks
based on token_idx % dcp_world_size == dcp_rank.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
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
    def test_token_allocator_dcp_interleaved_page_size_1(
        self, mock_rank, mock_world_size
    ):
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
        token_positions = torch.tensor(
            [0, 1, 2, 3, 4], dtype=torch.int64, device=self.device
        )
        need_size = 5

        initial_available = allocator.available_size()
        indices = allocator.alloc(need_size, token_positions=token_positions)

        # Verify: indices for rank 1 tokens (positions 1, 3) should be 0 (placeholder)
        self.assertIsNotNone(indices)
        self.assertEqual(len(indices), need_size)
        self.assertEqual(indices[1].item(), -1)  # Position 1 -> rank 1 -> placeholder
        self.assertEqual(indices[3].item(), -1)  # Position 3 -> rank 1 -> placeholder
        self.assertNotEqual(indices[0].item(), -1)  # Position 0 -> rank 0 -> allocated
        self.assertNotEqual(indices[2].item(), -1)  # Position 2 -> rank 0 -> allocated
        self.assertNotEqual(indices[4].item(), -1)  # Position 4 -> rank 0 -> allocated

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
        token_positions = torch.tensor(
            [0, 1, 2, 3, 4], dtype=torch.int64, device=self.device
        )
        need_size = 5

        initial_available = allocator.available_size()
        indices = allocator.alloc(need_size, token_positions=token_positions)

        # Verify: indices for rank 0 tokens (positions 0, 2, 4) should be 0 (placeholder)
        self.assertIsNotNone(indices)
        self.assertEqual(len(indices), need_size)
        self.assertEqual(indices[0].item(), -1)  # Position 0 -> rank 0 -> placeholder
        self.assertEqual(indices[2].item(), -1)  # Position 2 -> rank 0 -> placeholder
        self.assertEqual(indices[4].item(), -1)  # Position 4 -> rank 0 -> placeholder
        self.assertNotEqual(indices[1].item(), -1)  # Position 1 -> rank 1 -> allocated
        self.assertNotEqual(indices[3].item(), -1)  # Position 3 -> rank 1 -> allocated

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

        token_positions = torch.tensor(
            [0, 1, 2, 3, 4], dtype=torch.int64, device=self.device
        )
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
        token_positions = torch.tensor(
            [10, 11, 12, 13, 14], dtype=torch.int64, device=self.device
        )

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
        self.assertEqual(indices[1].item(), -1)  # Position 11 -> rank 1 -> placeholder
        self.assertEqual(indices[3].item(), -1)  # Position 13 -> rank 1 -> placeholder
        self.assertNotEqual(indices[0].item(), -1)  # Position 10 -> rank 0 -> allocated
        self.assertNotEqual(indices[2].item(), -1)  # Position 12 -> rank 0 -> allocated
        self.assertNotEqual(indices[4].item(), -1)  # Position 14 -> rank 0 -> allocated

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
        # Rank 1 should store: 5, 7 (token_idx % 2 == 1)
        # Rank 0 should store: 6 (token_idx % 2 == 0)
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
        self.assertNotEqual(indices[1].item(), -1)  # Position 6 -> rank 1 -> allocated
        self.assertEqual(indices[0].item(), -1)  # Position 5 -> rank 0 -> placeholder
        self.assertEqual(indices[2].item(), -1)  # Position 7 -> rank 0 -> placeholder

    @patch("sglang.srt.mem_cache.allocator.get_dcp_world_size")
    @patch("sglang.srt.mem_cache.allocator.get_dcp_rank")
    def test_paged_allocator_extend_no_token_positions(
        self, mock_rank, mock_world_size
    ):
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
        token_positions = torch.tensor(
            [0, 1, 2, 3, 4, 5], dtype=torch.int64, device=self.device
        )
        need_size = 6

        indices = allocator.alloc(need_size, token_positions=token_positions)

        # Verify: rank 1 should only store positions 1 and 4
        self.assertIsNotNone(indices)
        self.assertEqual(len(indices), need_size)
        self.assertEqual(indices[0].item(), -1)  # Position 0 -> rank 0 -> placeholder
        self.assertNotEqual(indices[1].item(), -1)  # Position 1 -> rank 1 -> allocated
        self.assertEqual(indices[2].item(), -1)  # Position 2 -> rank 2 -> placeholder
        self.assertEqual(indices[3].item(), -1)  # Position 3 -> rank 0 -> placeholder
        self.assertNotEqual(indices[4].item(), -1)  # Position 4 -> rank 1 -> allocated
        self.assertEqual(indices[5].item(), -1)  # Position 5 -> rank 2 -> placeholder

    @patch("sglang.srt.mem_cache.allocator.get_dcp_world_size")
    @patch("sglang.srt.mem_cache.allocator.get_dcp_rank")
    def test_multiple_allocs_with_free(self, mock_rank, mock_world_size):
        """Test multiple allocs followed by free operations."""
        # Setup: 2 DCP ranks, current rank is 0
        mock_world_size.return_value = 2
        mock_rank.return_value = 0

        allocator = TokenToKVPoolAllocator(
            size=100,
            dtype=self.dtype,
            device=self.device,
            kvcache=self.kvcache,
            need_sort=False,
        )

        # First allocation
        token_positions_1 = torch.tensor(
            [0, 1, 2, 3, 4], dtype=torch.int64, device=self.device
        )
        indices_1 = allocator.alloc(5, token_positions=token_positions_1)
        available_after_alloc1 = allocator.available_size()

        # Free the allocated indices (only the non-placeholder ones)
        rank_0_indices_1 = indices_1[indices_1 > 0]
        allocator.free(rank_0_indices_1)
        available_after_free1 = allocator.available_size()

        # Verify: available size increased after free
        self.assertGreater(available_after_free1, available_after_alloc1)

        # Second allocation should be able to reuse freed indices
        token_positions_2 = torch.tensor(
            [5, 6, 7, 8, 9], dtype=torch.int64, device=self.device
        )
        indices_2 = allocator.alloc(5, token_positions=token_positions_2)
        self.assertIsNotNone(indices_2)

        # Verify: The freed indices might be reused
        rank_0_indices_2 = indices_2[indices_2 > 0]
        # At least some indices should be valid
        self.assertTrue(torch.all(rank_0_indices_2 > 0))

    @patch("sglang.srt.mem_cache.allocator.get_dcp_world_size")
    @patch("sglang.srt.mem_cache.allocator.get_dcp_rank")
    def test_non_paged_large_seq_4_ranks(self, mock_rank, mock_world_size):
        """Test TokenToKVPoolAllocator with DCP interleaved storage, 4 ranks, large sequence."""
        # Setup: 4 DCP ranks
        mock_world_size.return_value = 4

        # Test each rank
        for rank in range(4):
            mock_rank.return_value = rank

            allocator = TokenToKVPoolAllocator(
                size=1000,
                dtype=self.dtype,
                device=self.device,
                kvcache=self.kvcache,
                need_sort=False,
            )

            # Large sequence: 100 tokens with positions [0, 1, 2, ..., 99]
            seq_len = 100
            token_positions = torch.arange(
                seq_len, dtype=torch.int64, device=self.device
            )

            initial_available = allocator.available_size()
            indices = allocator.alloc(seq_len, token_positions=token_positions)

            self.assertIsNotNone(indices, f"Rank {rank} allocation failed")
            self.assertEqual(len(indices), seq_len)

            # Verify: Each rank should store tokens where token_idx % 4 == rank
            expected_count = 0
            for pos in range(seq_len):
                if pos % 4 == rank:
                    # This token belongs to current rank
                    self.assertNotEqual(
                        indices[pos].item(),
                        -1,
                        f"Rank {rank}, position {pos} should be allocated",
                    )
                    expected_count += 1
                else:
                    # This token belongs to another rank
                    self.assertEqual(
                        indices[pos].item(),
                        -1,
                        f"Rank {rank}, position {pos} should be placeholder",
                    )

            # Verify: available_size decreased by expected_count
            final_available = allocator.available_size()
            self.assertEqual(
                initial_available - final_available,
                expected_count,
                f"Rank {rank} should allocate {expected_count} tokens",
            )

            # Verify: All allocated indices are unique
            allocated_indices = indices[indices >= 0]
            self.assertEqual(len(allocated_indices), expected_count)
            self.assertEqual(
                len(torch.unique(allocated_indices)),
                expected_count,
                f"Rank {rank} allocated indices should be unique",
            )

    @patch("sglang.srt.mem_cache.allocator.get_dcp_world_size")
    @patch("sglang.srt.mem_cache.allocator.get_dcp_rank")
    def test_paged_large_seq_4_ranks_extend(self, mock_rank, mock_world_size):
        """Test PagedTokenToKVPoolAllocator.alloc_extend with DCP interleaved storage, 4 ranks, large sequence."""
        # Setup: 4 DCP ranks
        mock_world_size.return_value = 4
        page_size = 16

        # Test each rank
        for rank in range(4):
            mock_rank.return_value = rank

            allocator = PagedTokenToKVPoolAllocator(
                size=1000,
                page_size=page_size,
                dtype=self.dtype,
                device=self.device,
                kvcache=self.kvcache,
                need_sort=False,
            )

            # Large extend: prefix_len=20, seq_len=120, so extend_len=100
            prefix_len = 20
            allocator.alloc(prefix_len)
            seq_len = 120
            extend_len = seq_len - prefix_len

            prefix_lens = torch.tensor(
                [prefix_len], dtype=torch.int64, device=self.device
            )
            prefix_lens_cpu = torch.tensor([prefix_len], dtype=torch.int64)
            seq_lens = torch.tensor([seq_len], dtype=torch.int64, device=self.device)
            seq_lens_cpu = torch.tensor([seq_len], dtype=torch.int64)
            last_loc = torch.tensor(
                [prefix_len - 1], dtype=torch.int64, device=self.device
            )

            # Token positions for extend tokens: [20, 21, 22, ..., 119]
            token_positions = torch.arange(
                prefix_len, seq_len, dtype=torch.int64, device=self.device
            )

            initial_available = allocator.available_size()
            indices = allocator.alloc_extend(
                prefix_lens=prefix_lens,
                prefix_lens_cpu=prefix_lens_cpu,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                last_loc=last_loc,
                extend_num_tokens=extend_len,
                token_positions=token_positions,
            )

            self.assertIsNotNone(indices, f"Rank {rank} extend allocation failed")
            self.assertEqual(len(indices), extend_len)

            # Verify: Each rank should store tokens where token_idx % 4 == rank
            expected_count = 0
            for i, pos in enumerate(range(prefix_len, seq_len)):
                if pos % 4 == rank:
                    # This token belongs to current rank
                    self.assertNotEqual(
                        indices[i].item(),
                        -1,
                        f"Rank {rank}, position {pos} should be allocated",
                    )
                    expected_count += 1
                else:
                    # This token belongs to another rank
                    self.assertEqual(
                        indices[i].item(),
                        -1,
                        f"Rank {rank}, position {pos} should be placeholder",
                    )

            # Verify: All allocated indices are unique
            allocated_indices = indices[indices >= 0]
            self.assertEqual(len(allocated_indices), expected_count)
            self.assertEqual(
                len(torch.unique(allocated_indices)),
                expected_count,
                f"Rank {rank} allocated indices should be unique",
            )

    @patch("sglang.srt.mem_cache.allocator.get_dcp_world_size")
    @patch("sglang.srt.mem_cache.allocator.get_dcp_rank")
    def test_paged_large_seq_4_ranks_decode(self, mock_rank, mock_world_size):
        """Test PagedTokenToKVPoolAllocator.alloc_decode with DCP interleaved storage, 4 ranks, large batch."""
        # Setup: 4 DCP ranks
        mock_world_size.return_value = 4
        page_size = 16

        # Test each rank
        for rank in range(4):
            mock_rank.return_value = rank

            allocator = PagedTokenToKVPoolAllocator(
                size=1000,
                page_size=page_size,
                dtype=self.dtype,
                device=self.device,
                kvcache=self.kvcache,
                need_sort=False,
            )

            # Large decode batch: 50 requests with varying sequence lengths
            batch_size = 50
            seq_lens_list = [
                100 + i for i in range(batch_size)
            ]  # seq_lens: [100, 101, ..., 149]
            seq_lens = torch.tensor(
                seq_lens_list, dtype=torch.int64, device=self.device
            )
            seq_lens_cpu = torch.tensor(seq_lens_list, dtype=torch.int64)
            last_loc = torch.tensor(
                [s - 2 for s in seq_lens_list], dtype=torch.int64, device=self.device
            )

            # Token positions: seq_len - 1 for each request
            token_positions = (seq_lens - 1).to(torch.int64)

            initial_available = allocator.available_size()
            indices = allocator.alloc_decode(
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                last_loc=last_loc,
                token_positions=token_positions,
            )

            self.assertIsNotNone(indices, f"Rank {rank} decode allocation failed")
            self.assertEqual(len(indices), batch_size)

            # Verify: Each rank should store tokens where token_idx % 4 == rank
            expected_count = 0
            for i, seq_len_val in enumerate(seq_lens_list):
                token_pos = seq_len_val - 1  # Decode token position
                if token_pos % 4 == rank:
                    # This token belongs to current rank
                    self.assertNotEqual(
                        indices[i].item(),
                        -1,
                        f"Rank {rank}, request {i}, position {token_pos} should be allocated",
                    )
                    expected_count += 1
                else:
                    # This token belongs to another rank
                    self.assertEqual(
                        indices[i].item(),
                        -1,
                        f"Rank {rank}, request {i}, position {token_pos} should be placeholder",
                    )

            # Verify: All allocated indices are unique
            allocated_indices = indices[indices >= 0]
            self.assertEqual(len(allocated_indices), expected_count)
            if expected_count > 0:
                self.assertEqual(
                    len(torch.unique(allocated_indices)),
                    expected_count,
                    f"Rank {rank} allocated indices should be unique",
                )

    @patch("sglang.srt.mem_cache.allocator.get_dcp_world_size")
    @patch("sglang.srt.mem_cache.allocator.get_dcp_rank")
    def test_paged_mixed_batch_4_ranks(self, mock_rank, mock_world_size):
        """Test PagedTokenToKVPoolAllocator with mixed extend and decode, 4 ranks."""
        # Setup: 4 DCP ranks
        mock_world_size.return_value = 4
        page_size = 16

        # Test rank 0
        mock_rank.return_value = 0

        allocator = PagedTokenToKVPoolAllocator(
            size=2000,
            page_size=page_size,
            dtype=self.dtype,
            device=self.device,
            kvcache=self.kvcache,
            need_sort=False,
        )

        # First: extend with large sequence
        prefix_len = 50
        allocator.alloc(prefix_len)
        seq_len = 200
        extend_len = seq_len - prefix_len

        prefix_lens = torch.tensor([prefix_len], dtype=torch.int64, device=self.device)
        prefix_lens_cpu = torch.tensor([prefix_len], dtype=torch.int64)
        seq_lens_extend = torch.tensor([seq_len], dtype=torch.int64, device=self.device)
        seq_lens_cpu_extend = torch.tensor([seq_len], dtype=torch.int64)
        last_loc_extend = torch.tensor(
            [prefix_len - 1], dtype=torch.int64, device=self.device
        )
        token_positions_extend = torch.arange(
            prefix_len, seq_len, dtype=torch.int64, device=self.device
        )

        indices_extend = allocator.alloc_extend(
            prefix_lens=prefix_lens,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=seq_lens_extend,
            seq_lens_cpu=seq_lens_cpu_extend,
            last_loc=last_loc_extend,
            extend_num_tokens=extend_len,
            token_positions=token_positions_extend,
        )

        self.assertIsNotNone(indices_extend)
        self.assertEqual(len(indices_extend), extend_len)

        # Count tokens allocated for rank 0 in extend
        extend_rank_0_count = sum(
            1 for pos in range(prefix_len, seq_len) if pos % 4 == 0
        )
        extend_allocated = indices_extend[indices_extend >= 0]
        self.assertEqual(len(extend_allocated), extend_rank_0_count)

        # Second: decode with batch
        batch_size = 30
        seq_lens_decode = torch.tensor(
            [seq_len + i for i in range(batch_size)],
            dtype=torch.int64,
            device=self.device,
        )
        seq_lens_cpu_decode = seq_lens_decode.cpu()
        last_loc_decode = torch.tensor(
            [s - 2 for s in seq_lens_decode], dtype=torch.int64, device=self.device
        )
        token_positions_decode = (seq_lens_decode - 1).to(torch.int64)

        indices_decode = allocator.alloc_decode(
            seq_lens=seq_lens_decode,
            seq_lens_cpu=seq_lens_cpu_decode,
            last_loc=last_loc_decode,
            token_positions=token_positions_decode,
        )

        self.assertIsNotNone(indices_decode)
        self.assertEqual(len(indices_decode), batch_size)

        # Count tokens allocated for rank 0 in decode
        decode_rank_0_count = sum(1 for s in seq_lens_decode if (s - 1) % 4 == 0)
        decode_allocated = indices_decode[indices_decode >= 0]
        self.assertEqual(len(decode_allocated), decode_rank_0_count)

        # Verify: All indices are unique across extend and decode
        all_allocated = torch.cat([extend_allocated, decode_allocated])
        self.assertEqual(
            len(torch.unique(all_allocated)),
            len(all_allocated),
            "All allocated indices should be unique",
        )


if __name__ == "__main__":
    unittest.main()
