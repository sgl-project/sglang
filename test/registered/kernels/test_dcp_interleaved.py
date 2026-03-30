"""Tests for DCP interleaved KV cache storage.

Verifies that tokens are correctly distributed across DCP ranks based on
token_idx % dcp_world_size == dcp_rank for both TokenToKVPoolAllocator
and PagedTokenToKVPoolAllocator.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, suite="stage-b-test-1-gpu-small")


class MockKVCache:
    def __init__(self, size, dtype, device):
        self.size = size
        self.dtype = dtype
        self.device = device

    def get_cpu_copy(self, indices):
        return None

    def load_cpu_copy(self, kv_cache_cpu, indices):
        pass


class TestDCPInterleavedTokenAllocator(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.dtype = torch.bfloat16

    def setUp(self):
        self.size = 100
        self.page_size = 16
        self.kvcache = MockKVCache(self.size, self.dtype, self.device)

    @patch("sglang.srt.mem_cache.allocator.get_dcp_world_size")
    @patch("sglang.srt.mem_cache.allocator.get_dcp_rank")
    def test_rank0_interleaved(self, mock_rank, mock_world_size):
        """Rank 0 stores positions 0,2,4 from [0..4] with dcp_size=2."""
        mock_world_size.return_value = 2
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

        initial_available = allocator.available_size()
        indices = allocator.alloc(5, token_positions=token_positions)

        self.assertIsNotNone(indices)
        self.assertEqual(len(indices), 5)
        self.assertEqual(indices[1].item(), -1)
        self.assertEqual(indices[3].item(), -1)
        self.assertNotEqual(indices[0].item(), -1)
        self.assertNotEqual(indices[2].item(), -1)
        self.assertNotEqual(indices[4].item(), -1)
        self.assertEqual(initial_available - allocator.available_size(), 3)

    @patch("sglang.srt.mem_cache.allocator.get_dcp_world_size")
    @patch("sglang.srt.mem_cache.allocator.get_dcp_rank")
    def test_rank1_interleaved(self, mock_rank, mock_world_size):
        """Rank 1 stores positions 1,3 from [0..4] with dcp_size=2."""
        mock_world_size.return_value = 2
        mock_rank.return_value = 1

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

        initial_available = allocator.available_size()
        indices = allocator.alloc(5, token_positions=token_positions)

        self.assertIsNotNone(indices)
        self.assertEqual(indices[0].item(), -1)
        self.assertEqual(indices[2].item(), -1)
        self.assertEqual(indices[4].item(), -1)
        self.assertNotEqual(indices[1].item(), -1)
        self.assertNotEqual(indices[3].item(), -1)
        self.assertEqual(initial_available - allocator.available_size(), 2)

    @patch("sglang.srt.mem_cache.allocator.get_dcp_world_size")
    @patch("sglang.srt.mem_cache.allocator.get_dcp_rank")
    def test_no_dcp(self, mock_rank, mock_world_size):
        """All tokens allocated when dcp_world_size=1."""
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

        initial_available = allocator.available_size()
        indices = allocator.alloc(5, token_positions=token_positions)

        self.assertIsNotNone(indices)
        self.assertTrue(torch.all(indices > 0))
        self.assertEqual(initial_available - allocator.available_size(), 5)

    @patch("sglang.srt.mem_cache.allocator.get_dcp_world_size")
    @patch("sglang.srt.mem_cache.allocator.get_dcp_rank")
    def test_3_ranks(self, mock_rank, mock_world_size):
        """3 DCP ranks, rank 1 stores positions 1,4 from [0..5]."""
        mock_world_size.return_value = 3
        mock_rank.return_value = 1

        allocator = TokenToKVPoolAllocator(
            size=self.size,
            dtype=self.dtype,
            device=self.device,
            kvcache=self.kvcache,
            need_sort=False,
        )

        token_positions = torch.tensor(
            [0, 1, 2, 3, 4, 5], dtype=torch.int64, device=self.device
        )
        indices = allocator.alloc(6, token_positions=token_positions)

        self.assertEqual(indices[0].item(), -1)
        self.assertNotEqual(indices[1].item(), -1)
        self.assertEqual(indices[2].item(), -1)
        self.assertEqual(indices[3].item(), -1)
        self.assertNotEqual(indices[4].item(), -1)
        self.assertEqual(indices[5].item(), -1)

    @patch("sglang.srt.mem_cache.allocator.get_dcp_world_size")
    @patch("sglang.srt.mem_cache.allocator.get_dcp_rank")
    def test_4_ranks_large_seq(self, mock_rank, mock_world_size):
        """4 DCP ranks, 100 tokens, each rank gets ~25 tokens."""
        mock_world_size.return_value = 4

        for rank in range(4):
            mock_rank.return_value = rank

            allocator = TokenToKVPoolAllocator(
                size=1000,
                dtype=self.dtype,
                device=self.device,
                kvcache=self.kvcache,
                need_sort=False,
            )

            seq_len = 100
            token_positions = torch.arange(
                seq_len, dtype=torch.int64, device=self.device
            )

            initial_available = allocator.available_size()
            indices = allocator.alloc(seq_len, token_positions=token_positions)

            expected_count = 0
            for pos in range(seq_len):
                if pos % 4 == rank:
                    self.assertNotEqual(indices[pos].item(), -1)
                    expected_count += 1
                else:
                    self.assertEqual(indices[pos].item(), -1)

            self.assertEqual(
                initial_available - allocator.available_size(), expected_count
            )

            allocated_indices = indices[indices >= 0]
            self.assertEqual(len(torch.unique(allocated_indices)), expected_count)

    @patch("sglang.srt.mem_cache.allocator.get_dcp_world_size")
    @patch("sglang.srt.mem_cache.allocator.get_dcp_rank")
    def test_alloc_then_free(self, mock_rank, mock_world_size):
        """Alloc+free cycle returns tokens to pool."""
        mock_world_size.return_value = 2
        mock_rank.return_value = 0

        allocator = TokenToKVPoolAllocator(
            size=100,
            dtype=self.dtype,
            device=self.device,
            kvcache=self.kvcache,
            need_sort=False,
        )

        token_positions_1 = torch.tensor(
            [0, 1, 2, 3, 4], dtype=torch.int64, device=self.device
        )
        indices_1 = allocator.alloc(5, token_positions=token_positions_1)
        available_after_alloc = allocator.available_size()

        rank_0_indices = indices_1[indices_1 > 0]
        allocator.free(rank_0_indices)
        available_after_free = allocator.available_size()

        self.assertGreater(available_after_free, available_after_alloc)


class TestDCPInterleavedPagedAllocator(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.dtype = torch.bfloat16

    def setUp(self):
        self.page_size = 16
        self.kvcache = MockKVCache(100, self.dtype, self.device)

    @patch("sglang.srt.mem_cache.allocator.get_dcp_world_size")
    @patch("sglang.srt.mem_cache.allocator.get_dcp_rank")
    def test_extend_interleaved(self, mock_rank, mock_world_size):
        """Paged extend with DCP interleaved storage."""
        mock_world_size.return_value = 2
        mock_rank.return_value = 0

        allocator = PagedTokenToKVPoolAllocator(
            size=100,
            page_size=self.page_size,
            dtype=self.dtype,
            device=self.device,
            kvcache=self.kvcache,
            need_sort=False,
        )

        prefix_lens = torch.tensor([10], dtype=torch.int64, device=self.device)
        prefix_lens_cpu = torch.tensor([10], dtype=torch.int64)
        seq_lens = torch.tensor([15], dtype=torch.int64, device=self.device)
        seq_lens_cpu = torch.tensor([15], dtype=torch.int64)
        last_loc = torch.tensor([9], dtype=torch.int64, device=self.device)
        token_positions = torch.tensor(
            [10, 11, 12, 13, 14], dtype=torch.int64, device=self.device
        )

        indices = allocator.alloc_extend(
            prefix_lens=prefix_lens,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            last_loc=last_loc,
            extend_num_tokens=5,
            token_positions=token_positions,
        )

        self.assertEqual(len(indices), 5)
        self.assertEqual(indices[1].item(), -1)
        self.assertEqual(indices[3].item(), -1)
        self.assertNotEqual(indices[0].item(), -1)
        self.assertNotEqual(indices[2].item(), -1)
        self.assertNotEqual(indices[4].item(), -1)

    @patch("sglang.srt.mem_cache.allocator.get_dcp_world_size")
    @patch("sglang.srt.mem_cache.allocator.get_dcp_rank")
    def test_decode_interleaved(self, mock_rank, mock_world_size):
        """Paged decode with DCP interleaved storage."""
        mock_world_size.return_value = 2
        mock_rank.return_value = 0

        allocator = PagedTokenToKVPoolAllocator(
            size=100,
            page_size=self.page_size,
            dtype=self.dtype,
            device=self.device,
            kvcache=self.kvcache,
            need_sort=False,
        )

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

        self.assertEqual(len(indices), 3)
        self.assertNotEqual(indices[1].item(), -1)
        self.assertEqual(indices[0].item(), -1)
        self.assertEqual(indices[2].item(), -1)

    @patch("sglang.srt.mem_cache.allocator.get_dcp_world_size")
    @patch("sglang.srt.mem_cache.allocator.get_dcp_rank")
    def test_extend_no_token_positions(self, mock_rank, mock_world_size):
        """Backward compat: extend without token_positions allocates all tokens."""
        mock_world_size.return_value = 2
        mock_rank.return_value = 0

        allocator = PagedTokenToKVPoolAllocator(
            size=100,
            page_size=self.page_size,
            dtype=self.dtype,
            device=self.device,
            kvcache=self.kvcache,
            need_sort=False,
        )

        prefix_lens = torch.tensor([10], dtype=torch.int64, device=self.device)
        prefix_lens_cpu = torch.tensor([10], dtype=torch.int64)
        seq_lens = torch.tensor([15], dtype=torch.int64, device=self.device)
        seq_lens_cpu = torch.tensor([15], dtype=torch.int64)
        last_loc = torch.tensor([9], dtype=torch.int64, device=self.device)

        indices = allocator.alloc_extend(
            prefix_lens=prefix_lens,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            last_loc=last_loc,
            extend_num_tokens=5,
            token_positions=None,
        )

        self.assertEqual(len(indices), 5)
        self.assertTrue(torch.all(indices > 0))

    @patch("sglang.srt.mem_cache.allocator.get_dcp_world_size")
    @patch("sglang.srt.mem_cache.allocator.get_dcp_rank")
    def test_4_ranks_extend_large_seq(self, mock_rank, mock_world_size):
        """4 DCP ranks, large extend sequence."""
        mock_world_size.return_value = 4

        for rank in range(4):
            mock_rank.return_value = rank

            allocator = PagedTokenToKVPoolAllocator(
                size=1000,
                page_size=self.page_size,
                dtype=self.dtype,
                device=self.device,
                kvcache=MockKVCache(1000, self.dtype, self.device),
                need_sort=False,
            )

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
            token_positions = torch.arange(
                prefix_len, seq_len, dtype=torch.int64, device=self.device
            )

            indices = allocator.alloc_extend(
                prefix_lens=prefix_lens,
                prefix_lens_cpu=prefix_lens_cpu,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                last_loc=last_loc,
                extend_num_tokens=extend_len,
                token_positions=token_positions,
            )

            self.assertEqual(len(indices), extend_len)

            expected_count = 0
            for i, pos in enumerate(range(prefix_len, seq_len)):
                if pos % 4 == rank:
                    self.assertNotEqual(indices[i].item(), -1)
                    expected_count += 1
                else:
                    self.assertEqual(indices[i].item(), -1)

            allocated_indices = indices[indices >= 0]
            self.assertEqual(len(allocated_indices), expected_count)
            self.assertEqual(len(torch.unique(allocated_indices)), expected_count)

    @patch("sglang.srt.mem_cache.allocator.get_dcp_world_size")
    @patch("sglang.srt.mem_cache.allocator.get_dcp_rank")
    def test_4_ranks_decode_large_batch(self, mock_rank, mock_world_size):
        """4 DCP ranks, large decode batch."""
        mock_world_size.return_value = 4

        for rank in range(4):
            mock_rank.return_value = rank

            allocator = PagedTokenToKVPoolAllocator(
                size=1000,
                page_size=self.page_size,
                dtype=self.dtype,
                device=self.device,
                kvcache=MockKVCache(1000, self.dtype, self.device),
                need_sort=False,
            )

            batch_size = 50
            seq_lens_list = [100 + i for i in range(batch_size)]
            seq_lens = torch.tensor(
                seq_lens_list, dtype=torch.int64, device=self.device
            )
            seq_lens_cpu = torch.tensor(seq_lens_list, dtype=torch.int64)
            last_loc = torch.tensor(
                [s - 2 for s in seq_lens_list], dtype=torch.int64, device=self.device
            )
            token_positions = (seq_lens - 1).to(torch.int64)

            indices = allocator.alloc_decode(
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                last_loc=last_loc,
                token_positions=token_positions,
            )

            self.assertEqual(len(indices), batch_size)

            expected_count = 0
            for i, seq_len_val in enumerate(seq_lens_list):
                token_pos = seq_len_val - 1
                if token_pos % 4 == rank:
                    self.assertNotEqual(indices[i].item(), -1)
                    expected_count += 1
                else:
                    self.assertEqual(indices[i].item(), -1)

            allocated_indices = indices[indices >= 0]
            self.assertEqual(len(allocated_indices), expected_count)
            if expected_count > 0:
                self.assertEqual(len(torch.unique(allocated_indices)), expected_count)


if __name__ == "__main__":
    unittest.main()
