"""
Unit tests for _release_finished_req in DecodeKVCacheOffloadManager.

Verifies that over-allocated KV cache slots (from speculative decoding v2)
are correctly freed when a request finishes, preventing GPU memory leaks.

Requires: torch, sglang (run in an environment with sglang installed)
"""

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="stage-b-test-small-1-gpu")


def _make_mock_req(
    req_pool_idx: int,
    kv_committed_len: int,
    kv_allocated_len: int,
    prefix_indices_len: int = 0,
):
    """Create a mock Req with the KV cache state needed for testing."""
    req = MagicMock()
    req.req_pool_idx = req_pool_idx
    req.kv_committed_len = kv_committed_len
    req.kv_allocated_len = kv_allocated_len
    req.kv_committed_freed = False
    req.kv_overallocated_freed = False
    req.prefix_indices = list(range(prefix_indices_len))

    def pop_committed():
        assert not req.kv_committed_freed
        req.kv_committed_freed = True
        return req.kv_committed_len

    def pop_overallocated():
        assert not req.kv_overallocated_freed
        req.kv_overallocated_freed = True
        return req.kv_committed_len, req.kv_allocated_len

    req.pop_committed_kv_cache = pop_committed
    req.pop_overallocated_kv_cache = pop_overallocated
    return req


def _make_manager(pool_size: int, page_size: int = 1):
    """Create a DecodeKVCacheOffloadManager with mock pools for testing."""
    # Build a real req_to_token tensor so indexing works
    req_to_token = torch.arange(pool_size, dtype=torch.int64).unsqueeze(0)

    req_to_token_pool = MagicMock()
    req_to_token_pool.req_to_token = req_to_token

    freed_indices = []

    allocator = MagicMock()
    allocator.free = MagicMock(
        side_effect=lambda idx: freed_indices.append(idx.clone())
    )

    tree_cache = MagicMock()
    tree_cache.protected_size_ = 0

    # Bypass __init__ entirely and set attributes directly
    manager = object.__new__(DecodeKVCacheOffloadManager)
    manager.req_to_token_pool = req_to_token_pool
    manager.token_to_kv_pool_allocator = allocator
    manager.page_size = page_size
    manager.tree_cache = tree_cache

    return manager, freed_indices


class TestReleaseFinishedReq(unittest.TestCase):
    """Tests for _release_finished_req overallocation cleanup."""

    def test_no_overallocation(self):
        """Without spec v2, kv_committed == kv_allocated; no extra free."""
        manager, freed = _make_manager(pool_size=32)
        req = _make_mock_req(
            req_pool_idx=0,
            kv_committed_len=20,
            kv_allocated_len=20,  # no overallocation
        )
        prefill_offloaded_len = 8

        manager._release_finished_req(req, prefill_offloaded_len)

        # Only one free call: the committed range [8:20]
        self.assertEqual(len(freed), 1)
        expected = torch.arange(8, 20, dtype=torch.int64)
        self.assertTrue(torch.equal(freed[0], expected))
        manager.req_to_token_pool.free.assert_called_once_with(req)

    def test_with_overallocation(self):
        """With spec v2, overallocated slots [committed:allocated] must be freed."""
        manager, freed = _make_manager(pool_size=32)
        req = _make_mock_req(
            req_pool_idx=0,
            kv_committed_len=20,
            kv_allocated_len=28,  # 8 over-allocated slots
        )
        prefill_offloaded_len = 8

        manager._release_finished_req(req, prefill_offloaded_len)

        # Two free calls: committed [8:20] and overallocated [20:28]
        self.assertEqual(len(freed), 2)
        expected_committed = torch.arange(8, 20, dtype=torch.int64)
        expected_overalloc = torch.arange(20, 28, dtype=torch.int64)
        self.assertTrue(torch.equal(freed[0], expected_committed))
        self.assertTrue(torch.equal(freed[1], expected_overalloc))
        manager.req_to_token_pool.free.assert_called_once_with(req)

    def test_overallocation_with_page_alignment(self):
        """With page_size > 1, start of overallocated range is ceil-aligned."""
        page_size = 4
        manager, freed = _make_manager(pool_size=32, page_size=page_size)
        req = _make_mock_req(
            req_pool_idx=0,
            kv_committed_len=10,  # not page-aligned
            kv_allocated_len=28,
        )
        prefill_offloaded_len = 4

        manager._release_finished_req(req, prefill_offloaded_len)

        # Committed range [4:10]
        # Overallocated: start_p = ceil_align(10, 4) = 12, end_p = 28 => [12:28]
        self.assertEqual(len(freed), 2)
        expected_committed = torch.arange(4, 10, dtype=torch.int64)
        expected_overalloc = torch.arange(12, 28, dtype=torch.int64)
        self.assertTrue(torch.equal(freed[0], expected_committed))
        self.assertTrue(torch.equal(freed[1], expected_overalloc))

    def test_overallocation_page_aligned_noop(self):
        """When ceil_align(committed, page_size) >= allocated, no overalloc free."""
        page_size = 4
        manager, freed = _make_manager(pool_size=32, page_size=page_size)
        req = _make_mock_req(
            req_pool_idx=0,
            kv_committed_len=10,  # ceil_align(10, 4) = 12
            kv_allocated_len=12,  # same as aligned start
        )
        prefill_offloaded_len = 4

        manager._release_finished_req(req, prefill_offloaded_len)

        # Only committed [4:10], no overalloc because start_p == end_p
        self.assertEqual(len(freed), 1)
        expected_committed = torch.arange(4, 10, dtype=torch.int64)
        self.assertTrue(torch.equal(freed[0], expected_committed))

    def test_prefix_indices_decremented(self):
        """protected_size_ is decremented by len(req.prefix_indices)."""
        manager, _ = _make_manager(pool_size=32)
        manager.tree_cache.protected_size_ = 10
        req = _make_mock_req(
            req_pool_idx=0,
            kv_committed_len=20,
            kv_allocated_len=20,
            prefix_indices_len=5,
        )

        manager._release_finished_req(req, prefill_offloaded_len=0)

        self.assertEqual(manager.tree_cache.protected_size_, 5)


if __name__ == "__main__":
    unittest.main()
