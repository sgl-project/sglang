"""
Unit tests for KV return insertion logic (_process_kv_return_insertions).

Tests the scheduler-side logic that drains the KV return queue and inserts
returned KV pages into RadixCache, without requiring GPU hardware. All
scheduler and KV manager dependencies are mocked.

Covers:
- Normal insertion: prompt prefix matched, gen pages appended, tree updated
- Duplicate insertion: tree already has full sequence, orphaned pages freed
- Empty queue: no-op when nothing to process
- Page-aligned insertion: pages truncated to page_size boundary
- Cancel handling: cancelled transfers free pages without inserting
- Multiple items: queue with several entries drained in one call

Usage:
    python -m pytest test/registered/radix_cache/test_kv_return_insertion_unit.py -v
"""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

# CPU-based unit test, no GPU needed
register_cuda_ci(est_time=5, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=5, suite="stage-b-test-small-1-gpu-amd")

import queue
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    InsertParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.radix_cache import RadixKey


def _make_scheduler(
    page_size=1,
    max_total_num_tokens=100,
):
    """Create a minimal mock scheduler with the attributes needed by
    _process_kv_return_insertions."""
    from sglang.srt.managers.scheduler_runtime_checker_mixin import (
        SchedulerRuntimeCheckerMixin,
    )

    sched = MagicMock()

    # Bind the real method under test
    sched._process_kv_return_insertions = (
        SchedulerRuntimeCheckerMixin.__dict__  # bypass descriptor protocol
    )
    # Actually, _process_kv_return_insertions is on Scheduler, not the mixin.
    # We need to import it from scheduler. Since importing scheduler pulls in
    # heavy deps, we bind the method manually from its source module.
    import types

    from sglang.srt.managers.scheduler import Scheduler

    sched._process_kv_return_insertions = types.MethodType(
        Scheduler._process_kv_return_insertions, sched
    )

    # Tree cache mock
    sched.tree_cache = MagicMock()
    sched.tree_cache.page_size = page_size

    # Token allocator mock
    sched.token_to_kv_pool_allocator = MagicMock()
    sched.token_to_kv_pool_allocator.free = MagicMock()

    # KV manager mock with insertion queue
    kv_mgr = MagicMock()
    kv_mgr.kv_return_insertion_queue = queue.Queue()
    sched.disagg_prefill_bootstrap_queue = MagicMock()
    sched.disagg_prefill_bootstrap_queue.kv_manager = kv_mgr

    # Device (CPU for testing)
    sched.device = "cpu"

    sched.max_total_num_tokens = max_total_num_tokens

    # Server args
    sched.server_args = MagicMock()
    sched.server_args.enable_kv_return = True

    return sched, kv_mgr


class TestKvReturnInsertion(unittest.TestCase):
    """Tests for _process_kv_return_insertions scheduler method."""

    def test_empty_queue_noop(self):
        """No items in queue — nothing happens, no crash."""
        sched, _ = _make_scheduler()
        sched._process_kv_return_insertions()

        sched.tree_cache.insert.assert_not_called()
        sched.token_to_kv_pool_allocator.free.assert_not_called()

    def test_normal_insertion(self):
        """Normal case: prompt prefix matched, gen pages concatenated and inserted."""
        sched, kv_mgr = _make_scheduler()

        # Simulate: 8 prompt tokens + 3 generated tokens = 11 total
        token_ids = list(range(11))
        gen_page_indices = [50, 51, 52]  # 3 gen pages

        # match_prefix returns the 8 prompt pages
        prompt_pages = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17], dtype=torch.int64)
        sched.tree_cache.match_prefix.return_value = MatchResult(
            device_indices=prompt_pages,
            last_device_node=MagicMock(),
            last_host_node=MagicMock(),
        )

        kv_mgr.kv_return_insertion_queue.put((token_ids, gen_page_indices, False))
        sched._process_kv_return_insertions()

        # tree_cache.insert() called once
        sched.tree_cache.insert.assert_called_once()
        call_args = sched.tree_cache.insert.call_args[0][0]
        self.assertIsInstance(call_args, InsertParams)
        # Value = prompt_pages + gen_pages = 11 entries
        self.assertEqual(len(call_args.value), 11)
        # Key = all 11 token_ids
        self.assertEqual(len(call_args.key), 11)
        # free() NOT called (pages are in tree, not orphaned)
        sched.token_to_kv_pool_allocator.free.assert_not_called()

    def test_duplicate_insertion_frees_orphaned_pages(self):
        """Duplicate: tree already has full sequence — gen pages freed, not inserted."""
        sched, kv_mgr = _make_scheduler()

        # 8 prompt + 2 gen = 10 tokens total
        token_ids = list(range(10))
        gen_page_indices = [90, 91]

        # match_prefix returns ALL 10 pages (tree has complete sequence)
        full_pages = torch.tensor(list(range(10)), dtype=torch.int64)
        sched.tree_cache.match_prefix.return_value = MatchResult(
            device_indices=full_pages,
            last_device_node=MagicMock(),
            last_host_node=MagicMock(),
        )

        kv_mgr.kv_return_insertion_queue.put((token_ids, gen_page_indices, False))
        sched._process_kv_return_insertions()

        # tree_cache.insert() NOT called (duplicate)
        sched.tree_cache.insert.assert_not_called()
        # Orphaned gen pages freed back to allocator
        sched.token_to_kv_pool_allocator.free.assert_called_once()
        freed = sched.token_to_kv_pool_allocator.free.call_args[0][0]
        self.assertEqual(freed.tolist(), [90, 91])

    def test_page_aligned_truncation(self):
        """With page_size > 1, key/value truncated to page boundary."""
        sched, kv_mgr = _make_scheduler(page_size=4)

        # 6 prompt + 2 gen = 8 tokens, but full_value has 8 entries
        # 8 is already aligned to page_size=4 so no truncation needed
        token_ids = list(range(8))
        gen_page_indices = [40, 41]

        prompt_pages = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int64)
        sched.tree_cache.match_prefix.return_value = MatchResult(
            device_indices=prompt_pages,
            last_device_node=MagicMock(),
            last_host_node=MagicMock(),
        )

        kv_mgr.kv_return_insertion_queue.put((token_ids, gen_page_indices, False))
        sched._process_kv_return_insertions()

        call_args = sched.tree_cache.insert.call_args[0][0]
        # 8 entries, aligned to page_size=4 → 8 (no truncation)
        self.assertEqual(len(call_args.value), 8)
        self.assertEqual(len(call_args.key), 8)

    def test_page_aligned_truncation_odd(self):
        """With page_size=4 and 9 total entries, truncation to 8."""
        sched, kv_mgr = _make_scheduler(page_size=4)

        # 6 prompt + 3 gen = 9 tokens total
        token_ids = list(range(9))
        gen_page_indices = [40, 41, 42]

        prompt_pages = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int64)
        sched.tree_cache.match_prefix.return_value = MatchResult(
            device_indices=prompt_pages,
            last_device_node=MagicMock(),
            last_host_node=MagicMock(),
        )

        kv_mgr.kv_return_insertion_queue.put((token_ids, gen_page_indices, False))
        sched._process_kv_return_insertions()

        call_args = sched.tree_cache.insert.call_args[0][0]
        # 9 entries truncated to 8 (floor of 9/4 * 4 = 8)
        self.assertEqual(len(call_args.value), 8)
        self.assertEqual(len(call_args.key), 8)

    def test_multiple_items_drained(self):
        """Multiple queue items processed in a single call."""
        sched, kv_mgr = _make_scheduler()

        # Item 1: normal insertion (5 prompt + 2 gen)
        kv_mgr.kv_return_insertion_queue.put((list(range(7)), [50, 51], False))
        # Item 2: different prompt (4 prompt + 2 gen)
        kv_mgr.kv_return_insertion_queue.put((list(range(100, 106)), [60, 61], False))

        def match_side_effect(params):
            n = len(params.key.token_ids)
            # Return prompt-length pages (total - 2 gen tokens)
            return MatchResult(
                device_indices=torch.arange(n - 2, dtype=torch.int64),
                last_device_node=MagicMock(),
                last_host_node=MagicMock(),
            )

        sched.tree_cache.match_prefix.side_effect = match_side_effect

        sched._process_kv_return_insertions()

        # Both inserted
        self.assertEqual(sched.tree_cache.insert.call_count, 2)

    def test_cancel_frees_pages(self):
        """Cancelled transfer frees allocated pages without inserting."""
        sched, kv_mgr = _make_scheduler()

        # Simulate a cancelled transfer with 3 pages
        token_ids = list(range(8))
        gen_page_indices = [50, 51, 52]

        kv_mgr.kv_return_insertion_queue.put((token_ids, gen_page_indices, True))
        sched._process_kv_return_insertions()

        # insert() NOT called (cancelled)
        sched.tree_cache.insert.assert_not_called()
        # Pages freed back to allocator
        sched.token_to_kv_pool_allocator.free.assert_called_once()
        freed = sched.token_to_kv_pool_allocator.free.call_args[0][0]
        self.assertEqual(freed.tolist(), [50, 51, 52])

    def test_no_kv_return_queue_attribute(self):
        """If kv_mgr has no kv_return_insertion_queue, method returns early."""
        sched, kv_mgr = _make_scheduler()
        del kv_mgr.kv_return_insertion_queue  # simulate attribute absence

        # Should not raise
        sched._process_kv_return_insertions()
        sched.tree_cache.insert.assert_not_called()


if __name__ == "__main__":
    unittest.main()
