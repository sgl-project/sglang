"""
Unit tests for _release_finished_req in DecodeKVCacheOffloadManager.

Verifies that over-allocated KV cache slots (from speculative decoding v2)
are correctly freed when a request finishes, preventing GPU memory leaks.

Requires: torch, sglang (run in an environment with sglang installed)
"""

import gc
import unittest
from unittest.mock import MagicMock
from weakref import WeakKeyDictionary as WeakKeyDict

import torch

from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.srt.disaggregation.kv_events import OffloadedState
from sglang.srt.managers.cache_controller import HiCacheAck
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _make_mock_req(
    req_pool_idx: int,
    kv_committed_len: int,
    kv_allocated_len: int,
    prefix_indices_len: int = 0,
    rid: int = 0,
    origin_len: int = 0,
):
    """Create a mock Req with the KV cache state needed for testing."""
    req = MagicMock()
    req.rid = rid
    req.origin_input_ids = list(range(origin_len))
    req.kv = ReqKvInfo(
        req_pool_idx=req_pool_idx,
        kv_committed_len=kv_committed_len,
        kv_allocated_len=kv_allocated_len,
    )
    req.prefix_indices = list(range(prefix_indices_len))
    req.effective_kv_committed_len = lambda: req.kv.kv_committed_len
    return req


class _RecordingAllocator(BaseTokenToKVPoolAllocator):
    """Single-pool double. Subclassing the base routes free_full / free_segment /
    free_segments into free(), so a new free API cannot slip past the recorder."""

    def __init__(self, page_size: int):
        super().__init__(
            size=1024,
            page_size=page_size,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        self.freed = []

    def clear(self):
        self.freed = []

    def alloc(self, need_size: int):
        raise NotImplementedError

    def free(self, free_index: torch.Tensor):
        self.freed.append(free_index.clone())


def _make_manager(pool_size: int, page_size: int = 1):
    """Create a DecodeKVCacheOffloadManager with mock pools for testing."""
    # Build a real req_to_token tensor so indexing works
    req_to_token = torch.arange(pool_size, dtype=torch.int64).unsqueeze(0)

    req_to_token_pool = MagicMock()
    req_to_token_pool.req_to_token = req_to_token

    allocator = _RecordingAllocator(page_size)
    freed_indices = allocator.freed

    tree_cache = MagicMock()
    tree_cache.protected_size_ = 0
    tree_cache.req_to_token_pool = req_to_token_pool
    tree_cache.token_to_kv_pool_allocator = allocator
    tree_cache.free_kv_row = lambda owner, ranges: BasePrefixCache.free_kv_row(
        tree_cache, owner, ranges
    )

    # Bypass __init__ entirely and set attributes directly
    manager = object.__new__(DecodeKVCacheOffloadManager)
    manager.req_to_token_pool = req_to_token_pool
    manager.token_to_kv_pool_allocator = allocator
    manager.page_size = page_size
    manager.tree_cache = tree_cache
    manager.offloaded_state = WeakKeyDict()
    manager.ongoing_offload = {}
    manager.ongoing_backup = {}
    manager.offload_inflight = WeakKeyDict()

    return manager, freed_indices


class _FinishedEvent:
    def synchronize(self):
        pass


class TestReleaseFinishedReq(unittest.TestCase):
    """Tests for _release_finished_req overallocation cleanup."""

    def test_no_overallocation(self):
        """Without spec v2, kv_committed == kv_allocated; no extra free."""
        manager, freed = _make_manager(pool_size=32)
        req = _make_mock_req(
            req_pool_idx=0,
            kv_committed_len=20,
            kv_allocated_len=20,  # no overallocation
            origin_len=8,
        )

        manager._release_finished_req(req)

        # Prefill [0:8] and committed [8:20]; no overalloc free.
        self.assertEqual(len(freed), 2)
        self.assertTrue(torch.equal(freed[0], torch.arange(0, 8, dtype=torch.int64)))
        self.assertTrue(torch.equal(freed[1], torch.arange(8, 20, dtype=torch.int64)))
        manager.req_to_token_pool.free.assert_called_once_with(req)

    def test_with_overallocation(self):
        """With spec v2, overallocated slots [committed:allocated] must be freed."""
        manager, freed = _make_manager(pool_size=32)
        req = _make_mock_req(
            req_pool_idx=0,
            kv_committed_len=20,
            kv_allocated_len=28,  # 8 over-allocated slots
            origin_len=8,
        )

        manager._release_finished_req(req)

        # Prefill [0:8], committed [8:20], overallocated [20:28].
        self.assertEqual(len(freed), 3)
        self.assertTrue(torch.equal(freed[0], torch.arange(0, 8, dtype=torch.int64)))
        self.assertTrue(torch.equal(freed[1], torch.arange(8, 20, dtype=torch.int64)))
        self.assertTrue(torch.equal(freed[2], torch.arange(20, 28, dtype=torch.int64)))
        manager.req_to_token_pool.free.assert_called_once_with(req)

    def test_overallocation_with_page_alignment(self):
        """With page_size > 1, start of overallocated range is ceil-aligned."""
        page_size = 4
        manager, freed = _make_manager(pool_size=32, page_size=page_size)
        req = _make_mock_req(
            req_pool_idx=0,
            kv_committed_len=10,  # not page-aligned
            kv_allocated_len=28,
            origin_len=4,
        )

        manager._release_finished_req(req)

        # Prefill [0:4], committed [4:10],
        # overallocated: start_p = ceil_align(10, 4) = 12, end_p = 28 => [12:28]
        self.assertEqual(len(freed), 3)
        self.assertTrue(torch.equal(freed[0], torch.arange(0, 4, dtype=torch.int64)))
        self.assertTrue(torch.equal(freed[1], torch.arange(4, 10, dtype=torch.int64)))
        self.assertTrue(torch.equal(freed[2], torch.arange(12, 28, dtype=torch.int64)))

    def test_overallocation_page_aligned_noop(self):
        """When ceil_align(committed, page_size) >= allocated, no overalloc free."""
        page_size = 4
        manager, freed = _make_manager(pool_size=32, page_size=page_size)
        req = _make_mock_req(
            req_pool_idx=0,
            kv_committed_len=10,  # ceil_align(10, 4) = 12
            kv_allocated_len=12,  # same as aligned start
            origin_len=4,
        )

        manager._release_finished_req(req)

        # Prefill [0:4] and committed [4:10]; no overalloc since start_p == end_p
        self.assertEqual(len(freed), 2)
        self.assertTrue(torch.equal(freed[0], torch.arange(0, 4, dtype=torch.int64)))
        self.assertTrue(torch.equal(freed[1], torch.arange(4, 10, dtype=torch.int64)))

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

        manager._release_finished_req(req)

        self.assertEqual(manager.tree_cache.protected_size_, 5)

    def test_release_finished_req_frees_prefill_and_pops_state(self):
        """
        _release_finished_req frees the prefill-aligned slots in addition to
        the committed range; freeing them mid-decode instead races with
        concurrent admission and cross-pollinates KV reads.
        """
        manager, freed = _make_manager(pool_size=32)
        req = _make_mock_req(
            req_pool_idx=0,
            kv_committed_len=20,
            kv_allocated_len=20,
            rid="req-prefill-present",
            origin_len=8,
        )
        manager.offloaded_state[req] = OffloadedState(inc_len=4)

        manager._release_finished_req(req)

        # Two frees in order: prefill [0:8] then committed [8:20].
        self.assertEqual(len(freed), 2)
        self.assertTrue(torch.equal(freed[0], torch.arange(0, 8, dtype=torch.int64)))
        self.assertTrue(torch.equal(freed[1], torch.arange(8, 20, dtype=torch.int64)))
        # State entry is removed at the end of _release_finished_req.
        self.assertNotIn(req, manager.offloaded_state)

    def test_release_finished_req_skips_prefill_free_when_prompt_below_page(self):
        """
        When the prompt is shorter than page_size (no prefill chunk was ever
        offloaded), no prefill-aligned free is emitted.
        """
        manager, freed = _make_manager(pool_size=32, page_size=4)
        req = _make_mock_req(
            req_pool_idx=0,
            kv_committed_len=10,
            kv_allocated_len=10,
            rid="req-prefill-zero",
            origin_len=3,  # 3 // 4 * 4 == 0
        )

        manager._release_finished_req(req)

        # Only the committed range [0:10] is freed.
        self.assertEqual(len(freed), 1)
        self.assertTrue(torch.equal(freed[0], torch.arange(0, 10, dtype=torch.int64)))

    def test_finalize_release_frees_prefill_without_prior_state(self):
        """
        finalize_release_on_finish handles the case where no incremental
        offload ever ran: the prefill-aligned slots must still be freed by
        the consolidated free site in _release_finished_req.
        """
        manager, freed = _make_manager(pool_size=32, page_size=4)
        req = _make_mock_req(
            req_pool_idx=0,
            kv_committed_len=13,
            kv_allocated_len=13,
            rid="req-finalize-no-state",
            origin_len=12,  # prefill_len = 12 // 4 * 4 = 12
        )

        manager.finalize_release_on_finish(req)

        # _release_finished_req frees prefill [0:12] then committed [12:13].
        self.assertEqual(len(freed), 2)
        expected_prefill = torch.arange(0, 12, dtype=torch.int64)
        expected_committed = torch.arange(12, 13, dtype=torch.int64)
        self.assertTrue(torch.equal(freed[0], expected_prefill))
        self.assertTrue(torch.equal(freed[1], expected_committed))
        # No state entry is left behind.
        self.assertNotIn(req, manager.offloaded_state)

    def test_unfinished_offload_ack_does_not_free_incremental_slots(self):
        manager, freed = _make_manager(pool_size=32)
        req = _make_mock_req(
            req_pool_idx=0, kv_committed_len=20, kv_allocated_len=20, rid=1
        )
        req.finished.return_value = False
        manager.offloaded_state[req] = OffloadedState(inc_len=4)
        manager.offload_inflight[req] = 1
        manager.ongoing_offload[7] = (
            req,
            torch.arange(4, 8, dtype=torch.int64),
            [10, 11, 12, 13],
            0.0,
        )
        manager.cache_controller = MagicMock()
        manager.cache_controller.ack_write_queue = [
            HiCacheAck(None, _FinishedEvent(), [7])
        ]
        manager._trigger_backup = MagicMock(return_value="last_hash")

        manager._check_offload_progress(1)

        self.assertEqual(freed, [])
        manager.req_to_token_pool.free.assert_not_called()
        self.assertNotIn(req, manager.offload_inflight)

    def test_offload_kv_cache_tracks_inflight_write_until_ack(self):
        manager, freed = _make_manager(pool_size=32, page_size=4)
        manager.cache_controller = MagicMock()
        manager.cache_controller.get_hash_str = MagicMock(return_value="prefill_hash")
        manager.cache_controller.write = MagicMock(
            return_value=torch.arange(4, 8, dtype=torch.int64)
        )
        manager.decode_host_mem_pool = MagicMock()
        manager.request_counter = 0
        manager.offload_stride = 4

        req = _make_mock_req(
            req_pool_idx=0, kv_committed_len=20, kv_allocated_len=20, rid=5
        )
        req.origin_input_ids = [0, 1, 2, 3]
        req.output_ids = [4, 5, 6, 7, 8]
        req.finished.return_value = False

        did_offload = manager.offload_kv_cache(req)

        self.assertTrue(did_offload)
        self.assertEqual(manager.offload_inflight[req], 1)
        self.assertEqual(manager.offloaded_state[req].inc_len, 4)
        manager.cache_controller.write.assert_called_once()

        manager.cache_controller.ack_write_queue = [
            HiCacheAck(None, _FinishedEvent(), [1])
        ]
        manager._trigger_backup = MagicMock(return_value="last_hash")

        manager._check_offload_progress(1)

        self.assertEqual(freed, [])
        self.assertNotIn(req, manager.offload_inflight)

    def test_reused_rid_does_not_share_offload_lifecycle(self):
        manager, _ = _make_manager(pool_size=32, page_size=4)
        manager.cache_controller = MagicMock()
        manager.cache_controller.get_hash_str.return_value = "prefill_hash"
        manager.cache_controller.write.return_value = torch.arange(
            4, 8, dtype=torch.int64
        )
        manager.decode_host_mem_pool = MagicMock()
        manager.request_counter = 0
        manager.offload_stride = 4

        old_req = _make_mock_req(
            req_pool_idx=0, kv_committed_len=20, kv_allocated_len=20, rid="reused"
        )
        new_req = _make_mock_req(
            req_pool_idx=0, kv_committed_len=20, kv_allocated_len=20, rid="reused"
        )
        for req in (old_req, new_req):
            req.origin_input_ids = [0, 1, 2, 3]
            req.output_ids = [4, 5, 6, 7, 8]
            req.finished.return_value = False

        self.assertTrue(manager.offload_kv_cache(old_req))
        old_req.finished.return_value = True

        # A completed request leaves the API before its asynchronous D2H copy
        # necessarily finishes, so a caller can reuse the same rid here.
        self.assertTrue(manager.offload_kv_cache(new_req))
        self.assertIsNot(old_req, new_req)
        self.assertIn(old_req, manager.offloaded_state)
        self.assertIn(new_req, manager.offloaded_state)
        self.assertEqual(manager.offload_inflight[old_req], 1)
        self.assertEqual(manager.offload_inflight[new_req], 1)

        manager.cache_controller.ack_write_queue = [
            HiCacheAck(None, _FinishedEvent(), [1])
        ]
        manager._trigger_backup = MagicMock(return_value="old_last_hash")
        manager._check_offload_progress(1)

        self.assertNotIn(old_req, manager.offloaded_state)
        self.assertNotIn(old_req, manager.offload_inflight)
        self.assertIn(new_req, manager.offloaded_state)
        self.assertEqual(manager.offloaded_state[new_req].inc_len, 4)
        self.assertEqual(manager.offload_inflight[new_req], 1)
        self.assertIn(2, manager.ongoing_offload)

    def test_finalize_release_defers_while_offload_is_in_flight(self):
        manager, freed = _make_manager(pool_size=32)
        req = _make_mock_req(
            req_pool_idx=0, kv_committed_len=20, kv_allocated_len=20, rid=2
        )
        manager.offloaded_state[req] = OffloadedState(inc_len=8)
        manager.offload_inflight[req] = 1

        manager.finalize_release_on_finish(req)

        self.assertEqual(freed, [])
        manager.req_to_token_pool.free.assert_not_called()
        self.assertIn(req, manager.offloaded_state)

    def test_finished_offload_ack_waits_for_other_inflight_writes(self):
        manager, freed = _make_manager(pool_size=32)
        req = _make_mock_req(
            req_pool_idx=0, kv_committed_len=20, kv_allocated_len=20, rid=3
        )
        req.finished.return_value = True
        manager.offloaded_state[req] = OffloadedState(inc_len=8)
        manager.offload_inflight[req] = 2
        manager.ongoing_offload[8] = (
            req,
            torch.arange(4, 8, dtype=torch.int64),
            [10, 11, 12, 13],
            0.0,
        )
        manager.cache_controller = MagicMock()
        manager.cache_controller.ack_write_queue = [
            HiCacheAck(None, _FinishedEvent(), [8])
        ]
        manager._trigger_backup = MagicMock(return_value="last_hash")

        manager._check_offload_progress(1)

        self.assertEqual(freed, [])
        manager.req_to_token_pool.free.assert_not_called()
        self.assertEqual(manager.offload_inflight[req], 1)

    def test_finished_request_releases_all_committed_slots_after_last_offload_ack(
        self,
    ):
        manager, freed = _make_manager(pool_size=32)
        req = _make_mock_req(
            req_pool_idx=0,
            kv_committed_len=20,
            kv_allocated_len=20,
            rid=4,
            origin_len=4,
        )
        req.finished.return_value = True
        manager.offloaded_state[req] = OffloadedState(inc_len=8)
        manager.offload_inflight[req] = 1
        manager.ongoing_offload[9] = (
            req,
            torch.arange(8, 12, dtype=torch.int64),
            [14, 15, 16, 17],
            0.0,
        )
        manager.cache_controller = MagicMock()
        manager.cache_controller.ack_write_queue = [
            HiCacheAck(None, _FinishedEvent(), [9])
        ]
        manager._trigger_backup = MagicMock(return_value="last_hash")

        manager._check_offload_progress(1)

        self.assertEqual(len(freed), 2)
        self.assertTrue(torch.equal(freed[0], torch.arange(0, 4, dtype=torch.int64)))
        self.assertTrue(torch.equal(freed[1], torch.arange(4, 20, dtype=torch.int64)))
        manager.req_to_token_pool.free.assert_called_once_with(req)
        self.assertNotIn(req, manager.offloaded_state)
        self.assertNotIn(req, manager.offload_inflight)

    def test_dropped_req_does_not_pin_offload_state(self):
        manager, _ = _make_manager(pool_size=32)
        req = _make_mock_req(
            req_pool_idx=0, kv_committed_len=20, kv_allocated_len=20, rid=6
        )
        manager.offloaded_state[req] = OffloadedState(inc_len=4)
        manager.offload_inflight[req] = 1

        del req
        gc.collect()

        self.assertEqual(len(manager.offloaded_state), 0)
        self.assertEqual(len(manager.offload_inflight), 0)


if __name__ == "__main__":
    unittest.main()
