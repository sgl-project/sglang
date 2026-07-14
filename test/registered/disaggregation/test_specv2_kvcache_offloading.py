"""
Unit tests for _release_finished_req in DecodeKVCacheOffloadManager.

Verifies that over-allocated KV cache slots (from speculative decoding v2)
are correctly freed when a request finishes, preventing GPU memory leaks.

Requires: torch, sglang (run in an environment with sglang installed)
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.srt.disaggregation.kv_events import OffloadedState
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=8, suite="stage-b-test-1-gpu-small-amd")


def _make_mock_req(
    req_pool_idx: int,
    kv_committed_len: int,
    kv_allocated_len: int,
    prefix_indices_len: int = 0,
    rid: int = 0,
):
    """Create a mock Req with the KV cache state needed for testing."""
    req = MagicMock()
    req.rid = rid
    req.req_pool_idx = req_pool_idx
    req.kv_committed_len = kv_committed_len
    req.kv = SimpleNamespace(kv_allocated_len=kv_allocated_len)
    req.prefix_indices = list(range(prefix_indices_len))
    req.effective_kv_committed_len = lambda: req.kv_committed_len
    return req


def _make_manager(
    pool_size: int,
    allocator_page_size: int = 1,
    storage_page_size: int | None = None,
) -> tuple[DecodeKVCacheOffloadManager, list[torch.Tensor]]:
    """Create a DecodeKVCacheOffloadManager with mock pools for testing."""
    if storage_page_size is None:
        storage_page_size = allocator_page_size

    # Build a real req_to_token tensor so indexing works
    req_to_token = torch.arange(pool_size, dtype=torch.int64).unsqueeze(0)

    req_to_token_pool = MagicMock()
    req_to_token_pool.req_to_token = req_to_token

    freed_indices = []

    allocator = MagicMock()
    allocator.page_size = allocator_page_size
    allocator.free = MagicMock(
        side_effect=lambda idx: freed_indices.append(idx.clone())
    )

    tree_cache = MagicMock()
    tree_cache.protected_size_ = 0

    # Bypass __init__ entirely and set attributes directly
    manager = object.__new__(DecodeKVCacheOffloadManager)
    manager.req_to_token_pool = req_to_token_pool
    manager.token_to_kv_pool_allocator = allocator
    manager.storage_page_size = storage_page_size
    manager.allocator_page_size = allocator_page_size
    manager.tree_cache = tree_cache
    manager.offloaded_state = {}
    manager.ongoing_offload = {}
    manager.ongoing_backup = {}
    manager.offload_inflight = {}

    return manager, freed_indices


class _FinishedEvent:
    def synchronize(self):
        pass


class TestDecodeKVCacheOffloadManagerInit(unittest.TestCase):
    def test_npu_guard_precedes_pool_and_controller_side_effects(self):
        """The constructor rejects NPU before accessing any KV pool resource."""
        req_to_token_pool = MagicMock()
        allocator = MagicMock()
        tp_group = MagicMock()
        tree_cache = MagicMock()
        server_args = MagicMock()

        with (
            patch(
                "sglang.srt.disaggregation.decode_kvcache_offload_manager."
                "current_platform.is_npu",
                return_value=True,
            ),
            self.assertRaisesRegex(ValueError, "not supported on NPU"),
        ):
            DecodeKVCacheOffloadManager(
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=allocator,
                tp_group=tp_group,
                tree_cache=tree_cache,
                server_args=server_args,
            )

        allocator.get_kvcache.assert_not_called()
        self.assertEqual(server_args.mock_calls, [])


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
        manager.offloaded_state[req.rid] = OffloadedState(
            prefill_len=0, inc_len=0, last_hash=None
        )

        manager._release_finished_req(req)

        # Only one free call: the committed range [0:20]
        self.assertEqual(len(freed), 1)
        expected = torch.arange(0, 20, dtype=torch.int64)
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
        manager.offloaded_state[req.rid] = OffloadedState(
            prefill_len=0, inc_len=0, last_hash=None
        )

        manager._release_finished_req(req)

        # Two free calls: committed [0:20] and overallocated [20:28]
        self.assertEqual(len(freed), 2)
        expected_committed = torch.arange(0, 20, dtype=torch.int64)
        expected_overalloc = torch.arange(20, 28, dtype=torch.int64)
        self.assertTrue(torch.equal(freed[0], expected_committed))
        self.assertTrue(torch.equal(freed[1], expected_overalloc))
        manager.req_to_token_pool.free.assert_called_once_with(req)

    def test_overallocation_with_page_alignment(self):
        """The committed tail and reservation are split at an allocator page."""
        page_size = 4
        manager, freed = _make_manager(pool_size=32, allocator_page_size=page_size)
        req = _make_mock_req(
            req_pool_idx=0,
            kv_committed_len=10,  # not page-aligned
            kv_allocated_len=28,
        )
        manager.offloaded_state[req.rid] = OffloadedState(
            prefill_len=4, inc_len=0, last_hash=None
        )

        manager._release_finished_req(req)

        # Prefill range [0:4] and committed page range [4:12]
        # Overallocated: start_p = ceil_align(10, 4) = 12, end_p = 28 => [12:28]
        self.assertEqual(len(freed), 3)
        expected_prefill = torch.arange(0, 4, dtype=torch.int64)
        expected_committed = torch.arange(4, 12, dtype=torch.int64)
        expected_overalloc = torch.arange(12, 28, dtype=torch.int64)
        self.assertTrue(torch.equal(freed[0], expected_prefill))
        self.assertTrue(torch.equal(freed[1], expected_committed))
        self.assertTrue(torch.equal(freed[2], expected_overalloc))

    def test_overallocation_page_aligned_noop(self):
        """No reservation free is emitted when rounded committed equals allocated."""
        page_size = 4
        manager, freed = _make_manager(pool_size=32, allocator_page_size=page_size)
        req = _make_mock_req(
            req_pool_idx=0,
            kv_committed_len=10,  # ceil_align(10, 4) = 12
            kv_allocated_len=12,  # same as aligned start
        )
        manager.offloaded_state[req.rid] = OffloadedState(
            prefill_len=4, inc_len=0, last_hash=None
        )

        manager._release_finished_req(req)

        # Prefill [0:4] and rounded committed [4:12], with no overalloc range.
        self.assertEqual(len(freed), 2)
        expected_prefill = torch.arange(0, 4, dtype=torch.int64)
        expected_committed = torch.arange(4, 12, dtype=torch.int64)
        self.assertTrue(torch.equal(freed[0], expected_prefill))
        self.assertTrue(torch.equal(freed[1], expected_committed))

    def test_aligned_committed_boundary_is_released_once(self):
        """An aligned committed boundary separates disjoint live and reserved pages."""
        manager, freed = _make_manager(pool_size=32, allocator_page_size=4)
        req = _make_mock_req(
            req_pool_idx=0,
            kv_committed_len=12,
            kv_allocated_len=20,
        )
        manager.offloaded_state[req.rid] = OffloadedState(
            prefill_len=4, inc_len=0, last_hash=None
        )

        manager._release_finished_req(req)

        self.assertEqual(len(freed), 3)
        self.assertTrue(torch.equal(freed[0], torch.arange(0, 4)))
        self.assertTrue(torch.equal(freed[1], torch.arange(4, 12)))
        self.assertTrue(torch.equal(freed[2], torch.arange(12, 20)))

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
        manager.offloaded_state[req.rid] = OffloadedState(
            prefill_len=0, inc_len=0, last_hash=None
        )

        manager._release_finished_req(req)

        self.assertEqual(manager.tree_cache.protected_size_, 5)

    def test_release_finished_req_frees_prefill_when_state_present(self):
        """A finished request releases its prefill pages at the final free site."""
        manager, freed = _make_manager(pool_size=32)
        rid = "req-prefill-present"
        req = _make_mock_req(
            req_pool_idx=0,
            kv_committed_len=20,
            kv_allocated_len=20,
            rid=rid,
        )
        manager.offloaded_state[rid] = OffloadedState(
            prefill_len=8, inc_len=0, last_hash=None
        )

        manager._release_finished_req(req)

        # Two frees in order: prefill [0:8] then committed [8:20].
        self.assertEqual(len(freed), 2)
        expected_prefill = torch.arange(0, 8, dtype=torch.int64)
        expected_committed = torch.arange(8, 20, dtype=torch.int64)
        self.assertTrue(torch.equal(freed[0], expected_prefill))
        self.assertTrue(torch.equal(freed[1], expected_committed))
        # State entry is removed at the end of _release_finished_req.
        self.assertNotIn(rid, manager.offloaded_state)

    def test_release_finished_req_skips_prefill_free_when_prefill_len_zero(self):
        """A zero prefill boundary emits no separate prefill free."""
        manager, freed = _make_manager(pool_size=32)
        rid = "req-prefill-zero"
        req = _make_mock_req(
            req_pool_idx=0,
            kv_committed_len=10,
            kv_allocated_len=10,
            rid=rid,
        )
        manager.offloaded_state[rid] = OffloadedState(
            prefill_len=0, inc_len=0, last_hash=None
        )

        manager._release_finished_req(req)

        # Only the committed range [0:10] is freed.
        self.assertEqual(len(freed), 1)
        expected_committed = torch.arange(0, 10, dtype=torch.int64)
        self.assertTrue(torch.equal(freed[0], expected_committed))

    def test_finalize_release_creates_state_so_prefill_is_freed(self):
        """Finalize materializes the missing state before releasing all pages."""
        page_size = 4
        manager, freed = _make_manager(pool_size=32, allocator_page_size=page_size)
        rid = "req-finalize-no-state"
        req = _make_mock_req(
            req_pool_idx=0,
            kv_committed_len=13,
            kv_allocated_len=16,
            rid=rid,
        )
        # 12 input tokens => prefill_len = 12 // 4 * 4 = 12
        req.origin_input_ids = list(range(12))

        manager.finalize_release_on_finish(req)

        # finalize creates state, then _release_finished_req frees:
        #   prefill [0:12] then the rounded committed page [12:16].
        self.assertEqual(len(freed), 2)
        expected_prefill = torch.arange(0, 12, dtype=torch.int64)
        expected_committed = torch.arange(12, 16, dtype=torch.int64)
        self.assertTrue(torch.equal(freed[0], expected_prefill))
        self.assertTrue(torch.equal(freed[1], expected_committed))
        # State is deleted by _release_finished_req on the way out.
        self.assertNotIn(rid, manager.offloaded_state)

    def test_dcp_style_page_mismatch_uses_allocator_boundaries(self):
        """Device release uses allocator pages while hashing uses storage pages."""
        manager, freed = _make_manager(
            pool_size=32,
            allocator_page_size=8,
            storage_page_size=4,
        )
        manager.cache_controller = MagicMock()
        manager.cache_controller.get_hash_str.side_effect = ["hash-0", "hash-1"]
        req = _make_mock_req(
            req_pool_idx=0,
            kv_committed_len=18,
            kv_allocated_len=24,
            rid="dcp-style",
        )
        req.origin_input_ids = list(range(11))

        manager.finalize_release_on_finish(req)
        hashes = manager._compute_prefix_hash(list(range(8)))

        self.assertEqual(hashes, ["hash-0", "hash-1"])
        self.assertEqual(len(freed), 2)
        self.assertTrue(torch.equal(freed[0], torch.arange(0, 8)))
        self.assertTrue(torch.equal(freed[1], torch.arange(8, 24)))

    def test_unfinished_offload_ack_does_not_free_incremental_slots(self):
        """An unfinished request keeps device ownership after its write ACK."""
        manager, freed = _make_manager(pool_size=32)
        req = _make_mock_req(
            req_pool_idx=0, kv_committed_len=20, kv_allocated_len=20, rid=1
        )
        req.finished.return_value = False
        manager.offloaded_state[req.rid] = OffloadedState(
            prefill_len=4, inc_len=4, last_hash=None
        )
        manager.offload_inflight[req.rid] = 1
        manager.ongoing_offload[7] = (
            req,
            torch.arange(4, 8, dtype=torch.int64),
            [10, 11, 12, 13],
            0.0,
            4,
            8,
        )
        manager.cache_controller = MagicMock()
        manager.cache_controller.ack_write_queue = [(None, _FinishedEvent(), [7])]
        manager._trigger_backup = MagicMock(return_value="last_hash")

        manager._check_offload_progress(1)

        self.assertEqual(freed, [])
        manager.req_to_token_pool.free.assert_not_called()
        self.assertNotIn(req.rid, manager.offload_inflight)

    def test_missing_state_ack_fails_without_releasing_device_slots(self):
        """An ACK without lifecycle state fails instead of guessing a release start."""
        manager, freed = _make_manager(pool_size=32)
        req = _make_mock_req(
            req_pool_idx=0, kv_committed_len=20, kv_allocated_len=20, rid=6
        )
        req.finished.return_value = True
        manager.offload_inflight[req.rid] = 1
        manager.ongoing_offload[10] = (
            req,
            torch.arange(4, 8, dtype=torch.int64),
            [10, 11, 12, 13],
            0.0,
            4,
            8,
        )
        manager.cache_controller = MagicMock()
        manager.cache_controller.ack_write_queue = [(None, _FinishedEvent(), [10])]
        manager._trigger_backup = MagicMock()

        with self.assertRaisesRegex(RuntimeError, "Missing offload state"):
            manager._check_offload_progress(1)

        self.assertEqual(freed, [])
        manager._trigger_backup.assert_not_called()
        manager.req_to_token_pool.free.assert_not_called()

    def test_offload_kv_cache_tracks_inflight_write_until_ack(self):
        """A queued device-to-host write remains tracked until its ACK arrives."""
        manager, freed = _make_manager(pool_size=32, allocator_page_size=4)
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
        self.assertEqual(manager.offload_inflight[req.rid], 1)
        self.assertEqual(manager.offloaded_state[req.rid].inc_len, 4)
        manager.cache_controller.write.assert_called_once()

        manager.cache_controller.ack_write_queue = [(None, _FinishedEvent(), [1])]
        manager._trigger_backup = MagicMock(return_value="last_hash")

        manager._check_offload_progress(1)

        self.assertEqual(freed, [])
        self.assertNotIn(req.rid, manager.offload_inflight)

    def test_offload_prefill_boundary_uses_allocator_page_size(self):
        """Prefill ownership is floored by the allocator page, not the storage page."""
        manager, _ = _make_manager(
            pool_size=32,
            allocator_page_size=8,
            storage_page_size=4,
        )
        manager.cache_controller = MagicMock()
        manager.cache_controller.get_hash_str.side_effect = ["hash-0", "hash-1"]
        manager.cache_controller.write.return_value = torch.arange(4)
        manager.decode_host_mem_pool = MagicMock()
        manager.request_counter = 0
        manager.offload_stride = 4
        req = _make_mock_req(
            req_pool_idx=0,
            kv_committed_len=16,
            kv_allocated_len=16,
            rid="allocator-page-prefill",
        )
        req.origin_input_ids = list(range(10))
        req.output_ids = [10, 11, 12, 13, 14]

        self.assertTrue(manager.offload_kv_cache(req))

        self.assertEqual(manager.offloaded_state[req.rid].prefill_len, 8)
        written_indices = manager.cache_controller.write.call_args.kwargs[
            "device_indices"
        ]
        self.assertTrue(torch.equal(written_indices, torch.arange(8, 12)))

    def test_finalize_release_defers_while_offload_is_in_flight(self):
        """Finalize preserves device pages while an offload write is in flight."""
        manager, freed = _make_manager(pool_size=32)
        req = _make_mock_req(
            req_pool_idx=0, kv_committed_len=20, kv_allocated_len=20, rid=2
        )
        manager.offloaded_state[req.rid] = OffloadedState(
            prefill_len=4, inc_len=8, last_hash=None
        )
        manager.offload_inflight[req.rid] = 1

        manager.finalize_release_on_finish(req)

        self.assertEqual(freed, [])
        manager.req_to_token_pool.free.assert_not_called()
        self.assertIn(req.rid, manager.offloaded_state)

    def test_finished_offload_ack_waits_for_other_inflight_writes(self):
        """A finished request waits until every in-flight write is acknowledged."""
        manager, freed = _make_manager(pool_size=32)
        req = _make_mock_req(
            req_pool_idx=0, kv_committed_len=20, kv_allocated_len=20, rid=3
        )
        req.finished.return_value = True
        manager.offloaded_state[req.rid] = OffloadedState(
            prefill_len=4, inc_len=8, last_hash=None
        )
        manager.offload_inflight[req.rid] = 2
        manager.ongoing_offload[8] = (
            req,
            torch.arange(4, 8, dtype=torch.int64),
            [10, 11, 12, 13],
            0.0,
            4,
            8,
        )
        manager.cache_controller = MagicMock()
        manager.cache_controller.ack_write_queue = [(None, _FinishedEvent(), [8])]
        manager._trigger_backup = MagicMock(return_value="last_hash")

        manager._check_offload_progress(1)

        self.assertEqual(freed, [])
        manager.req_to_token_pool.free.assert_not_called()
        self.assertEqual(manager.offload_inflight[req.rid], 1)

    def test_finished_request_releases_all_committed_slots_after_last_offload_ack(
        self,
    ):
        """The final write ACK triggers the request's only device release."""
        manager, freed = _make_manager(pool_size=32)
        req = _make_mock_req(
            req_pool_idx=0, kv_committed_len=20, kv_allocated_len=20, rid=4
        )
        req.finished.return_value = True
        manager.offloaded_state[req.rid] = OffloadedState(
            prefill_len=4, inc_len=8, last_hash=None
        )
        manager.offload_inflight[req.rid] = 1
        manager.ongoing_offload[9] = (
            req,
            torch.arange(8, 12, dtype=torch.int64),
            [14, 15, 16, 17],
            0.0,
            8,
            12,
        )
        manager.cache_controller = MagicMock()
        manager.cache_controller.ack_write_queue = [(None, _FinishedEvent(), [9])]
        manager._trigger_backup = MagicMock(return_value="last_hash")

        manager._check_offload_progress(1)

        self.assertEqual(len(freed), 2)
        self.assertTrue(torch.equal(freed[0], torch.arange(0, 4, dtype=torch.int64)))
        self.assertTrue(torch.equal(freed[1], torch.arange(4, 20, dtype=torch.int64)))
        manager.req_to_token_pool.free.assert_called_once_with(req)
        self.assertNotIn(req.rid, manager.offloaded_state)
        self.assertNotIn(req.rid, manager.offload_inflight)


if __name__ == "__main__":
    unittest.main()
