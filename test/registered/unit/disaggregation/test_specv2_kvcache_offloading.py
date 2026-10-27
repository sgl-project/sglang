"""
Unit tests for _release_finished_req in DecodeKVCacheOffloadManager.

Verifies that over-allocated KV cache slots (from speculative decoding v2)
are correctly freed when a request finishes, preventing GPU memory leaks.

Requires: torch, sglang (run in an environment with sglang installed)
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock
from weakref import WeakKeyDictionary as WeakKeyDict

import torch

from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.srt.disaggregation.kv_events import OffloadedState
from sglang.srt.managers.cache_controller import HiCacheAck
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.utils import get_hash_str, get_storage_hash_str
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=19, suite="base-a-test-cpu")


def _make_mock_req(
    req_pool_idx: int,
    kv_committed_len: int,
    kv_allocated_len: int,
    prefix_len: int = 0,
    rid: int = 0,
    origin_len: int = 0,
):
    """Create a mock Req with the KV cache state needed for testing."""
    req = MagicMock()
    req.rid = rid
    req.extra_key = None  # base traffic: storage hashes chain from tokens alone
    req.cache_salt = None
    req.origin_input_ids = list(range(origin_len))
    req.kv = ReqKvInfo(
        req_pool_idx=req_pool_idx,
        kv_committed_len=kv_committed_len,
        kv_allocated_len=kv_allocated_len,
    )
    req.prefix_len = prefix_len
    req.owned_kv_len = lambda: req.kv.kv_committed_len
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
    manager.kv_cache = MagicMock()
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

    def test_decode_offload_hash_chain_matches_prefill(self):
        """Decode pages must keep the prefill namespace across offload chunks."""
        manager, _ = _make_manager(pool_size=8, page_size=2)
        manager.cache_controller = MagicMock(get_hash_str=get_hash_str)
        tokens = [1, 2, 3, 4, 5, 6]
        for extra_key, cache_salt in [
            (None, None),
            ("lora-a", None),
            (None, "tenant-a"),
            ("lora-a", "tenant-a"),
        ]:
            with self.subTest(extra_key=extra_key, cache_salt=cache_salt):
                namespace = dict(extra_key=extra_key, cache_salt=cache_salt)
                req = SimpleNamespace(**namespace)
                prefix = manager._compute_prefix_hash(req, tokens[:4])
                tail = manager._compute_prefix_hash(req, tokens[4:], prefix[-1])
                self.assertEqual(
                    prefix + tail,
                    get_storage_hash_str(RadixKey(tokens, **namespace), page_size=2),
                )

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
            [],
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
            [],
        )
        manager.cache_controller = MagicMock()
        manager.cache_controller.ack_write_queue = [
            HiCacheAck(None, _FinishedEvent(), [9])
        ]
        manager._trigger_backup = MagicMock(return_value="last_hash")

        manager._check_offload_progress(1)

        self.assertEqual(len(freed), 1)
        self.assertTrue(torch.equal(freed[0], torch.arange(0, 20, dtype=torch.int64)))
        manager.req_to_token_pool.free.assert_called_once_with(req)
        self.assertNotIn(req, manager.offloaded_state)
        self.assertNotIn(req, manager.offload_inflight)


class TestSamplingMaskAbortOffload(CustomTestCase):
    def test_abort_waits_for_existing_offload_before_reusing_slots(self):
        """An abort must not recycle slots while a previous D2H copy reads them."""
        for inflight in (False, True):
            with self.subTest(inflight=inflight):
                manager, freed = _make_manager(pool_size=32)
                req = _make_mock_req(0, 20, 20)
                req.multimodal_inputs = None
                req.finished.return_value = True
                manager.req_to_token_pool.free.side_effect = lambda req: setattr(
                    req.kv, "req_pool_idx", None
                )
                processor = SimpleNamespace(decode_offload_manager=manager)
                if inflight:
                    manager.offload_inflight[req] = 1
                    manager.ongoing_offload[1] = (req, torch.arange(4), [1], 0.0, [])
                    manager.cache_controller = MagicMock()
                    manager.cache_controller.ack_write_queue = [
                        HiCacheAck(None, _FinishedEvent(), [1])
                    ]
                    manager._trigger_backup = MagicMock(return_value="hash")

                with get_context().override_server_args(
                    disaggregation_decode_enable_offload_kvcache=True,
                    enable_hisparse=False,
                ):
                    SchedulerBatchResultProcessor._handle_sampling_mask_abort(
                        processor, req
                    )

                if inflight:
                    self.assertEqual(freed, [])
                    self.assertEqual(req.kv.req_pool_idx, 0)
                    manager._check_offload_progress(1)
                self.assertEqual(len(freed), 1)
                self.assertTrue(torch.equal(freed[0], torch.arange(20)))
                self.assertIsNone(req.kv.req_pool_idx)
                manager.finalize_release_on_finish(req)
                self.assertEqual(len(freed), 1)


if __name__ == "__main__":
    unittest.main()
