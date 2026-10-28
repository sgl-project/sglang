"""
Unit tests for the decode KV offload release (RequestRetirer.retire_offloaded).

Verifies that over-allocated KV cache slots (from speculative decoding v2)
are correctly freed when a request finishes, preventing GPU memory leaks.

Requires: torch, sglang (run in an environment with sglang installed)
"""

import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.srt.disaggregation.kv_events import OffloadedState
from sglang.srt.disaggregation.req_state import DisaggReqState
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
from sglang.test.scheduler_request_lists import make_request_retirer
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


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
    req.disagg = DisaggReqState()
    return req


def _in_flight(manager, req, ack_id, tokens=(10, 11, 12, 13)):
    """Record a submitted copy of req's KV that has not landed."""
    if req.disagg.offload is None:
        req.disagg.offload = OffloadedState()
    req.disagg.offload.ack_ids.append(ack_id)
    manager.ongoing_offload[ack_id] = (
        req.rid,
        torch.arange(4, 8, dtype=torch.int64),
        list(tokens),
        ["hash"],
        0.0,
        [],
    )


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
    manager.ongoing_offload = {}
    manager.ongoing_backup = {}

    return manager, freed_indices


def _retirer(manager):
    return make_request_retirer(
        kv=SimpleNamespace(
            tree_cache=manager.tree_cache,
            req_to_token_pool=manager.req_to_token_pool,
        ),
        decode_offload_manager=manager,
    )


def _retire_offloaded(manager, req):
    _retirer(manager).retire_offloaded(req)


class _FinishedEvent:
    def synchronize(self):
        pass


class TestReleaseFinishedReq(unittest.TestCase):
    """Tests for RequestRetirer.retire_offloaded overallocation cleanup."""

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

    def test_backup_hashes_chain_across_copies(self):
        """Each backup carries its chunk's hashes, chained from the prefill
        pages through the earlier chunks, also with copies in flight."""
        manager, _ = _make_manager(pool_size=32, page_size=2)
        manager.cache_controller = MagicMock(
            get_hash_str=lambda tokens, prior: f"{prior}>{tokens}"
        )
        manager.cache_controller.write.return_value = torch.arange(2)
        manager.decode_host_mem_pool = MagicMock()
        manager.request_counter = 0
        manager.offload_stride = 2

        req = _make_mock_req(
            req_pool_idx=0, kv_committed_len=20, kv_allocated_len=20, rid="r"
        )
        req.origin_input_ids = [0, 1]
        req.output_ids = [2, 3, 4]
        self.assertTrue(manager.offload_kv_cache(req))
        req.output_ids += [5, 6]
        self.assertTrue(manager.offload_kv_cache(req))

        manager.cache_controller.ack_write_queue = [
            HiCacheAck(None, _FinishedEvent(), [1]),
            HiCacheAck(None, _FinishedEvent(), [2]),
        ]
        manager._check_offload_progress(2, [])

        backed_up = [
            call.kwargs["hash_value"]
            for call in manager.cache_controller.write_storage.call_args_list
        ]
        expected = manager._compute_prefix_hash(req, [0, 1, 2, 3, 4, 5])
        self.assertEqual(backed_up, [expected[1:2], expected[2:3]])

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
        self.assertTrue(manager.has_inflight_offload(old_req))
        self.assertTrue(manager.has_inflight_offload(new_req))

        manager.cache_controller.ack_write_queue = [
            HiCacheAck(None, _FinishedEvent(), [1])
        ]
        manager._trigger_backup = MagicMock()
        # Both releasing under one rid, the new one first: the ack id decides.
        released = manager._check_offload_progress(1, [new_req, old_req])
        for released_req in released:
            _retire_offloaded(manager, released_req)

        self.assertEqual(released, [old_req])
        self.assertIsNone(old_req.disagg.offload)
        self.assertEqual(new_req.disagg.offload.inc_len, 4)
        self.assertTrue(manager.has_inflight_offload(new_req))
        self.assertIn(2, manager.ongoing_offload)

    def test_finished_offload_ack_waits_for_other_inflight_writes(self):
        manager, freed = _make_manager(pool_size=32)
        req = _make_mock_req(
            req_pool_idx=0, kv_committed_len=20, kv_allocated_len=20, rid=3
        )
        req.finished.return_value = True
        _in_flight(manager, req, 8)
        _in_flight(manager, req, 9)
        manager.cache_controller = MagicMock()
        manager.cache_controller.ack_write_queue = [
            HiCacheAck(None, _FinishedEvent(), [8])
        ]
        manager._trigger_backup = MagicMock()

        self.assertEqual(manager._check_offload_progress(1, [req]), [])

        self.assertEqual(freed, [])
        manager.req_to_token_pool.free.assert_not_called()
        self.assertTrue(manager.has_inflight_offload(req))

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
        _in_flight(manager, req, 9, tokens=(14, 15, 16, 17))
        manager.cache_controller = MagicMock()
        manager.cache_controller.ack_write_queue = [
            HiCacheAck(None, _FinishedEvent(), [9])
        ]
        manager._trigger_backup = MagicMock()

        self.assertEqual(manager._check_offload_progress(1, [req]), [req])
        _retire_offloaded(manager, req)

        self.assertEqual(len(freed), 1)
        self.assertTrue(torch.equal(freed[0], torch.arange(0, 20, dtype=torch.int64)))
        manager.req_to_token_pool.free.assert_called_once_with(req)
        self.assertIsNone(req.disagg.offload)
        self.assertFalse(manager.has_inflight_offload(req))

    def test_copy_in_flight_does_not_pin_the_request(self):
        manager, _ = _make_manager(pool_size=32)
        req = _make_mock_req(
            req_pool_idx=0, kv_committed_len=20, kv_allocated_len=20, rid=6
        )
        _in_flight(manager, req, 1)
        ref = weakref.ref(req)

        del req
        gc.collect()

        self.assertIsNone(ref())
        self.assertIn(1, manager.ongoing_offload)


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
                processor = SimpleNamespace(
                    decode_offload_manager=manager,
                    request_retirer=_retirer(manager),
                )
                if inflight:
                    _in_flight(manager, req, 1)
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
                    self.assertEqual(manager._check_offload_progress(1, [req]), [req])
                    _retire_offloaded(manager, req)
                self.assertEqual(len(freed), 1)
                self.assertTrue(torch.equal(freed[0], torch.arange(20)))
                self.assertIsNone(req.kv.req_pool_idx)
                _retire_offloaded(manager, req)
                self.assertEqual(len(freed), 1)


if __name__ == "__main__":
    unittest.main()
