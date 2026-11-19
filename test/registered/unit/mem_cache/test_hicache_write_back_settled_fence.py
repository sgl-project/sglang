"""Write_back eviction copies fenced on the last processed forward.

The PP disaggregated-prefill loop hands HiCacheController the forward-done
event of the micro-batch whose result it processes. A device eviction walk's
victim whose backup reads no other node then waits for that event instead of
the forward in flight; every other copy keeps the forward-stream fence.
"""

import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers.cache_controller import CacheOperation, HiCacheController
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.hicache_write_back_fixtures import (
    BATCH_TOKENS,
    PAGE,
    SEQ_TOKENS,
    InstantWriteBackController,
    batch_seqs,
    hybrid_write_back_cache,
    insert_hybrid_seq,
    insert_seq,
    match_seq,
    write_back_cache,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class _Stream:
    """A D2H stream that logs what it is told to wait for."""

    def __init__(self, log):
        self.log = log

    def wait_event(self, event):
        self.log.append(("wait_event", event))

    def wait_stream(self, stream):
        self.log.append(("wait_stream", stream))


def _completion(log):
    log.append(("submit",))
    return SimpleNamespace(
        start_event=None,
        finish_event=SimpleNamespace(synchronize=lambda: None, query=lambda: True),
        timing_enabled=False,
    )


def _bare_controller():
    """HiCacheController.start_writing over a logging D2H stream."""
    cc = object.__new__(HiCacheController)
    cc.write_queue = []
    cc.ack_write_queue = []
    cc.kv_storage = None
    cc.write_fence_stream = object()
    cc.write_settle_event = None
    cc.mem_pool_host = SimpleNamespace(size_per_token=1)
    cc.log = []
    cc.l2_transfer_engine = SimpleNamespace(
        device_to_host_stream=_Stream(cc.log),
        submit_device_to_host=lambda transfers: _completion(cc.log),
    )
    cc._move_write_operation = lambda op: (op.host_indices, op.device_indices, None)
    cc._l2_transfers = lambda host_indices, device_indices, pool_transfers: []
    return cc


def _op(rows, settled):
    rows = torch.tensor(rows)
    return CacheOperation(rows, rows, int(rows[0]), settled=settled)


class TestStartWritingFence(CustomTestCase):
    def test_a_settled_copy_waits_for_the_settle_event_only(self):
        cc = _bare_controller()
        cc.write_settle_event = settle = object()
        cc.write_queue.append(_op([3, 4], settled=True))
        cc.start_writing()
        self.assertEqual(cc.log, [("wait_event", settle), ("submit",)])
        self.assertEqual(len(cc.ack_write_queue), 1)

    def test_without_an_event_a_settled_copy_waits_for_the_forward(self):
        cc = _bare_controller()
        cc.write_queue.append(_op([3], settled=True))
        cc.start_writing()
        self.assertEqual(cc.log, [("wait_stream", cc.write_fence_stream), ("submit",)])

    def test_an_unsettled_copy_waits_for_the_forward(self):
        cc = _bare_controller()
        cc.write_settle_event = object()
        cc.write_queue.append(_op([3], settled=False))
        cc.start_writing()
        self.assertEqual(cc.log, [("wait_stream", cc.write_fence_stream), ("submit",)])

    def test_a_merged_copy_is_settled_only_if_every_part_is(self):
        self.assertTrue(
            CacheOperation.merge_ops([_op([1], True), _op([2], True)]).settled
        )
        self.assertFalse(
            CacheOperation.merge_ops([_op([1], True), _op([2], False)]).settled
        )
        cc = _bare_controller()
        cc.write_settle_event = object()
        cc.write_queue.extend([_op([3], settled=True), _op([5], settled=False)])
        cc.start_writing()
        self.assertEqual(cc.log, [("wait_stream", cc.write_fence_stream), ("submit",)])

    def test_the_event_only_replaces_the_forward_fence(self):
        cc = _bare_controller()
        cc.write_fence_stream = None
        cc.write_settle_event = object()
        cc.write_queue.append(_op([3], settled=True))
        cc.start_writing()
        self.assertEqual(cc.log, [("submit",)])

    def test_a_reset_drops_the_event(self):
        cc = _bare_controller()
        cc.write_settle_event = object()
        cc.load_queue, cc.ack_load_queue = [], []
        cc.enable_storage = False
        cc.storage_stop_event = threading.Event()
        with mock.patch.object(HiCacheController, "_stop_storage_threads"):
            cc.reset()
        self.assertIsNone(cc.write_settle_event)


class TestVictimCopySettled(CustomTestCase):
    VICTIM = 5

    def _settled(self, event, xfers):
        cache = SimpleNamespace(
            cache_controller=SimpleNamespace(write_settle_event=event)
        )
        return UnifiedRadixCache._victim_copy_settled(
            cache, node_id=self.VICTIM, comp_xfers=xfers
        )

    @staticmethod
    def _xfers(nodes):
        return {"state": [PoolTransfer(name=PoolName.MAMBA, nodes_to_load=nodes)]}

    def test_needs_the_settle_event(self):
        self.assertFalse(self._settled(None, {}))
        self.assertTrue(self._settled(object(), {}))

    def test_component_transfers_of_the_victim_alone(self):
        self.assertTrue(self._settled(object(), self._xfers([self.VICTIM])))
        self.assertTrue(self._settled(object(), self._xfers(None)))

    def test_a_transfer_that_reads_another_node_keeps_the_fence(self):
        self.assertFalse(self._settled(object(), self._xfers([self.VICTIM, 9])))
        self.assertFalse(self._settled(object(), self._xfers([9])))


class _SettleController(InstantWriteBackController):
    """Submits D->H copies through HybridCacheController.start_writing onto a
    D2H stream that logs whether it waited for the forward or an event."""

    start_writing = HybridCacheController.start_writing
    _l2_transfers = HybridCacheController._l2_transfers
    _num_tokens_by_pool = HybridCacheController._num_tokens_by_pool

    def __init__(self, controller):
        super().__init__(controller.mem_pool_host, controller.mem_pool_device_allocator)
        self.log = []
        self.write_fence_stream = object()
        self.l2_transfer_engine = SimpleNamespace(
            batch_direct=False,
            device_to_host_stream=_Stream(self.log),
            submit_device_to_host=lambda transfers: _completion(self.log),
        )

    def _move_write_operation(self, op):
        return op.host_indices, op.device_indices, op.pool_transfers

    def _transfer_num_bytes(self, op):
        return 0

    def fences(self):
        return [entry for entry in self.log if entry[0] != "submit"]


class _SettledFenceWalkSuite:
    tree_core_backend: str

    def _cache_with_one_batch(self):
        cache, allocator, host_pool = write_back_cache(
            self.tree_core_backend,
            device_tokens=4 * BATCH_TOKENS,
            host_tokens=2 * BATCH_TOKENS,
        )
        for seq in batch_seqs(0):
            insert_seq(cache, allocator, seq)
        cc = _SettleController(cache.cache_controller)
        cache.cache_controller = cc
        return cache, cc

    def test_eviction_victims_wait_for_the_settle_event(self):
        cache, cc = self._cache_with_one_batch()
        cc.write_settle_event = settle = object()
        result = cache.evict(EvictParams(num_tokens=BATCH_TOKENS))
        self.assertEqual(result.num_tokens_evicted, BATCH_TOKENS)
        self.assertEqual(cc.fences(), [("wait_event", settle)])
        for seq in batch_seqs(0):
            self.assertEqual(match_seq(cache, seq).host_hit_length, SEQ_TOKENS)
        cache.sanity_check()

    def test_without_an_event_victims_wait_for_the_forward(self):
        cache, cc = self._cache_with_one_batch()
        cache.evict(EvictParams(num_tokens=BATCH_TOKENS))
        self.assertEqual(cc.fences(), [("wait_stream", cc.write_fence_stream)])
        cache.sanity_check()

    def test_an_internal_backup_waits_for_the_forward(self):
        cache, cc = self._cache_with_one_batch()
        cc.write_settle_event = object()
        node = match_seq(cache, batch_seqs(0)[0]).last_device_node
        self.assertTrue(cache.backup_node_for_write_back(node))
        self.assertEqual(cc.fences(), [("wait_stream", cc.write_fence_stream)])

    def test_swa_window_copies_that_read_a_parent_wait_for_the_forward(self):
        cache, allocator, _, _ = hybrid_write_back_cache(
            self.tree_core_backend,
            device_tokens=4 * BATCH_TOKENS,
            swa_device_tokens=4 * BATCH_TOKENS,
            host_tokens=4 * BATCH_TOKENS,
            swa_host_tokens=4 * BATCH_TOKENS,
            sliding_window=3 * PAGE,
        )
        for seq in batch_seqs(0)[:2]:
            insert_hybrid_seq(cache, allocator, seq[:PAGE])
            insert_hybrid_seq(cache, allocator, seq)
        cc = _SettleController(cache.cache_controller)
        cc.write_settle_event = object()
        cache.cache_controller = cc
        result = cache.evict(EvictParams(num_tokens=2 * SEQ_TOKENS))
        self.assertEqual(result.num_tokens_evicted, 2 * SEQ_TOKENS)
        self.assertEqual(cc.fences(), [("wait_stream", cc.write_fence_stream)])
        cache.sanity_check()


class TestSettledFenceWalkRustCore(_SettledFenceWalkSuite, CustomTestCase):
    tree_core_backend = "rust"


class TestSettledFenceWalkPythonCore(_SettledFenceWalkSuite, CustomTestCase):
    tree_core_backend = "python"


class TestPPPrefillHandoff(CustomTestCase):
    @staticmethod
    def _scheduler(write_policy="write_back", hicache=True):
        controller = SimpleNamespace(write_policy=write_policy)
        return SimpleNamespace(
            enable_hierarchical_cache=hicache,
            tree_cache=SimpleNamespace(cache_controller=controller),
        )

    def test_other_caches_take_no_part(self):
        no_controller = self._scheduler()
        no_controller.tree_cache.cache_controller = None
        for scheduler in (
            self._scheduler(write_policy="write_through"),
            self._scheduler(hicache=False),
            no_controller,
        ):
            self.assertIsNone(SchedulerPPMixin._pp_write_settle_controller(scheduler))

    def test_publishes_the_forward_done_event_of_the_processed_micro_batch(self):
        done = object()
        scheduler = SimpleNamespace(
            mb_metadata=[None, SimpleNamespace(forward_done=done)]
        )
        controller = SimpleNamespace(write_settle_event=None)
        SchedulerPPMixin._pp_publish_write_settle_event(
            scheduler, controller=controller, mb_id=1
        )
        self.assertIs(controller.write_settle_event, done)
        # A slot without a launched forward leaves the last event in place.
        SchedulerPPMixin._pp_publish_write_settle_event(
            scheduler, controller=controller, mb_id=0
        )
        self.assertIs(controller.write_settle_event, done)
        SchedulerPPMixin._pp_publish_write_settle_event(
            scheduler, controller=None, mb_id=1
        )


if __name__ == "__main__":
    unittest.main()
