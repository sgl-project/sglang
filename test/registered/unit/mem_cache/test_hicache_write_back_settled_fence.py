"""Write_back eviction copies fenced on the last processed forward.

The PP disaggregated-prefill loop hands HiCacheController the forward-done
event of the micro-batch whose result it processes. A device eviction walk's
victim whose backup reads no other node then waits for that event instead of
the forward in flight; every other copy keeps the forward-stream fence. With
SGLANG_OPT_HICACHE_WB_SETTLED_STATE_VICTIM the walk's internal state backups
count as victims too.
"""

import threading
import unittest
from array import array
from itertools import accumulate
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.managers import scheduler_pp_mixin
from sglang.srt.managers.cache_controller import CacheOperation, HiCacheController
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.pool_host.group import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_exec
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.hicache_write_back_fixtures import (
    BATCH_TOKENS,
    PAGE,
    SEQ_TOKENS,
    InstantWriteBackController,
    SlotOnlyHostPool,
    batch_seqs,
    hybrid_write_back_cache,
    insert_hybrid_seq,
    insert_seq,
    match_seq,
    write_back_cache,
)
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    UnifiedGeometry,
    build_unified_bundle,
    setup_allocator_context,
)

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


class _SettledStateVictimSuite:
    """Write_back caches whose internal node keeps its own Mamba state or SWA window
    under a leaf that a running request holds; a state eviction walk backs it up."""

    tree_core_backend: str

    def _cache_with_an_internal_state(self, *, state_victims, event):
        setup_allocator_context()
        bundle = build_unified_bundle(
            UnifiedGeometry(
                chain="mamba",
                full_layer_ids=(0,),
                state_layer_ids=(1,),
                page_size=1,
                full_tokens=64,
                state_slots=16,
                context_len=64,
            )
        )
        allocator = bundle.token_to_kv_pool_allocator
        # Keep every checkpoint on a path, as --hybrid-cache-checkpoints-level 3 does.
        with get_exec().mamba.override(mamba_path_state_cap=-1):
            cache, _, _ = write_back_cache(
                self.tree_core_backend,
                device_tokens=0,
                host_tokens=64,
                allocator=allocator,
                req_to_token_pool=bundle.req_to_token_pool,
                components=(ComponentType.FULL, ComponentType.MAMBA),
                page_size=1,
            )
        state_host = PoolEntry(
            name=PoolName.MAMBA,
            host_pool=SlotOnlyHostPool(16, page_size=1),
            device_pool=None,
            layer_mapper=lambda layer_id: layer_id,
        )
        cache.host_pool_group = HostPoolGroup(
            [cache.host_pool_group.anchor_entry, state_host]
        )
        cache.cache_controller.mem_pool_host = cache.host_pool_group
        nodes = []
        for tokens in ([1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8]):
            cache.insert(
                InsertParams(
                    key=RadixKey(array("q", tokens)),
                    value=allocator.alloc(len(tokens)),
                    mamba_value=allocator.alloc_state_slots(1).reshape(1),
                )
            )
            nodes.append(match_seq(cache, tokens).last_device_node)
        internal, leaf = nodes
        lock = cache.inc_lock_ref(leaf).to_dec_params()
        self.addCleanup(cache.dec_lock_ref, leaf, lock)
        cc = _SettleController(cache.cache_controller)
        cc.write_settle_state_victims = state_victims
        cc.write_settle_event = event
        cache.cache_controller = cc
        return cache, cc, internal

    def _evict_one_state(self, cache, internal):
        result = cache.evict(EvictParams(mamba_num=1))
        self.assertEqual(result.mamba_num_evicted, 1)
        self.assertTrue(
            cache.tree_core.component_has_host_value_only(internal, ComponentType.MAMBA)
        )
        cache.sanity_check()

    def test_a_walks_state_backup_waits_for_the_settle_event(self):
        settle = object()
        cache, cc, internal = self._cache_with_an_internal_state(
            state_victims=True, event=settle
        )
        self._evict_one_state(cache, internal)
        self.assertEqual(cc.fences(), [("wait_event", settle)])

    def test_unless_the_loop_engaged_it_waits_for_the_forward(self):
        cache, cc, internal = self._cache_with_an_internal_state(
            state_victims=False, event=object()
        )
        self._evict_one_state(cache, internal)
        self.assertEqual(cc.fences(), [("wait_stream", cc.write_fence_stream)])

    def test_without_an_event_it_waits_for_the_forward(self):
        cache, cc, internal = self._cache_with_an_internal_state(
            state_victims=True, event=None
        )
        self._evict_one_state(cache, internal)
        self.assertEqual(cc.fences(), [("wait_stream", cc.write_fence_stream)])

    def test_a_backup_outside_the_walk_waits_for_the_forward(self):
        cache, cc, internal = self._cache_with_an_internal_state(
            state_victims=True, event=object()
        )
        self.assertTrue(cache.backup_node_for_write_back(internal))
        self.assertEqual(cc.fences(), [("wait_stream", cc.write_fence_stream)])

    def _held_window_path(self, *, sliding_window, parts, event):
        """A FULL+SWA path of one node per part; the leaf spans a whole window, so
        its request lock leaves the older windows evictable."""
        cache, allocator, _, _ = hybrid_write_back_cache(
            self.tree_core_backend,
            device_tokens=4 * BATCH_TOKENS,
            swa_device_tokens=4 * BATCH_TOKENS,
            host_tokens=4 * BATCH_TOKENS,
            swa_host_tokens=4 * BATCH_TOKENS,
            sliding_window=sliding_window,
        )
        seq = list(range(1, 1 + sum(parts)))
        nodes = []
        for end in accumulate(parts):
            insert_hybrid_seq(cache, allocator, seq[:end])
            nodes.append(match_seq(cache, seq[:end]).last_device_node)
        lock = cache.inc_lock_ref(nodes[-1]).to_dec_params()
        self.addCleanup(cache.dec_lock_ref, nodes[-1], lock)
        cc = _SettleController(cache.cache_controller)
        cc.write_settle_state_victims = True
        cc.write_settle_event = event
        cache.cache_controller = cc
        return cache, cc, nodes, seq

    def _evict_one_window(self, cache, internal):
        pool = cache.window_sub_pool_of_slot(ComponentType.SWA)
        result = cache.evict(EvictParams(window_num_tokens={pool: PAGE}))
        self.assertEqual(result.window_num_tokens_evicted, {pool: PAGE})
        self.assertTrue(
            cache.tree_core.component_has_host_value_only(internal, ComponentType.SWA)
        )
        cache.sanity_check()

    def test_a_walks_one_node_window_backup_waits_for_the_settle_event(self):
        settle = object()
        cache, cc, nodes, _ = self._held_window_path(
            sliding_window=PAGE, parts=(PAGE, PAGE), event=settle
        )
        self._evict_one_window(cache, nodes[0])
        self.assertEqual(cc.fences(), [("wait_event", settle)])

    def test_a_walks_window_backup_that_reads_a_parent_waits_for_the_forward(self):
        cache, cc, nodes, seq = self._held_window_path(
            sliding_window=3 * PAGE, parts=(PAGE, PAGE, 3 * PAGE), event=object()
        )
        # Refresh the first node, so the walk takes the second, whose window reaches it.
        match_seq(cache, seq[:PAGE])
        self._evict_one_window(cache, nodes[1])
        self.assertEqual(cc.fences(), [("wait_stream", cc.write_fence_stream)])


class TestSettledStateVictimRustCore(_SettledStateVictimSuite, CustomTestCase):
    tree_core_backend = "rust"


class TestSettledStateVictimPythonCore(_SettledStateVictimSuite, CustomTestCase):
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

    @staticmethod
    def _decide_state_victims(war_barrier=True):
        """The loop-entry decision and the INFO lines it logs."""
        scheduler = SimpleNamespace(_war_barrier_enabled=war_barrier)
        controller = SimpleNamespace(write_settle_state_victims=False)
        with mock.patch.object(scheduler_pp_mixin.logger, "info") as info:
            SchedulerPPMixin._pp_settle_state_victims(scheduler, controller=controller)
        lines = [call.args[0] % call.args[1:] for call in info.call_args_list]
        return controller.write_settle_state_victims, lines

    def test_state_victims_follow_the_runtime_master(self):
        master = envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION
        with master.override(True):
            engaged, lines = self._decide_state_victims()
        self.assertTrue(engaged)
        self.assertEqual(len(lines), 1)
        self.assertIn("last processed forward", lines[0])
        # Off by default: no decision and no line.
        with master.override(False):
            self.assertEqual(self._decide_state_victims(), (False, []))

    def test_an_explicit_child_value_wins(self):
        master = envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION
        child = envs.SGLANG_OPT_HICACHE_WB_SETTLED_STATE_VICTIM
        with master.override(True), child.override(False):
            engaged, lines = self._decide_state_victims()
        self.assertFalse(engaged)
        self.assertEqual(len(lines), 1)
        self.assertIn("SGLANG_OPT_HICACHE_WB_SETTLED_STATE_VICTIM=0", lines[0])
        with master.override(False), child.override(True):
            self.assertTrue(self._decide_state_victims()[0])

    def test_state_victims_need_the_war_barrier(self):
        with envs.SGLANG_OPT_HICACHE_WB_SETTLED_STATE_VICTIM.override(True):
            engaged, lines = self._decide_state_victims(war_barrier=False)
        self.assertFalse(engaged)
        self.assertEqual(len(lines), 1)
        self.assertIn("no WAR barrier", lines[0])

    def test_no_state_victims_without_a_settled_controller(self):
        with (
            envs.SGLANG_OPT_HICACHE_WB_SETTLED_STATE_VICTIM.override(True),
            mock.patch.object(scheduler_pp_mixin.logger, "info") as info,
        ):
            SchedulerPPMixin._pp_settle_state_victims(
                SimpleNamespace(), controller=None
            )
        info.assert_not_called()

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
