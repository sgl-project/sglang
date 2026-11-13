"""CPU contracts for C128 storage keys, staging ownership and L3 publication."""

import unittest
from collections import defaultdict
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.hardware_backend.npu.dsv4.c128_sidecar_component import (
    C128SidecarComponent,
)
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.unified_cache.cache_action import FreeComponentHostSlot
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components import CacheTransferPhase, EvictLayer
from sglang.srt.mem_cache.unified_cache.unified_tree_core import (
    UnifiedTreeCore,
    UnifiedTreeNode,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache, _OngoingPrefetch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

C128 = ComponentType.C128
POOL = PoolName.DEEPSEEK_V4_C128


class TestC128Storage(CustomTestCase):
    def setUp(self):
        self.component = C128SidecarComponent.__new__(C128SidecarComponent)
        self.component.cache = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(
                c128_attn_allocator=SimpleNamespace(page_size=16)
            ),
            evict_host=MagicMock(),
        )
        self.lru = MagicMock()
        self.lru.in_list.return_value = False
        self.component.tree_core = SimpleNamespace(
            page_size=128,
            host_lru_lists={C128: self.lru},
            _update_evictable_leaf_sets=MagicMock(),
            node_by_id=MagicMock(),
        )
        self.component._c128_kv_pool_host = MagicMock()

    @staticmethod
    def _node(start=0, end=32, slots=32):
        root = SimpleNamespace(parent=None)
        parent = SimpleNamespace(parent=root, key=range(start * 128)) if start else root
        return SimpleNamespace(
            id=7,
            parent=parent,
            key=range((end - start) * 128),
            hash_value=[f"H{i}" for i in range(start + 1, end + 1)],
            component_data={
                C128: SimpleNamespace(value=None, host_value=torch.arange(slots))
            },
        )

    def test_backup_uses_absolute_block_endpoints(self):
        for start, end, slots, expected in (
            (0, 32, 32, ["H16", "H32"]),
            (8, 32, 32, ["H16", "H32"]),
            (24, 48, 32, ["H32", "H48"]),
            (8, 31, 16, ["H16"]),
        ):
            with self.subTest(start=start, end=end):
                node = self._node(start, end, slots)
                transfers = self.component.build_hicache_transfers(
                    node, CacheTransferPhase.BACKUP_STORAGE
                )
                self.assertEqual(len(transfers), 1)
                transfer = transfers[0]
                self.assertEqual(transfer.keys, expected)
                self.assertEqual(transfer.host_indices.tolist(), list(range(slots)))
                self.assertEqual(transfer.hit_policy, PoolHitPolicy.BLOCK_GROUPS)
                self.assertEqual((transfer.block_pages, transfer.group_pages), (16, 1))
                self.assertIsNone(transfer.indices_from_pool)

    def test_backup_declines_missing_endpoints_or_host_slots(self):
        for node in (self._node(0, 15, 16), self._node(0, 32, 0)):
            self.assertIsNone(
                self.component.build_hicache_transfers(
                    node, CacheTransferPhase.BACKUP_STORAGE
                )
            )
        node = self._node()
        node.component_data[C128].host_value = None
        self.assertIsNone(
            self.component.build_hicache_transfers(
                node, CacheTransferPhase.BACKUP_STORAGE
            )
        )

    def test_prefetch_plan_contains_only_complete_pages(self):
        for tokens, blocks in ((0, 0), (2047, 0), (2048, 1), (4095, 1), (4096, 2)):
            with self.subTest(tokens=tokens):
                result = self.component.prepare_prefetch(7, prefetch_tokens=tokens)
                self.assertEqual(result.staging_tokens, blocks * 2048)
                transfers = self.component.build_hicache_transfers(
                    self._node(),
                    CacheTransferPhase.PREFETCH,
                    prefetch_tokens=tokens,
                    staging_tokens=result.staging_tokens,
                )
                if blocks:
                    self.assertEqual(len(transfers), 1)
                    self.assertEqual(len(transfers[0].keys), blocks)
                    self.assertIsNone(transfers[0].host_indices)
                    self.assertIsNone(transfers[0].indices_from_pool)
                else:
                    self.assertIsNone(transfers)
        self.component._c128_kv_pool_host = None
        self.assertEqual(
            self.component.prepare_prefetch(7, prefetch_tokens=4096).staging_tokens, 0
        )

    def test_staging_retries_after_eviction(self):
        indices = torch.arange(32)
        pool = self.component._c128_kv_pool_host
        pool.alloc.side_effect = [None, indices]
        self.assertIs(self.component.alloc_prefetch_staging(32), indices)
        self.assertEqual([c.args for c in pool.alloc.call_args_list], [(32,), (32,)])
        self.component.cache.evict_host.assert_called_once_with(32, C128)

    def test_commit_publishes_on_inserted_leaf(self):
        target = self._node()
        target.component_data[C128].host_value = None
        self.component.tree_core.node_by_id.return_value = target
        indices = torch.arange(64, 96)
        actions = []
        self.component._commit_prefetch(
            self._node(),
            [PoolTransfer(name=POOL, host_indices=indices)],
            cache_actions=actions,
            insert_result=SimpleNamespace(inserted_host_node=target.id),
            # IO counts C128 objects, whereas the existence query counts FULL pages.
            pool_storage_result=PoolTransferResult(32, {POOL: 2}),
        )
        self.component.tree_core.node_by_id.assert_called_once_with(target.id)
        published = target.component_data[C128].host_value
        self.assertTrue(torch.equal(published, indices))
        self.assertNotEqual(published.data_ptr(), indices.data_ptr())
        self.assertEqual(actions, [])
        self.lru.insert_mru.assert_called_once_with(target)
        updates = self.component.tree_core._update_evictable_leaf_sets.call_args_list
        self.assertEqual(len(updates), 2)
        self.assertIs(updates[0].args[0], target)
        self.assertIs(updates[1].args[0], target.parent)

    def test_commit_releases_unpublishable_slots_once(self):
        for reason in ("partial", "no_result", "no_target", "duplicate"):
            with self.subTest(reason=reason):
                target = self._node()
                original = target.component_data[C128].host_value
                if reason != "duplicate":
                    target.component_data[C128].host_value = None
                self.component.tree_core.node_by_id.return_value = target
                indices = torch.arange(64, 96)
                actions = []
                self.component._commit_prefetch(
                    self._node(),
                    [PoolTransfer(name=POOL, host_indices=indices)],
                    cache_actions=actions,
                    insert_result=(
                        None
                        if reason == "no_target"
                        else SimpleNamespace(inserted_host_node=target.id)
                    ),
                    pool_storage_result=(
                        None
                        if reason == "no_result"
                        else PoolTransferResult(
                            32, {POOL: 1 if reason == "partial" else 2}
                        )
                    ),
                )
                self.assertEqual(len(actions), 1)
                self.assertIsInstance(actions[0], FreeComponentHostSlot)
                self.assertEqual(actions[0].component_type, C128)
                self.assertIs(actions[0].host_indices[0], indices)
                if reason == "duplicate":
                    self.assertIs(target.component_data[C128].host_value, original)
                else:
                    self.assertIsNone(target.component_data[C128].host_value)
        self.lru.insert_mru.assert_not_called()
        self.component.tree_core._update_evictable_leaf_sets.assert_not_called()


class TestC128HostEviction(CustomTestCase):
    def setUp(self):
        self.component = C128SidecarComponent.__new__(C128SidecarComponent)
        self.component._c128_kv_pool_host = MagicMock()
        self.root = UnifiedTreeNode((ComponentType.FULL, C128))
        self.tree = SimpleNamespace(
            root_node=self.root,
            evictable_host_leaves=set(),
            eviction_strategy=SimpleNamespace(
                get_priority=lambda n: n.last_access_time
            ),
        )
        self.tree._is_host_leaf = lambda n: UnifiedTreeCore._is_host_leaf(self.tree, n)
        self.tree._evict_host_leaf = MagicMock(side_effect=self._evict_leaf)
        self.component.tree_core = self.tree
        self.tracker = defaultdict(int)
        self.device_frees = defaultdict(list)
        self.host_frees = defaultdict(list)

    def _node(self, slots=16, parent=None, full_tokens=128):
        node = UnifiedTreeNode((ComponentType.FULL, C128))
        node.parent = parent if parent is not None else self.root
        node.parent.children[node.id] = node
        node.component_data[ComponentType.FULL].host_value = torch.arange(full_tokens)
        if slots:
            node.component_data[C128].host_value = torch.arange(slots)
        self.tree.evictable_host_leaves.discard(node.parent)
        self.tree.evictable_host_leaves.add(node)
        return node

    def _evict_leaf(self, node, tracker, device_frees, host_frees, *, cache_actions):
        # Model the tree's atomic leaf removal, using real C128 slot collection.
        self.assertTrue(self.tree._is_host_leaf(node))
        full = node.component_data[ComponentType.FULL]
        tracker[ComponentType.FULL] += full.host_value.numel()
        full.host_value = None
        self.component.evict_component(
            node, device_frees, host_frees, cache_actions, target=EvictLayer.HOST
        )
        self.tree.evictable_host_leaves.remove(node)
        node.parent.children.pop(node.id)
        if self.tree._is_host_leaf(node.parent):
            self.tree.evictable_host_leaves.add(node.parent)

    def _evict(self, slots):
        self.component.drive_host_eviction(
            slots, self.tracker, self.device_frees, self.host_frees, []
        )

    def test_stops_on_c128_slots_not_full_token_count(self):
        first = self._node(full_tokens=4096)
        second = self._node(full_tokens=128)
        untouched = self._node()
        self._evict(32)
        self.assertIsNone(first.component_data[C128].host_value)
        self.assertIsNone(second.component_data[C128].host_value)
        self.assertIsNotNone(untouched.component_data[C128].host_value)
        self.assertEqual(self.tracker[C128], 32)
        self.assertEqual(self.tracker[ComponentType.FULL], 4224)
        self.assertEqual(self.tree._evict_host_leaf.call_count, 2)

    def test_preserves_locked_nodes_and_prefers_non_session_leaves(self):
        locked = self._node()
        locked.component_data[C128].host_lock_ref = 1
        self.tree.evictable_host_leaves.discard(locked)
        session = self._node()
        session.component_data[C128].session_ref = 1
        victim = self._node()
        self._evict(16)
        self.assertIsNotNone(locked.component_data[C128].host_value)
        self.assertIsNotNone(session.component_data[C128].host_value)
        self.assertIsNone(victim.component_data[C128].host_value)
        self.assertEqual(self.tracker[C128], 16)
        self.tree._evict_host_leaf.assert_called_once()

    def test_removes_empty_tail_before_c128_parent(self):
        parent = self._node()
        tail = self._node(slots=0, parent=parent)
        self._evict(16)
        victims = [call.args[0] for call in self.tree._evict_host_leaf.call_args_list]
        self.assertEqual(victims, [tail, parent])
        self.assertEqual(self.tracker[C128], 16)
        self.assertEqual(len(self.host_frees[C128]), 1)

    def test_no_evictable_leaf_returns_without_reclaiming(self):
        locked = self._node()
        locked.component_data[ComponentType.FULL].lock_ref = 1
        self.tree.evictable_host_leaves.discard(locked)
        self._evict(16)
        self.tree._evict_host_leaf.assert_not_called()
        self.assertEqual(self.tracker[C128], 0)
        self.assertIsNotNone(locked.component_data[C128].host_value)


class TestC128PrefetchCleanup(CustomTestCase):
    def setUp(self):
        self.anchor = torch.arange(4096)
        self.c128_slots = torch.arange(64, 96)
        self.swa_slots = torch.arange(512, 640)
        self.c128 = PoolTransfer(
            name=POOL,
            keys=["H16", "H32"],
            hit_policy=PoolHitPolicy.BLOCK_GROUPS,
            block_pages=16,
            group_pages=1,
        )
        self.swa = PoolTransfer(name=PoolName.SWA, keys=["H32"])
        # KV-derived C4 rows alias the anchor allocation, not an independent pool.
        self.c4 = PoolTransfer(
            name=PoolName.DEEPSEEK_V4_C4,
            indices_from_pool=PoolName.KV,
            host_indices=self.anchor,
        )
        self.request = CacheRequestHandle("c128-cleanup", 0)
        self.operation = SimpleNamespace(
            handle=self.request,
            hash_value=[f"H{i}" for i in range(1, 33)],
            storage_hit_count=4096,
            storage_start=0,
            completed_tokens=4096,
            sidecar_hash_values=None,
            deadline_expired=False,
            pool_transfers=[self.c128, self.swa, self.c4],
            pool_transfers_done=True,
            is_terminated=lambda: False,
        )
        cc = HybridCacheController.__new__(HybridCacheController)
        cc.mem_pool_host = SimpleNamespace(
            page_size=128,
            entry_map={
                name: SimpleNamespace(
                    host_pool=SimpleNamespace(page_size=size),
                    is_primary_index_anchor=False,
                )
                for name, size in (
                    (POOL, 16),
                    (PoolName.SWA, 128),
                    (PoolName.DEEPSEEK_V4_C4, 128),
                )
            },
            free=MagicMock(),
        )
        cc.host_mem_release_queue = Queue()
        cc.extra_host_mem_release_queues = {
            name: Queue() for name in cc.mem_pool_host.entry_map
        }
        cc.prefetch_hit_queue = Queue()
        cc.ack_prefetch_queue = Queue()
        cc.ack_backup_queue = Queue()
        cc.prefetch_buffer = Queue()
        cc.prefetch_tokens_occupied = 4096
        cc.allocate_storage_hit = MagicMock(return_value=(self.anchor, 4096))
        cc._finish_op_trace = MagicMock()
        self.cc = cc

        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.cache_controller = cc
        cache.tree_core = MagicMock(page_size=128)
        cache.host_memory_mode = "cache"
        cache.buffer_pipeline = None
        cache.prefetch_threshold = 128
        cache.enable_storage_metrics = False
        cache.components = {
            C128: SimpleNamespace(
                alloc_prefetch_staging=MagicMock(return_value=self.c128_slots)
            ),
            ComponentType.SWA: SimpleNamespace(
                alloc_prefetch_staging=MagicMock(return_value=None)
            ),
        }
        self.lock_params = object()
        cache.ongoing_prefetch = {
            self.request: _OngoingPrefetch(
                anchor_node_id=7,
                prefetch_key=range(4096),
                host_indices=None,
                operation=self.operation,
                anchor_lock_params=self.lock_params,
                comp_xfers={C128: [self.c128], ComponentType.SWA: [self.swa, self.c4]},
            )
        }
        cache.prefetch_loaded_tokens_by_reqid = {}
        cache.prefetch_loaded_storage_start_by_reqid = {self.request: 0}
        cache.storage_existence_cache = MagicMock()
        cache.storage_prefetch_retries = MagicMock()
        # Keep allocation, retirement and release real; stub unrelated accounting.
        for name in (
            "dec_host_lock_ref",
            "evict_host",
            "_belief_policy",
            "_finish_storage_prefetch",
            "_invalidate_absent_from_hit_query",
            "_record_storage_prefetch_hit",
            "_account_prefetch_outcome",
            "_log_storage_prefetch_deferred",
        ):
            setattr(cache, name, MagicMock())
        self.cache = cache

    def _drain(self, storage_hits=0):
        self.cache._drain_storage_control_queues_impl(
            n_storage_hit=storage_hits,
            n_ack_prefetch=0,
            n_backup=0,
            n_release=None,
            extra_release_counts={
                name: None for name in self.cc.extra_host_mem_release_queues
            },
            log_metrics=False,
        )

    def _assert_released_once(self, expected):
        calls = list(self.cc.mem_pool_host.free.call_args_list)
        self.assertEqual(len(calls), len(expected))
        for call in calls:
            pool = call.kwargs.get("pool", PoolName.KV)
            self.assertIn(pool, expected)
            self.assertTrue(torch.equal(call.args[0], expected.pop(pool)))
        self.assertEqual(expected, {})
        self.assertNotIn(self.request, self.cache.ongoing_prefetch)
        self.assertEqual(self.cc.prefetch_tokens_occupied, 0)
        self.cache.dec_host_lock_ref.assert_called_once_with(7, self.lock_params)
        self.cache.tree_core.insert_host.assert_not_called()
        self.assertTrue(self.cc.prefetch_buffer.empty())
        # A later cancellation/poll must not free the same slots a second time.
        self.cache.revoke_pending_prefetch(self.request)
        self._drain()
        self.assertEqual(self.cc.mem_pool_host.free.call_count, len(calls))
        self.cache.dec_host_lock_ref.assert_called_once_with(7, self.lock_params)

    def test_aux_allocation_failure_rolls_back_anchor_and_c128(self):
        # C128 succeeds, then SWA fails. Exercise the hit-queue caller as well
        # as auxiliary rollback, so clearing ownership cannot leak or double free.
        self.cc.prefetch_hit_queue.put(self.operation)
        self._drain(storage_hits=1)
        self.cache.components[C128].alloc_prefetch_staging.assert_called_once_with(32)
        self.cache.components[
            ComponentType.SWA
        ].alloc_prefetch_staging.assert_called_once_with(128)
        self.assertIsNone(self.c128.host_indices)
        self.assertIsNone(self.swa.host_indices)
        self._assert_released_once({PoolName.KV: self.anchor, POOL: self.c128_slots})

    def test_partial_c128_read_discards_prefetch_and_releases_all_slots(self):
        self.c128.host_indices = self.c128_slots
        self.swa.host_indices = self.swa_slots
        self.cache.ongoing_prefetch[self.request] = self.cache.ongoing_prefetch[
            self.request
        ]._replace(host_indices=self.anchor)
        # KV and SWA were read completely, but only one of two C128 objects was
        # read. IO results count objects, not the FULL pages covered by them.
        self.operation.pool_storage_result = PoolTransferResult(
            32, {POOL: 1, PoolName.SWA: 1}
        )
        self.cache._handle_prefetch_result(self.operation)
        self.assertEqual(self.cache.prefetch_loaded_tokens_by_reqid[self.request], 0)
        self.assertNotIn(
            self.request, self.cache.prefetch_loaded_storage_start_by_reqid
        )
        self.cache._finish_storage_prefetch.assert_called_once_with(
            self.request, fulfilled_tokens=0, reason="storage_transfer"
        )
        self._drain()
        self._assert_released_once(
            {
                PoolName.KV: self.anchor,
                POOL: self.c128_slots,
                PoolName.SWA: self.swa_slots,
            }
        )


if __name__ == "__main__":
    unittest.main()
