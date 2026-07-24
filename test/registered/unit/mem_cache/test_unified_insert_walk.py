"""CUDA unit tests for the resumable insert walk, CI-active while the gated
unified radix cache unittest module is deferred on trunk."""

import unittest
from array import array
from unittest import mock

import torch
from test_unified_radix_cache_unittest import (
    CacheConfig,
    UnifiedRadixCacheSuite,
    _write_backup,
    build_fixture,
)

from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.cache_action import FreeDeviceKV
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=6, stage="base-b", runner_config="1-gpu-small")


class _InsertWalkSuite(CustomTestCase):
    """Fixture helpers borrowed from the gated suite, without its tests."""

    _rid = 0
    _make_req = UnifiedRadixCacheSuite._make_req
    _alloc = UnifiedRadixCacheSuite._alloc
    _insert = UnifiedRadixCacheSuite._insert
    _init_hicache = UnifiedRadixCacheSuite._init_hicache
    _build_hicache_fixture = UnifiedRadixCacheSuite._build_hicache_fixture


@unittest.skipUnless(torch.cuda.is_available(), "cache fixtures need CUDA")
class TestResumableInsertWalk(_InsertWalkSuite):
    cfg = CacheConfig()

    def test_walk_backup_can_host_evict_on_path_h_leaf(self):
        """A crossing node's backup runs at its walk step, so its host eviction
        can still take an H-leaf deeper on the inserted path."""
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()

        self._insert(cache, allocator, req_to_token_pool, [1, 2, 3, 4])
        top = next(iter(cache.root_node.children.values()))
        self._insert(cache, allocator, req_to_token_pool, list(range(1, 9)))
        h_leaf = next(iter(top.children.values()))
        self.assertGreater(_write_backup(cache, h_leaf, write_back=True), 0)
        cache.writing_check(write_back=True)
        cache.evict(EvictParams(num_tokens=4))
        self.assertTrue(h_leaf.evicted)

        # Fill the host pool below len(top) free, keeping the on-path H-leaf
        # the oldest host entry and pinning the unbacked path root.
        cache.inc_lock_ref(top.id)
        host_pool = cache.cache_controller.mem_pool_host
        start = 1000
        while host_pool.available_size() >= len(top.key):
            count = min(host_pool.available_size() - len(top.key) + 1, 250)
            tokens = list(range(start, start + count))
            start += 1000
            self._insert(cache, allocator, req_to_token_pool, tokens)
            filler = None
            for child in cache.root_node.children.values():
                if child is not top and not child.evicted:
                    filler = child
            self.assertIsNotNone(filler)
            self.assertGreater(_write_backup(cache, filler, write_back=True), 0)
            cache.writing_check(write_back=True)
            cache.evict(EvictParams(num_tokens=count))
            self.assertTrue(filler.evicted)
        cache.dec_lock_ref(top.id)

        # The crossing backup evicts exactly the on-path H-leaf, then the
        # remaining suffix is recreated as a fresh leaf.
        cache.write_through_threshold = top.hit_count + 1
        self._insert(cache, allocator, req_to_token_pool, list(range(1, 13)))
        cache.writing_check(write_back=True)

        self.assertTrue(top.backuped)
        self.assertNotIn(h_leaf, top.children.values())
        self.assertIsNone(h_leaf.component_data[ComponentType.FULL].host_value)
        (child_key_len,) = {len(c.key) for c in top.children.values()}
        self.assertEqual(child_key_len, 8)
        cache.sanity_check()

    def test_insert_aborts_continuation_when_action_apply_fails(self):
        """An exception while executing a barrier's actions aborts the suspended
        insert instead of leaking its continuation."""
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        cache.write_through_threshold = 2
        self._insert(cache, allocator, req_to_token_pool, [1, 2])

        with mock.patch.object(
            cache, "_execute_and_commit_kv_backup", side_effect=RuntimeError("boom")
        ):
            with self.assertRaises(RuntimeError):
                self._insert(cache, allocator, req_to_token_pool, [1, 2, 3, 4])
        self.assertIsNone(cache.tree_core._ongoing_insert_walk_state)

        # The tree stays usable and the crossing re-fires on the next walk.
        self._insert(cache, allocator, req_to_token_pool, [1, 2, 3, 4])
        cache.writing_check(write_back=True)
        ancestor = next(iter(cache.root_node.children.values()))
        self.assertTrue(ancestor.backuped)
        cache.sanity_check()

    def test_begin_insert_rejects_concurrent_walk(self):
        """Insert walks are single-flight: beginning a second insert while one
        is suspended at a barrier is re-entrancy and must fail fast."""
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        cache.write_through_threshold = 2
        self._insert(cache, allocator, req_to_token_pool, [1, 2])

        # Suspend an insert at its crossing barrier by pumping it directly.
        params = InsertParams(
            key=RadixKey(array("q", [1, 2, 3, 4])), value=self._alloc(allocator, 4)
        )
        step = cache.tree_core.begin_insert(params)
        self.assertIsNone(step.result)
        with self.assertRaises(AssertionError):
            cache.tree_core.begin_insert(params)
        cache.tree_core.end_insert()

    def test_insert_abort_drains_pending_deferred_frees(self):
        """A mid-insert failure after a deferred dup-free accumulated must still
        return those slots to the allocator via the end_insert drain."""
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        self._insert(cache, allocator, req_to_token_pool, [1, 2, 3, 4])

        # The overlap walk defers a 4-slot dup-free; the commit hook then raises.
        available = allocator.available_size()
        full_comp = cache.components[ComponentType.FULL]
        with mock.patch.object(
            full_comp, "commit_insert_component_data", side_effect=RuntimeError("boom")
        ):
            with self.assertRaises(RuntimeError):
                self._insert(cache, allocator, req_to_token_pool, list(range(1, 9)))

        # 8 alloc'd for the insert, 4 dup slots drained back on abort.
        self.assertEqual(allocator.available_size(), available - 4)
        self.assertIsNone(cache.tree_core._ongoing_insert_walk_state)

    def test_deferrable_actions_ride_final_step_without_suspension(self):
        """A walk whose only actions are deferrable frees completes in a single
        step, the batched frees riding the final step's actions."""
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        self._insert(cache, allocator, req_to_token_pool, [1, 2, 3, 4])

        # The overlap re-insert defers a dup-free; no barrier action fires.
        params = InsertParams(
            key=RadixKey(array("q", list(range(1, 9)))),
            value=self._alloc(allocator, 8),
        )
        step = cache.tree_core.begin_insert(params)
        self.assertIsNotNone(step.result)
        self.assertTrue(any(isinstance(a, FreeDeviceKV) for a in step.actions))
        cache._apply_cache_actions(step.actions)
        cache.tree_core.end_insert()
        cache.sanity_check()


@unittest.skipUnless(torch.cuda.is_available(), "cache fixtures need CUDA")
class TestResumableInsertWalkSWA(_InsertWalkSuite):
    cfg = CacheConfig(
        components=(ComponentType.FULL, ComponentType.SWA), sliding_window_size=8
    )

    def test_swa_recovery_keeps_recovered_node_below_window_nodes(self):
        """A tombstone recovered during the walk lands below the in-window path
        in the SWA LRU, so eviction takes the recovered span first."""

        sw = self.cfg.sliding_window_size
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq = list(range(1, 2 * sw + 1))
        key = RadixKey(array("q", seq))
        cache.insert(
            InsertParams(
                key=key, value=self._alloc(allocator, len(seq)), swa_evicted_seqlen=sw
            )
        )
        prefix_node = next(iter(cache.root_node.children.values()))
        window_node = next(iter(prefix_node.children.values()))
        self.assertIsNone(prefix_node.component_data[ComponentType.SWA].value)

        # Re-inserting fully in-window recovers the prefix span's SWA data.
        cache.insert(
            InsertParams(
                key=key, value=self._alloc(allocator, len(seq)), swa_evicted_seqlen=0
            )
        )
        self.assertIsNotNone(prefix_node.component_data[ComponentType.SWA].value)

        # SWA eviction takes the recovered span and keeps the window leaf.
        cache.evict(EvictParams(num_tokens=0, swa_num_tokens=sw))
        self.assertIsNone(prefix_node.component_data[ComponentType.SWA].value)
        self.assertIsNotNone(window_node.component_data[ComponentType.SWA].value)
        self.assertIsNotNone(window_node.component_data[ComponentType.FULL].value)
        cache.sanity_check()


if __name__ == "__main__":
    unittest.main()
