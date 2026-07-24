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

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
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

    def test_backup_executor_skips_already_backed_nodes(self):
        """Overlapping BackupKV chains must not back a node twice: a second
        backup would allocate a second host copy and leak the first."""
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        self._insert(cache, allocator, req_to_token_pool, [1, 2, 3, 4])
        node = next(iter(cache.root_node.children.values()))

        self.assertGreater(_write_backup(cache, node, write_back=True), 0)
        cache.writing_check(write_back=True)
        self.assertTrue(node.backuped)
        host_avail = cache.cache_controller.mem_pool_host.available_size()

        # Re-applying an overlapping chain is a no-op skip, not a re-backup.
        self.assertEqual(_write_backup(cache, node, write_back=True), 0)
        cache.writing_check(write_back=True)
        self.assertEqual(
            cache.cache_controller.mem_pool_host.available_size(), host_avail
        )

    def test_shallower_crossing_backs_up_above_backuped_middle(self):
        """A shallower crossing node above a backuped middle must back up in
        the same insert as the deeper crossing (its own walk barrier)."""
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        self._insert(cache, allocator, req_to_token_pool, [1, 2, 3, 4])
        top = next(iter(cache.root_node.children.values()))

        # A storage-prefetch completion host-inserts a backuped node below the
        # still-unbacked top, legitimately breaking backup continuity.
        host_indices = cache.cache_controller.mem_pool_host.alloc(8)
        host_result = cache.tree_core.insert_host(
            cache.root_node.id,
            RadixKey(array("q", list(range(1, 9)))),
            host_indices,
            [f"h{i}" for i in range(8)],
        )
        cache.cache_controller.mem_pool_host.free(
            host_indices[: host_result.prefix_len]
        )
        middle = next(iter(top.children.values()))
        self.assertTrue(middle.backuped)
        self.assertFalse(top.backuped)

        # The device insert unevicts the middle and adds the deep leaf.
        self._insert(cache, allocator, req_to_token_pool, list(range(1, 13)))
        deep = next(iter(middle.children.values()))

        cache.write_through_threshold = min(top.hit_count, deep.hit_count) + 1
        self._insert(cache, allocator, req_to_token_pool, list(range(1, 17)))
        cache.writing_check(write_back=True)
        self.assertTrue(top.backuped)
        self.assertTrue(deep.backuped)

    def test_evict_drains_collected_frees_when_walk_raises(self):
        """A device-eviction walk that raises mid-way must still free the
        already-collected slots via the finally drain."""
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        self._insert(cache, allocator, req_to_token_pool, [1, 2, 3, 4])
        self._insert(cache, allocator, req_to_token_pool, [10, 11, 12, 13])
        available = allocator.available_size()

        real_next = cache.tree_core.evict_device_next_node
        calls = []

        def raise_on_second(*args, **kwargs):
            calls.append(args)
            if len(calls) == 2:
                raise RuntimeError("boom")
            return real_next(*args, **kwargs)

        with mock.patch.object(
            cache.tree_core, "evict_device_next_node", side_effect=raise_on_second
        ):
            with self.assertRaises(RuntimeError):
                cache.evict(EvictParams(num_tokens=8))
        self.assertEqual(allocator.available_size(), available + 4)

    def test_evict_host_drains_collected_frees_when_walk_raises(self):
        """A host-eviction walk that raises mid-way must still free the
        already-evicted host slots via the finally drain."""
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        for start in (1, 100):
            self._insert(
                cache, allocator, req_to_token_pool, list(range(start, start + 4))
            )
        for child in list(cache.root_node.children.values()):
            self.assertGreater(_write_backup(cache, child, write_back=True), 0)
        cache.writing_check(write_back=True)
        cache.evict(EvictParams(num_tokens=8))

        host_avail = cache.cache_controller.mem_pool_host.available_size()
        real_evict = cache.tree_core._evict_host_leaf
        calls = []

        def raise_on_second(*args, **kwargs):
            calls.append(args)
            if len(calls) == 2:
                raise RuntimeError("boom")
            return real_evict(*args, **kwargs)

        with mock.patch.object(
            cache.tree_core, "_evict_host_leaf", side_effect=raise_on_second
        ):
            with self.assertRaises(RuntimeError):
                cache.evict_host(8)
        self.assertEqual(
            cache.cache_controller.mem_pool_host.available_size(), host_avail + 4
        )


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

    def test_dec_swa_lock_only_drains_frees_when_walk_raises(self):
        """A SWA early-release that raises mid-walk must still free the
        already-evicted SWA slots via the finally drain."""
        sw = self.cfg.sliding_window_size
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq = list(range(1, 2 * sw + 1))
        key = RadixKey(array("q", seq))
        cache.insert(InsertParams(key=key, value=self._alloc(allocator, len(seq))))
        m = cache.match_prefix(MatchPrefixParams(key=key))
        lock = cache.inc_lock_ref(m.last_device_node)
        # Release FULL first so the SWA early-release is the last lock standing.
        cache.dec_lock_ref(m.last_device_node, lock.to_dec_params(), skip_swa=True)

        swa_avail = allocator.swa_attn_allocator.available_size()
        real_evict = cache.tree_core._evict_component_and_detach_lru

        def evict_then_raise(*args, **kwargs):
            real_evict(*args, **kwargs)
            raise RuntimeError("boom")

        with mock.patch.object(
            cache.tree_core,
            "_evict_component_and_detach_lru",
            side_effect=evict_then_raise,
        ):
            with self.assertRaises(RuntimeError):
                cache.dec_swa_lock_only(m.last_device_node, lock.swa_uuid_for_lock)
        self.assertEqual(allocator.swa_attn_allocator.available_size(), swa_avail + sw)


if __name__ == "__main__":
    unittest.main()
