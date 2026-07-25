"""CUDA unit tests for the resumable insert walk and the returned-values drain
contract, CI-active while the gated unified radix cache unittest module is
deferred on trunk."""

import sys
import unittest
from array import array
from collections import defaultdict
from unittest import mock

import torch
from test_unified_radix_cache_unittest import (
    CacheConfig,
    UnifiedRadixCacheSuite,
    _write_backup,
    build_fixture,
)

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.cache_action import FreeDeviceKV
from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
    DecSwaLockOnlyResult,
    DemoteResult,
    DriveHostEvictionResult,
    DropSubtreeNoHostResult,
    EvictDeviceLeafResult,
    EvictDeviceNextNodeResult,
)
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
    _make_seq = UnifiedRadixCacheSuite._make_seq
    _skip_unsupported_hicache_test = (
        UnifiedRadixCacheSuite._skip_unsupported_hicache_test
    )


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

    def test_match_split_relocation_survives_finalizer_failure(self):
        """A match-walk split's pending write-through relocation applies before
        the finalizers, so a finalizer failure cannot strand the stale record."""
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        cache.write_through_threshold = 1
        # The leaf backs up on insert; its ack stays pending (no writing_check).
        self._insert(cache, allocator, req_to_token_pool, [1, 2, 3, 4])
        self.assertTrue(cache.ongoing_write_through)

        full_comp = cache.components[ComponentType.FULL]
        with mock.patch.object(
            full_comp,
            "finalize_match_result_in_cache",
            side_effect=RuntimeError("boom"),
        ):
            with self.assertRaises(RuntimeError):
                cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", [1, 2]))))

        # The relocation reached the pending record: the ack clears the
        # pending marker on both split halves, not just the stale node.
        cache.writing_check(write_back=True)
        parent = next(iter(cache.root_node.children.values()))
        (child,) = parent.children.values()
        self.assertIsNone(parent.write_through_pending_id)
        self.assertIsNone(child.write_through_pending_id)
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

    def test_dec_swa_lock_only_early_release_keeps_full_lock(self):
        """The scheduler's early SWA release (decode past the window) drops
        only the SWA lock; the Full path lock stays held until dec_lock_ref."""
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq = self._make_seq(1, self.cfg.sliding_window_size + 4)
        self._insert(cache, allocator, req_to_token_pool, seq)
        m = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq))))
        node = cache.resolve_node_handle(m.last_device_node)
        swa_cd = node.component_data[ComponentType.SWA]
        full_cd = node.component_data[ComponentType.FULL]

        lock_result = cache.inc_lock_ref(node.id)
        self.assertGreaterEqual(swa_cd.lock_ref, 1)
        cache.dec_swa_lock_only(node.id, lock_result.swa_uuid_for_lock)
        self.assertEqual(swa_cd.lock_ref, 0)
        self.assertGreaterEqual(full_cd.lock_ref, 1)

        cache.dec_lock_ref(node.id, DecLockRefParams(swa_uuid_for_lock=None))
        cache.sanity_check()


@unittest.skipUnless(torch.cuda.is_available(), "cache fixtures need CUDA")
class TestResumableInsertWalkWriteBack(_InsertWalkSuite):
    cfg = CacheConfig()

    def test_hicache_write_back_evict_drops_unbacked_leaf_when_host_full(self):
        """Write-back eviction will keep freeing device KV when the host pool
        is exhausted and host eviction cannot free space to prevent OOM."""
        if self._skip_unsupported_hicache_test():
            return
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        self._init_hicache(cache, write_policy="write_back")

        # Two-node chain: the drop must cascade parent-ward across eviction
        # iterations, not just delete a single leaf.
        seq_parent = self._make_seq(1, 2)
        self._insert(cache, allocator, req_to_token_pool, seq_parent)
        seq = seq_parent + self._make_seq(1000, 1)
        self._insert(cache, allocator, req_to_token_pool, seq)

        # Exhaust the KV host pool. The tree has no host leaves, so
        # evict_host cannot free anything and every D->H backup fails.
        host_pool = cache.cache_controller.mem_pool_host
        self.assertIsNotNone(host_pool.alloc(host_pool.available_size()))
        self.assertEqual(host_pool.available_size(), 0)

        result = cache.evict(EvictParams(num_tokens=len(seq)))
        self.assertGreaterEqual(result.num_tokens_evicted, len(seq))

        # The chain is gone entirely: no device hit, no host hit.
        m = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq))))
        self.assertEqual(len(m.device_indices), 0)
        self.assertEqual(m.host_hit_length, 0)
        cache.sanity_check()

    def test_hicache_write_back_drop_respects_pins_then_frees_subtree(self):
        """The host-pressure drop fallback must decline while any node in the
        unbacked subtree is host-pinned, then reclaim the whole subtree --
        including a demoted child's host backup -- once unpinned."""
        if self._skip_unsupported_hicache_test():
            return
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        self._init_hicache(cache, write_policy="write_back")
        host_pool = cache.cache_controller.mem_pool_host
        baseline_host = host_pool.available_size()

        parent_seq = self._make_seq(1, 2)
        self._insert(cache, allocator, req_to_token_pool, parent_seq)
        child_seq = parent_seq + self._make_seq(1000, 1)
        self._insert(cache, allocator, req_to_token_pool, child_seq)
        m = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", child_seq))))
        child = cache.resolve_node_handle(m.last_device_node)

        # Evict only the child leaf -> real backup + demote, leaving it
        # host-only under a still-unbacked device parent (write-back backs up
        # single nodes leaf-first, so this is a normal intermediate state).
        result = cache.evict(EvictParams(num_tokens=len(child.key)))
        self.assertGreaterEqual(result.num_tokens_evicted, len(child.key))
        self.assertTrue(child.evicted and child.backuped)
        parent = child.parent
        self.assertFalse(parent.backuped)
        self.assertGreater(baseline_host - host_pool.available_size(), 0)

        # From here every backup fails (controller.write returns None), so
        # each evict() attempts the drop fallback on the parent.
        with mock.patch.object(cache.cache_controller, "write", return_value=None):
            # Pinned subtree root: drop declines, chain stays intact.
            cache.inc_host_lock_ref(parent.id)
            result = cache.evict(EvictParams(num_tokens=len(parent_seq)))
            self.assertEqual(result.num_tokens_evicted, 0)
            cache.dec_host_lock_ref(parent.id)

            # Pinned host-only descendant: drop declines as well.
            cache.inc_host_lock_ref(child.id)
            result = cache.evict(EvictParams(num_tokens=len(parent_seq)))
            self.assertEqual(result.num_tokens_evicted, 0)
            m = cache.match_prefix(
                MatchPrefixParams(key=RadixKey(array("q", parent_seq)))
            )
            self.assertEqual(len(m.device_indices), len(parent_seq))
            cache.dec_host_lock_ref(child.id)

            # Unpinned: the subtree drops and the child's host slots return.
            result = cache.evict(EvictParams(num_tokens=len(parent_seq)))
        self.assertGreaterEqual(result.num_tokens_evicted, len(parent_seq))
        self.assertEqual(host_pool.available_size(), baseline_host)
        m = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", child_seq))))
        self.assertEqual(len(m.device_indices), 0)
        self.assertEqual(m.host_hit_length, 0)
        cache.sanity_check()

    def test_drop_fallback_frees_host_for_later_backups_same_round(self):
        """Host slots reclaimed by the drop fallback itself (an interior
        host-only descendant) must be reusable by later write-back backups in
        the same eviction round (pre-split freed them inline)."""
        if self._skip_unsupported_hicache_test():
            return
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        self._init_hicache(cache, write_policy="write_back")
        host_pool = cache.cache_controller.mem_pool_host

        # Chain 1: unbacked parent -> host-only child -> host-only grandchild.
        p1 = self._make_seq(1, 2)
        self._insert(cache, allocator, req_to_token_pool, p1)
        c1 = p1 + self._make_seq(1000, 2)
        self._insert(cache, allocator, req_to_token_pool, c1)
        g1 = c1 + self._make_seq(2000, 2)
        self._insert(cache, allocator, req_to_token_pool, g1)
        cache.evict(EvictParams(num_tokens=4))
        m = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", p1))))
        parent = cache.resolve_node_handle(m.last_device_node)
        (child,) = parent.children.values()
        (grandchild,) = child.children.values()
        self.assertTrue(child.evicted and child.backuped)
        self.assertTrue(grandchild.evicted and grandchild.backuped)

        # Chain 2: a younger unbacked leaf needing 4 host slots; only 2 can
        # come from evict_host (the grandchild leaf) — the other 2 exist only
        # if the drop fallback's child slots are drained within the round.
        p2 = self._make_seq(5000, 4)
        self._insert(cache, allocator, req_to_token_pool, p2)
        leaf2 = None
        for node in cache.root_node.children.values():
            if list(node.key.token_ids[: len(p2)]) == list(p2):
                leaf2 = node
        self.assertIsNotNone(leaf2)
        self.assertIsNotNone(host_pool.alloc(host_pool.available_size()))

        real_write = cache.cache_controller.write
        calls = []

        def fail_first(*args, **kwargs):
            calls.append(args)
            if len(calls) == 1:
                return None
            return real_write(*args, **kwargs)

        with mock.patch.object(cache.cache_controller, "write", side_effect=fail_first):
            result = cache.evict(EvictParams(num_tokens=len(p1) + len(p2)))
        self.assertGreaterEqual(result.num_tokens_evicted, len(p1) + len(p2))
        self.assertTrue(leaf2.evicted and leaf2.backuped)
        cache.sanity_check()


@unittest.skipUnless(torch.cuda.is_available(), "cache fixtures need CUDA")
class TestReturnedValuesDrain(_InsertWalkSuite):
    """Drain contract of the returned-values eviction API: every tree-core step
    result is drained exactly once, in per-component insertion order."""

    cfg = CacheConfig()

    def test_each_eviction_step_result_is_drained(self):
        """Every step wrapper hands its result's frees to _free_values and
        passes the payload through; a wrapper that forgets strands pool slots."""
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        self._insert(cache, allocator, req_to_token_pool, [1, 2, 3, 4])
        node = next(iter(cache.root_node.children.values()))
        tracker = {ct: 0 for ct in cache.tree_components}
        sentinel = torch.tensor([7], dtype=torch.int64)

        def make(result_cls, **fields):
            result = result_cls(**fields)
            result.device_frees[ComponentType.FULL].append(sentinel)
            result.host_frees[ComponentType.FULL].append(sentinel)
            return result

        cases = [
            (
                "evict_device_next_node",
                lambda: make(EvictDeviceNextNodeResult, node_id=node.id),
                lambda: cache._evict_device_next_node(ComponentType.FULL, tracker),
                node.id,
            ),
            (
                "evict_device_leaf",
                lambda: make(EvictDeviceLeafResult),
                lambda: cache._evict_device_leaf(node.id, tracker),
                None,
            ),
            (
                "demote",
                lambda: make(DemoteResult),
                lambda: cache._demote(node.id, tracker),
                None,
            ),
            (
                "drop_subtree_no_host",
                lambda: make(DropSubtreeNoHostResult, is_dropped=True),
                lambda: cache._drop_subtree_no_host(node.id, tracker),
                True,
            ),
            (
                "drive_host_eviction",
                lambda: make(DriveHostEvictionResult),
                lambda: cache.evict_host(4),
                0,
            ),
            (
                "dec_swa_lock_only",
                lambda: make(DecSwaLockOnlyResult),
                lambda: cache.dec_swa_lock_only(node.id),
                None,
            ),
        ]
        for name, make_result, call, expected in cases:
            with self.subTest(step=name):
                drained = []

                def record(device_frees, host_frees):
                    drained.append((dict(device_frees), dict(host_frees)))
                    device_frees.clear()
                    host_frees.clear()

                with mock.patch.object(
                    cache.tree_core, name, return_value=make_result()
                ), mock.patch.object(cache, "_free_values", side_effect=record):
                    returned = call()
                self.assertEqual(returned, expected)
                ((device_frees, host_frees),) = drained
                self.assertIs(device_frees[ComponentType.FULL][0], sentinel)
                self.assertIs(host_frees[ComponentType.FULL][0], sentinel)

    def test_free_values_frees_in_component_insertion_order(self):
        """Device frees apply before host frees, each in per-component
        insertion order — the allocator free-list order the pre-split inline
        frees produced."""
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        order = [ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA]
        device_frees = defaultdict(list)
        host_frees = defaultdict(list)
        for ct in order:
            device_frees[ct].append(torch.tensor([1]))
            host_frees[ct].append(torch.tensor([2]))

        freed = []
        fake_components = {
            ct: mock.MagicMock(
                free_host_values=mock.MagicMock(
                    side_effect=lambda values, ct=ct: freed.append(("host", ct))
                )
            )
            for ct in order
        }
        with mock.patch.object(
            cache,
            "_apply_cache_action",
            side_effect=lambda action: freed.append(("device", action.component_type)),
        ), mock.patch.dict(cache.components, fake_components):
            cache._free_values(device_frees, host_frees)

        self.assertEqual(
            freed, [("device", ct) for ct in order] + [("host", ct) for ct in order]
        )
        self.assertFalse(device_frees)
        self.assertFalse(host_frees)

    def test_free_values_mid_drain_failure_cannot_replay_freed_entries(self):
        """A device free that raises must leave only un-attempted entries in
        the dict (no replay of successes) while host frees still drain."""
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        order = [ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA]
        device_frees = defaultdict(list)
        host_frees = defaultdict(list)
        for ct in order:
            device_frees[ct].append(torch.tensor([1]))
        host_frees[ComponentType.FULL].append(torch.tensor([2]))

        def boom_on_swa(action):
            if action.component_type is ComponentType.SWA:
                raise RuntimeError("boom")

        host_mock = mock.MagicMock()
        with mock.patch.object(
            cache, "_apply_cache_action", side_effect=boom_on_swa
        ), mock.patch.dict(cache.components, {ComponentType.FULL: host_mock}):
            with self.assertRaises(RuntimeError):
                cache._free_values(device_frees, host_frees)

        self.assertEqual(list(device_frees), [ComponentType.MAMBA])
        self.assertFalse(host_frees)
        host_mock.free_host_values.assert_called_once()

    def test_undrained_result_trips_the_del_assert(self):
        """Dropping a result without draining fires the __del__ tripwire (the
        only forgotten-drain detection); a drained result stays silent."""
        seen = []
        old_hook = sys.unraisablehook
        sys.unraisablehook = lambda unraisable: seen.append(unraisable.exc_value)
        try:
            undrained = DemoteResult()
            undrained.device_frees[ComponentType.FULL].append(torch.tensor([1]))
            del undrained

            drained = DemoteResult()
            drained.device_frees[ComponentType.FULL].append(torch.tensor([1]))
            drained.device_frees.clear()
            del drained
        finally:
            sys.unraisablehook = old_hook
        self.assertEqual(len(seen), 1)
        self.assertIsInstance(seen[0], AssertionError)

    def test_evict_host_drains_freed_host_values_to_the_pool(self):
        """Host eviction's returned frees must reach the host pool in the same
        call; a dropped drain leaves the pool permanently smaller."""
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        self._insert(cache, allocator, req_to_token_pool, [1, 2, 3, 4])
        leaf = next(iter(cache.root_node.children.values()))
        self.assertGreater(_write_backup(cache, leaf, write_back=True), 0)
        cache.writing_check(write_back=True)
        cache.evict(EvictParams(num_tokens=4))
        self.assertTrue(leaf.evicted)

        host_pool = cache.cache_controller.mem_pool_host
        available_before = host_pool.available_size()
        evicted = cache.evict_host(4)
        self.assertGreater(evicted, 0)
        self.assertGreater(host_pool.available_size(), available_before)
        cache.sanity_check()


if __name__ == "__main__":
    unittest.main()
