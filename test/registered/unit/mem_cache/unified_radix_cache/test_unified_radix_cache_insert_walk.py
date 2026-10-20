"""Resumable insert walk, returned-value drain and segment lock tests for UnifiedRadixCache."""

import sys
import unittest
from array import array
from collections import defaultdict
from dataclasses import replace
from unittest import mock

import torch

from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    InsertParams,
    TreeLock,
)
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.cache_action import (
    FreeDeviceKV,
    FreeDeviceKVFullOnly,
    UpdatePendingBackupOnNodeSplit,
)
from sglang.srt.mem_cache.unified_cache.components.base import (
    ComponentType,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
    DecSwaLockOnlyResult,
    DemoteResult,
    DriveHostEvictionResult,
    DropSubtreeNoHostResult,
    EvictDeviceLeafResult,
    EvictDeviceNextNodeResult,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.unified_radix_cache.config import (
    CacheConfig,
)
from sglang.test.kits.unified_radix_cache.fixtures import (
    build_fixture,
    match_prefix,
    node_children,
    node_device_lock_ref,
    node_device_value,
    node_key_length,
    node_parent,
    node_token_ids,
    write_backup,
)
from sglang.test.kits.unified_radix_cache.insert_walk import (
    ResumableInsertWalkSWAFixtures,
    SegmentLockFuzzFixtures,
    SegmentLockProtocolFixtures,
    StreamingSessionLockLifecycleFixtures,
)
from sglang.test.kits.unified_radix_cache.mixins import UnifiedRadixCacheFixtures
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


@unittest.skipUnless(torch.cuda.is_available(), "cache fixtures need CUDA")
class TestResumableInsertWalk(UnifiedRadixCacheFixtures, CustomTestCase):
    cfg = CacheConfig()

    def test_walk_backup_can_host_evict_on_path_h_leaf(self):
        """A crossing node's backup runs at its walk step, so its host eviction
        can still take an H-leaf deeper on the inserted path."""
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()

        self._insert(cache, allocator, req_to_token_pool, [1, 2, 3, 4])
        (top,) = node_children(cache, cache.root_node_handle())
        self._insert(cache, allocator, req_to_token_pool, list(range(1, 9)))
        (h_leaf,) = node_children(cache, top)
        self.assertGreater(write_backup(cache, h_leaf, write_back=True), 0)
        cache.writing_check(write_back=True)
        cache.evict(EvictParams(num_tokens=4))
        self.assertTrue(cache.tree_core.is_full_device_evicted(h_leaf))

        # Fill the host pool below len(top) free, keeping the on-path H-leaf
        # the oldest host entry and pinning the unbacked path root.
        top_lock = cache.inc_lock_ref(top)
        host_pool = cache.cache_controller.mem_pool_host
        start = 1000
        top_len = node_key_length(cache, top)
        while host_pool.available_size() >= top_len:
            count = min(host_pool.available_size() - top_len + 1, 250)
            tokens = list(range(start, start + count))
            start += 1000
            self._insert(cache, allocator, req_to_token_pool, tokens)
            filler = None
            for child in node_children(cache, cache.root_node_handle()):
                if child != top and not cache.tree_core.is_full_device_evicted(child):
                    filler = child
            self.assertIsNotNone(filler)
            self.assertGreater(write_backup(cache, filler, write_back=True), 0)
            cache.writing_check(write_back=True)
            cache.evict(EvictParams(num_tokens=count))
            self.assertTrue(cache.tree_core.is_full_device_evicted(filler))
        cache.dec_lock_ref(top, top_lock.to_dec_params())

        # The crossing backup evicts exactly the on-path H-leaf, then the
        # remaining suffix is recreated as a fresh leaf.
        cache.write_through_threshold = cache.tree_core.get_node_hit_count(top) + 1
        self._insert(cache, allocator, req_to_token_pool, list(range(1, 13)))
        cache.writing_check(write_back=True)

        self.assertTrue(cache.tree_core.is_backuped(top))
        self.assertNotIn(h_leaf, node_children(cache, top))
        self.assertFalse(cache.tree_core.contains_node(h_leaf))
        (child_key_len,) = {
            node_key_length(cache, child) for child in node_children(cache, top)
        }
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
        self.assertFalse(cache.tree_core.has_ongoing_insert())

        # The tree stays usable and the crossing re-fires on the next walk.
        self._insert(cache, allocator, req_to_token_pool, [1, 2, 3, 4])
        cache.writing_check(write_back=True)
        (ancestor,) = node_children(cache, cache.root_node_handle())
        self.assertTrue(cache.tree_core.is_backuped(ancestor))
        cache.sanity_check()

    def test_insert_abort_drains_pending_deferred_frees(self):
        """A mid-insert failure after a deferred dup-free accumulated must still
        return those slots to the allocator via the end_insert drain."""
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        cache.write_through_threshold = sys.maxsize
        self._insert(cache, allocator, req_to_token_pool, [1, 2, 3, 4])
        self._insert(cache, allocator, req_to_token_pool, [1, 2])

        (parent,) = node_children(cache, cache.root_node_handle())
        cache.write_through_threshold = cache.tree_core.get_node_hit_count(parent) + 1

        # The parent emits a backup barrier. The injected failure advances the
        # next overlap step, leaving its duplicate free pending in TreeCore.
        available = allocator.available_size()

        def fail_after_one_walk_step():
            cache.tree_core.advance_insert_walk_once()
            raise RuntimeError("boom")

        with mock.patch.object(
            cache.tree_core, "resume_insert", fail_after_one_walk_step
        ):
            with self.assertRaises(RuntimeError):
                self._insert(cache, allocator, req_to_token_pool, [1, 2, 3, 4])

        # All four duplicate slots are returned: two at the barrier, then two
        # from end_insert after the injected abort.
        self.assertEqual(allocator.available_size(), available)
        self.assertFalse(cache.tree_core.has_ongoing_insert())
        cache.writing_check()
        cache.sanity_check()

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
        (node,) = node_children(cache, cache.root_node_handle())

        self.assertGreater(write_backup(cache, node, write_back=True), 0)
        cache.writing_check(write_back=True)
        self.assertTrue(cache.tree_core.is_backuped(node))
        host_avail = cache.cache_controller.mem_pool_host.available_size()

        # Re-applying an overlapping chain is a no-op skip, not a re-backup.
        self.assertEqual(write_backup(cache, node, write_back=True), 0)
        cache.writing_check(write_back=True)
        self.assertEqual(
            cache.cache_controller.mem_pool_host.available_size(), host_avail
        )

    def test_shallower_crossing_backs_up_above_backuped_middle(self):
        """A shallower crossing node above a backuped middle must back up in
        the same insert as the deeper crossing (its own walk barrier)."""
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        self._insert(cache, allocator, req_to_token_pool, [1, 2, 3, 4])
        (top,) = node_children(cache, cache.root_node_handle())

        # Break backup continuity: a backuped (then device-evicted) middle
        # below the still-unbacked top. insert_host refills below an
        # un-backed-up node are dropped under write-through
        # (host_insert_dropped), so the state is built through an explicit
        # backup + device eviction.
        self._insert(cache, allocator, req_to_token_pool, list(range(1, 9)))
        (middle,) = node_children(cache, top)
        self.assertGreater(write_backup(cache, middle, write_back=True), 0)
        cache.writing_check(write_back=True)
        cache.evict(EvictParams(num_tokens=4))
        self.assertTrue(cache.tree_core.is_full_device_evicted(middle))
        self.assertTrue(cache.tree_core.is_backuped(middle))
        self.assertFalse(cache.tree_core.is_backuped(top))

        # The device insert unevicts the middle and adds the deep leaf.
        self._insert(cache, allocator, req_to_token_pool, list(range(1, 13)))
        (deep,) = node_children(cache, middle)

        cache.write_through_threshold = (
            min(
                cache.tree_core.get_node_hit_count(top),
                cache.tree_core.get_node_hit_count(deep),
            )
            + 1
        )
        self._insert(cache, allocator, req_to_token_pool, list(range(1, 17)))
        cache.writing_check(write_back=True)
        self.assertTrue(cache.tree_core.is_backuped(top))
        self.assertTrue(cache.tree_core.is_backuped(deep))

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

    def test_write_through_publish_list_is_ordered_ancestors_first(self):
        """One ack spanning several nodes publishes a parent before its children,
        whatever order the component transfers listed them in."""
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        self._insert(cache, allocator, req_to_token_pool, [1, 2, 3, 4])
        (parent,) = node_children(cache, cache.root_node_handle())
        self._insert(cache, allocator, req_to_token_pool, [1, 2, 3, 4, 5, 6])
        (child,) = node_children(cache, parent)

        cache._track_write_through_node(
            child, lock_params=None, publish_node_ids=[child, parent]
        )

        self.assertEqual(
            cache.ongoing_write_through[child].publish_node_ids, [parent, child]
        )
        cache._finish_write_through_ack(child)
        self.assertIsNone(cache.tree_core.get_write_through_pending_id(parent))
        self.assertIsNone(cache.tree_core.get_write_through_pending_id(child))
        cache.sanity_check()

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
                match_prefix(cache, [1, 2])

        # The relocation reached the pending record: the ack clears the
        # pending marker on both split halves, not just the stale node.
        cache.writing_check(write_back=True)
        (parent,) = node_children(cache, cache.root_node_handle())
        (child,) = node_children(cache, parent)
        self.assertIsNone(cache.tree_core.get_write_through_pending_id(parent))
        self.assertIsNone(cache.tree_core.get_write_through_pending_id(child))
        cache.sanity_check()


@unittest.skipUnless(torch.cuda.is_available(), "cache fixtures need CUDA")
class TestResumableInsertWalkSWA(ResumableInsertWalkSWAFixtures, CustomTestCase):
    cfg = CacheConfig(
        components=(ComponentType.FULL, ComponentType.SWA), sliding_window_size=8
    )

    def test_swa_tombstone_recovery_reports_unlocked_full_adoption(self):
        result, seq_len = self._recover_swa_tombstone_with_tracking(lock_full=False)
        self.assertEqual(result.adopted_ranges[ComponentType.FULL], [(0, seq_len)])
        self.assertEqual(result.adopted_ranges[ComponentType.SWA], [(0, seq_len)])

    def test_swa_tombstone_recovery_keeps_locked_full_out_of_adoption(self):
        result, seq_len = self._recover_swa_tombstone_with_tracking(lock_full=True)
        self.assertNotIn(ComponentType.FULL, result.adopted_ranges)
        self.assertEqual(result.adopted_ranges[ComponentType.SWA], [(0, seq_len)])

    def test_swa_tombstone_recovery_frees_full_only(self):
        sw = self.cfg.sliding_window_size
        cache, allocator, _ = build_fixture(self.cfg)
        seq = list(range(1, 2 * sw + 1))
        key = RadixKey(array("q", seq))
        evicted = self._alloc(allocator, len(seq))
        allocator.free_swa(evicted[:sw])
        cache.insert(
            InsertParams(
                key=key,
                value=evicted,
                component_evicted_seqlens={ComponentType.SWA: sw},
            )
        )
        value = self._alloc(allocator, len(seq))
        full_available = allocator.full_attn_allocator.available_size()
        swa_available = allocator.swa_attn_allocator.available_size()
        cache.insert(InsertParams(key=key, value=value))

        self.assertEqual(
            allocator.full_attn_allocator.available_size(),
            full_available + len(seq),
        )
        self.assertEqual(
            allocator.swa_attn_allocator.available_size(),
            swa_available + sw,
        )
        cache.sanity_check()

    def test_swa_recovery_keeps_recovered_node_below_window_nodes(self):
        """A tombstone recovered during the walk lands below the in-window path
        in the SWA LRU, so eviction takes the recovered span first."""

        sw = self.cfg.sliding_window_size
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq = list(range(1, 2 * sw + 1))
        key = RadixKey(array("q", seq))
        evicted = self._alloc(allocator, len(seq))
        allocator.free_swa(evicted[:sw])
        cache.insert(
            InsertParams(
                key=key,
                value=evicted,
                component_evicted_seqlens={ComponentType.SWA: sw},
            )
        )
        (prefix_node,) = node_children(cache, cache.root_node_handle())
        (window_node,) = node_children(cache, prefix_node)
        self.assertIsNone(node_device_value(cache, prefix_node, ComponentType.SWA))

        # Re-inserting fully in-window recovers the prefix span's SWA data.
        cache.insert(InsertParams(key=key, value=self._alloc(allocator, len(seq))))
        self.assertIsNotNone(node_device_value(cache, prefix_node, ComponentType.SWA))

        # SWA eviction takes the recovered span and keeps the window leaf.
        cache.evict(EvictParams(num_tokens=0, swa_num_tokens=sw))
        self.assertIsNone(node_device_value(cache, prefix_node, ComponentType.SWA))
        self.assertIsNotNone(node_device_value(cache, window_node, ComponentType.SWA))
        self.assertIsNotNone(node_device_value(cache, window_node, ComponentType.FULL))
        cache.sanity_check()

    def test_dup_slice_below_eviction_floor_frees_full_only(self):
        """A re-insert whose duplicate slice starts below the request's eviction
        floor gives back only the full side there: those SWA peers are gone."""
        sw = self.cfg.sliding_window_size
        cache, allocator, _ = build_fixture(self.cfg)
        seq = list(range(1, 2 * sw + 1))
        key = RadixKey(array("q", seq))
        cache.insert(InsertParams(key=key, value=self._alloc(allocator, len(seq))))

        value = self._alloc(allocator, len(seq))
        # Window eviction already released the peers below the floor.
        allocator.free_swa(value[:sw])
        with mock.patch.object(
            cache, "_apply_cache_action", wraps=cache._apply_cache_action
        ) as spy:
            cache.insert(
                InsertParams(
                    key=key,
                    value=value,
                    component_evicted_seqlens={ComponentType.SWA: sw},
                )
            )
        actions = [c.args[0] for c in spy.call_args_list]

        full_only = [
            i for a in actions if isinstance(a, FreeDeviceKVFullOnly) for i in a.indices
        ]
        both = [i for a in actions if isinstance(a, FreeDeviceKV) for i in a.indices]
        torch.testing.assert_close(torch.cat(full_only), value[:sw])
        torch.testing.assert_close(torch.cat(both), value[sw:])
        cache.sanity_check()

    def test_dec_swa_lock_only_early_release_keeps_full_lock(self):
        """The scheduler's early SWA release (decode past the window) drops
        only the SWA lock; the Full path lock stays held until dec_lock_ref."""
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq = self._make_seq(1, self.cfg.sliding_window_size + 4)
        self._insert(cache, allocator, req_to_token_pool, seq)
        m = match_prefix(cache, seq)
        node = m.last_device_node

        lock_result = cache.inc_lock_ref(node)
        self.assertGreaterEqual(node_device_lock_ref(cache, node, ComponentType.SWA), 1)
        cache.release_swa(TreeLock(node, lock_result.to_dec_params()))
        self.assertEqual(node_device_lock_ref(cache, node, ComponentType.SWA), 0)
        self.assertGreaterEqual(
            node_device_lock_ref(cache, node, ComponentType.FULL), 1
        )

        cache.dec_lock_ref(node, lock_result.to_dec_params(), skip_swa=True)
        cache.sanity_check()


@unittest.skipUnless(torch.cuda.is_available(), "cache fixtures need CUDA")
class TestResumableInsertWalkWriteBack(UnifiedRadixCacheFixtures, CustomTestCase):
    cfg = CacheConfig()

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
        m = match_prefix(cache, p1)
        parent = m.last_device_node
        (child,) = node_children(cache, parent)
        (grandchild,) = node_children(cache, child)
        self.assertTrue(
            cache.tree_core.is_full_device_evicted(child)
            and cache.tree_core.is_backuped(child)
        )
        self.assertTrue(
            cache.tree_core.is_full_device_evicted(grandchild)
            and cache.tree_core.is_backuped(grandchild)
        )

        # Chain 2: a younger unbacked leaf needing 4 host slots; only 2 can
        # come from evict_host (the grandchild leaf) — the other 2 exist only
        # if the drop fallback's child slots are drained within the round.
        p2 = self._make_seq(5000, 4)
        self._insert(cache, allocator, req_to_token_pool, p2)
        leaf2 = None
        for node in node_children(cache, cache.root_node_handle()):
            if node_token_ids(cache, node)[: len(p2)] == list(p2):
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
        self.assertTrue(
            cache.tree_core.is_full_device_evicted(leaf2)
            and cache.tree_core.is_backuped(leaf2)
        )
        cache.sanity_check()


@unittest.skipUnless(torch.cuda.is_available(), "cache fixtures need CUDA")
class TestReturnedValuesDrain(UnifiedRadixCacheFixtures, CustomTestCase):
    """Drain contract of the returned-values eviction API: every tree-core step
    result is drained exactly once, in per-component insertion order."""

    cfg = CacheConfig()

    def test_action_only_eviction_step_makes_progress(self):
        # The patched hook is the Python component's; the Rust core runs its own.
        cache, _, _ = build_fixture(self.cfg, tree_core_backend="python")
        core = cache.tree_core
        component = cache.components[ComponentType.FULL]
        root = cache.root_node_handle()
        action = UpdatePendingBackupOnNodeSplit(
            ack_id=None, old_node_id=root, new_node_id=root, new_child_node_id=root
        )

        def emit_action(tracker, device_frees, host_frees, cache_actions):
            cache_actions.append(action)
            return None

        core.evict_device_start(ComponentType.FULL, 1)
        try:
            with (
                mock.patch.object(component, "_evict_device_next_node", emit_action),
                mock.patch.object(cache, "_apply_cache_action") as apply,
            ):
                self.assertEqual(
                    cache._evict_device_next_node(ComponentType.FULL, {}),
                    (None, True),
                )
                apply.assert_called_once_with(action)
        finally:
            core.evict_device_end(ComponentType.FULL)

    def test_each_eviction_step_result_is_drained(self):
        """Every step wrapper hands its result's frees to _free_values and
        passes the payload through; a wrapper that forgets strands pool slots."""
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        self._insert(cache, allocator, req_to_token_pool, [1, 2, 3, 4])
        (node,) = node_children(cache, cache.root_node_handle())
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
                lambda: make(
                    EvictDeviceNextNodeResult,
                    node_id=node,
                    made_progress=True,
                ),
                lambda: cache._evict_device_next_node(ComponentType.FULL, tracker),
                (node, True),
            ),
            (
                "evict_device_leaf",
                lambda: make(EvictDeviceLeafResult),
                lambda: cache._evict_device_leaf(node, tracker),
                None,
            ),
            (
                "demote",
                lambda: make(DemoteResult),
                lambda: cache._demote(node, tracker),
                None,
            ),
            (
                "drop_subtree_no_host",
                lambda: make(DropSubtreeNoHostResult, is_dropped=True),
                lambda: cache._drop_subtree_no_host(node, tracker),
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
                lambda: cache.release_swa(TreeLock(node, DecLockRefParams())),
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

                with (
                    mock.patch.object(
                        cache.tree_core, name, return_value=make_result()
                    ),
                    mock.patch.object(cache, "_free_values", side_effect=record),
                ):
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
        with (
            mock.patch.object(
                cache,
                "_apply_cache_action",
                side_effect=lambda action: freed.append(
                    ("device", action.component_type)
                ),
            ),
            mock.patch.dict(cache.components, fake_components),
        ):
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
        with (
            mock.patch.object(cache, "_apply_cache_action", side_effect=boom_on_swa),
            mock.patch.dict(cache.components, {ComponentType.FULL: host_mock}),
        ):
            with self.assertRaises(RuntimeError):
                cache._free_values(device_frees, host_frees)

        self.assertEqual(list(device_frees), [ComponentType.MAMBA])
        self.assertFalse(host_frees)
        host_mock.free_host_values.assert_called_once()

    def test_node_info_facades_forward_to_the_tree(self):
        """The scheduler's storage-prefetch facades (is_backuped / is_root /
        get_last_hash_value / get_prefix_hash_values) resolve NodeIds tree-side
        with node-level semantics."""
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        self._insert(cache, allocator, req_to_token_pool, [1, 2, 3, 4])
        self._insert(cache, allocator, req_to_token_pool, list(range(1, 9)))
        (parent,) = node_children(cache, cache.root_node_handle())
        (child,) = node_children(cache, parent)

        self.assertTrue(cache.is_root(cache.root_node_handle()))
        self.assertFalse(cache.is_root(child))
        self.assertFalse(cache.is_backuped(parent))
        self.assertIsNone(cache.get_last_hash_value(parent))

        self.assertGreater(write_backup(cache, parent, write_back=True), 0)
        cache.writing_check(write_back=True)
        self.assertTrue(cache.is_backuped(parent))

        cache.tree_core.set_node_hash_values(parent, ["h1", "h2"])
        self.assertEqual(cache.get_last_hash_value(parent), "h2")
        # The prefix chain carries the ancestors' hashes, not the node's own.
        self.assertEqual(cache.get_prefix_hash_values(parent), [])
        self.assertEqual(cache.get_prefix_hash_values(child), ["h1", "h2"])


@unittest.skipUnless(torch.cuda.is_available(), "cache fixtures need CUDA")
class TestSegmentLockProtocol(SegmentLockProtocolFixtures, CustomTestCase):
    """Segment-lock protocol regressions, replaying the production lock-theft
    failure classes (F1/F2) and the split hazards.

    The protocol: a lock covers the contiguous node segment
    [start, boundary-uuid], counting every node (tombstones included), so a
    release needs only the receipt (anchor node, boundary uuid, skipped
    components) and any ref==0 met inside the segment is a hard protocol
    violation. The replays read the tree through the inspection interface, so
    they run unchanged against the Python and Rust cores.
    """

    cfg = CacheConfig(
        components=(ComponentType.FULL, ComponentType.SWA), sliding_window_size=8
    )

    def test_rebuilt_tombstone_relock_release_no_theft(self):
        """F1 attribution replay: A locks a window containing a tombstone; the
        tombstone is rebuilt and locked by B mid-hold; A's release must leave
        B's refs intact (the old skip-set protocol decremented B's lock)."""
        sw = self.cfg.sliding_window_size
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq = self._make_seq(1, 2 * sw)
        # SWA data only for the last sw//2 positions: the window has a hole.
        # As in production, the evicted prefix's SWA slots are released before
        # the insert so a later FULL free finds no live SWA peer.
        swa_evicted = len(seq) - sw // 2
        value = self._alloc(allocator, len(seq))
        allocator.free_swa(value[:swa_evicted])
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", seq)),
                value=value,
                component_evicted_seqlens={ComponentType.SWA: swa_evicted},
            )
        )
        leaf = self._deepest(cache)
        segment = self._segment(cache, leaf, sw)
        self.assertTrue(
            any(
                node_device_value(cache, n, ComponentType.SWA) is None for n in segment
            ),
            "fixture must place a tombstone inside the window",
        )

        lock_a = cache.inc_lock_ref(leaf)
        # Count-everything: every segment node carries A's ref, tombstones
        # included, and the boundary uuid is always stamped.
        self.assertIsNotNone(lock_a.component_lock_uuids[ComponentType.SWA])
        self.assertEqual(lock_a.node_id, leaf)
        for n in segment:
            self.assertEqual(self._swa_ref(cache, n), 1)
        cache.sanity_check()

        # Rebuild the tombstones under A's lock (Recover path: FULL is
        # locked); the rebuilt values must be credited to protected.
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", seq)),
                value=self._alloc(allocator, len(seq)),
            )
        )
        cache.sanity_check()
        leaf = self._deepest(cache)
        segment = self._segment(cache, leaf, sw)
        for n in segment:
            self.assertIsNotNone(node_device_value(cache, n, ComponentType.SWA))

        lock_b = cache.inc_lock_ref(leaf)
        for n in segment:
            self.assertEqual(self._swa_ref(cache, n), 2)

        # THE regression: A's release takes back exactly A's refs.
        cache.dec_lock_ref(leaf, lock_a.to_dec_params())
        for n in segment:
            self.assertEqual(self._swa_ref(cache, n), 1)
        cache.sanity_check()

        cache.dec_lock_ref(leaf, lock_b.to_dec_params())
        for n in segment:
            self.assertEqual(self._swa_ref(cache, n), 0)
        cache.sanity_check()

    def test_root_release_needs_no_boundary(self):
        cache, _, _ = build_fixture(self.cfg)
        core = cache.tree_core
        root = cache.root_node_handle()
        receipt = DecLockRefParams(node_id=root)
        core.dec_lock_ref(root, receipt)
        core.dec_host_lock_ref(root, receipt)
        released = core.dec_swa_lock_only(root, receipt)
        self.assertFalse(released.device_frees)
        self.assertFalse(released.host_frees)
        released = core.dec_window_lock_only(root, ComponentType.SWA, receipt)
        self.assertFalse(released.device_frees)
        self.assertFalse(released.host_frees)
        cache.sanity_check()

    def test_root_acquire_receipt_releases_on_root(self):
        """A cold request locks the root, so its receipt is anchored on root."""
        cache, _, _ = build_fixture(self.cfg)
        root = cache.root_node_handle()
        lock = cache.inc_lock_ref(root).to_dec_params()
        self.assertEqual(lock.node_id, root)
        cache.dec_lock_ref(root, lock)
        cache.sanity_check()

    def test_release_with_incorrect_root_boundary_fails_loud(self):
        """An incorrect root boundary must fail at the acquired segment's edge."""
        sw = self.cfg.sliding_window_size
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq = self._make_seq(1, 3 * sw)
        self._insert(cache, allocator, req_to_token_pool, seq)
        leaf = self._match_leaf(cache, seq)
        lock = cache.inc_lock_ref(leaf)
        self.assertIsNotNone(lock.component_lock_uuids[ComponentType.SWA])

        self._assert_protocol_violation(
            lambda: cache.dec_lock_ref(
                leaf,
                DecLockRefParams(
                    node_id=leaf, component_lock_uuids={ComponentType.SWA: None}
                ),
            ),
            "lock_ref=0",
        )

    def test_release_on_another_node_fails_loud(self):
        """The receipt anchors the lock on the node it was taken on; replaying
        it on a different node (the rematch-clobbered ``req.last_node`` class
        of bug) must assert instead of walking that node's segment."""
        sw = self.cfg.sliding_window_size
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq = self._make_seq(1, 2 * sw)
        self._insert(cache, allocator, req_to_token_pool, seq)
        leaf = self._match_leaf(cache, seq)
        parent = node_parent(cache, leaf)
        self.assertFalse(cache.tree_core.is_root(parent))
        lock = cache.inc_lock_ref(leaf)
        self.assertEqual(lock.node_id, leaf)

        self._assert_protocol_violation(
            lambda: cache.dec_lock_ref(parent, lock.to_dec_params()),
            "lock receipt anchored on node",
        )

    def test_unanchored_device_release_fails_loud(self):
        """#41261 replay: a receipt that lost its anchor must not walk the
        node's segment and drop locks other holders own."""
        cache, allocator, leaf = self._leaf_with_lock_path()
        lock = cache.inc_lock_ref(leaf).to_dec_params()
        self._assert_unanchored_release_rejected(
            cache,
            allocator,
            leaf,
            lambda: cache.dec_lock_ref(leaf, replace(lock, node_id=None)),
            lambda: cache.dec_lock_ref(leaf, lock),
        )

    def test_unanchored_host_release_fails_loud(self):
        cache, allocator, leaf = self._leaf_with_lock_path(hicache=True)
        self._backup_node(cache, leaf)
        lock = cache.inc_host_lock_ref(leaf).to_dec_params()
        self._assert_unanchored_release_rejected(
            cache,
            allocator,
            leaf,
            lambda: cache.dec_host_lock_ref(leaf, replace(lock, node_id=None)),
            lambda: cache.dec_host_lock_ref(leaf, lock),
        )

    def test_unanchored_swa_only_release_fails_loud(self):
        cache, allocator, leaf = self._leaf_with_lock_path()
        lock = cache.inc_lock_ref(leaf).to_dec_params()
        self._assert_unanchored_release_rejected(
            cache,
            allocator,
            leaf,
            lambda: cache.tree_core.dec_swa_lock_only(
                leaf, replace(lock, node_id=None)
            ),
            lambda: cache.dec_lock_ref(leaf, lock),
        )

    def test_unanchored_window_only_release_fails_loud(self):
        cache, allocator, leaf = self._leaf_with_lock_path()
        lock = cache.inc_lock_ref(leaf).to_dec_params()
        self._assert_unanchored_release_rejected(
            cache,
            allocator,
            leaf,
            lambda: cache.tree_core.dec_window_lock_only(
                leaf, ComponentType.SWA, replace(lock, node_id=None)
            ),
            lambda: cache.dec_lock_ref(leaf, lock),
        )

    def test_unanchored_root_release_fails_loud(self):
        """Every acquire anchors its receipt, root included; an unanchored
        receipt lost its anchor upstream even where releasing is a no-op."""
        cache, _, _ = build_fixture(self.cfg)
        root = cache.root_node_handle()
        self._assert_protocol_violation(
            lambda: cache.tree_core.dec_lock_ref(root, DecLockRefParams()),
            "lock receipt anchored on node",
        )

    def test_double_release_fails_loud(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq = self._make_seq(1, 2 * self.cfg.sliding_window_size)
        self._insert(cache, allocator, req_to_token_pool, seq)
        leaf = self._match_leaf(cache, seq)
        lock = cache.inc_lock_ref(leaf)
        cache.dec_lock_ref(leaf, lock.to_dec_params())
        self._assert_protocol_violation(
            lambda: cache.dec_lock_ref(leaf, lock.to_dec_params()), "lock_ref=0"
        )

    def test_finish_after_early_release_without_skip_swa_fails_loud(self):
        """F2 replay: retraction-after-early-release used to run a second SWA
        walk that stole ancestors' locks; now it dies at the first node."""
        sw = self.cfg.sliding_window_size
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq = self._make_seq(1, 2 * sw)
        self._insert(cache, allocator, req_to_token_pool, seq)
        leaf = self._match_leaf(cache, seq)
        lock = cache.inc_lock_ref(leaf)
        cache.release_swa(TreeLock(leaf, lock.to_dec_params()))
        self._assert_protocol_violation(
            lambda: cache.dec_lock_ref(leaf, lock.to_dec_params()), "lock_ref=0"
        )

    def test_finish_after_early_release_with_skip_swa(self):
        """The correct F2 flow: unlock honors the early release."""
        sw = self.cfg.sliding_window_size
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq = self._make_seq(1, 2 * sw)
        self._insert(cache, allocator, req_to_token_pool, seq)
        leaf = self._match_leaf(cache, seq)
        lock = cache.lock(leaf)
        cache.release_swa(lock)
        self.assertTrue(lock.swa_released)
        cache.release_swa(lock)  # a repeat is a no-op
        cache.unlock(lock)
        self.assertEqual(self._swa_ref(cache, leaf), 0)
        self.assertEqual(node_device_lock_ref(cache, leaf, ComponentType.FULL), 0)
        cache.sanity_check()

    def test_split_under_lock_releases_balanced(self):
        """A mid-segment split copies Full and SWA device refs and migrates the
        SWA boundary uuid; the original receipt (whose anchor stays on the
        deeper half) releases every copied ref without leaking either lock."""
        sw = self.cfg.sliding_window_size
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq = self._make_seq(1, 2 * sw)
        self._insert(cache, allocator, req_to_token_pool, seq)
        leaf = self._match_leaf(cache, seq)
        lock = cache.inc_lock_ref(leaf)
        pre_segment = self._segment(cache, leaf, sw)
        pre_full_path = self._full_path(cache, leaf)

        # Diverge inside the window to force a split of a locked node.
        fork = seq[: len(seq) - sw // 2] + self._make_seq(9000, sw)
        self._insert(cache, allocator, req_to_token_pool, fork)
        post_segment = self._segment(cache, leaf, sw)
        post_full_path = self._full_path(cache, leaf)
        self.assertGreater(len(post_segment), len(pre_segment))
        self.assertGreater(len(post_full_path), len(pre_full_path))
        for n in post_segment:
            self.assertEqual(self._swa_ref(cache, n), 1)
        for n in post_full_path:
            self.assertEqual(node_device_lock_ref(cache, n, ComponentType.FULL), 1)

        cache.dec_lock_ref(leaf, lock.to_dec_params())
        for n in post_segment:
            self.assertEqual(self._swa_ref(cache, n), 0)
        for n in post_full_path:
            self.assertEqual(node_device_lock_ref(cache, n, ComponentType.FULL), 0)
        cache.sanity_check()

    def test_aux_release_readmits_the_leaf_whatever_the_release_order(self):
        """Each component's release refreshes the leaf sets of the nodes it
        unlocks, so a leaf whose last lock is an auxiliary one is readmitted
        even when Full released first. Component-level replay of the Python
        core; the Rust crate covers its own order in its unit tests."""
        cache, allocator, req_to_token_pool = build_fixture(
            self.cfg, tree_core_backend="python"
        )
        seq = self._make_seq(1, self.cfg.sliding_window_size)
        self._insert(cache, allocator, req_to_token_pool, seq)
        leaf = self._match_leaf(cache, seq)
        node = cache.tree_core.node_by_id(leaf)
        self.assertIn(node, cache.tree_core.evictable_device_leaves)

        params = cache.inc_lock_ref(leaf).to_dec_params()
        self.assertNotIn(node, cache.tree_core.evictable_device_leaves)
        # Full first: its walk still sees the SWA lock, so the leaf stays out.
        cache.components[ComponentType.FULL].release_component_lock(node, params)
        self.assertNotIn(node, cache.tree_core.evictable_device_leaves)
        # The SWA release drops the last lock and must readmit the leaf itself.
        cache.components[ComponentType.SWA].release_component_lock(node, params)
        self.assertIn(node, cache.tree_core.evictable_device_leaves)
        cache.sanity_check()


@unittest.skipUnless(torch.cuda.is_available(), "cache fixtures need CUDA")
class TestSegmentLockFuzz(SegmentLockFuzzFixtures, CustomTestCase):
    """Random lock/insert/evict interleavings with the tree's own ledger
    recomputation (sanity_check) as the per-step oracle, on both cores."""

    cfg = CacheConfig(
        components=(ComponentType.FULL, ComponentType.SWA),
        sliding_window_size=8,
        kv_size=4096,
        max_num_reqs=256,
    )

    def test_fuzz_seed0(self):
        self._run_seed(0)

    def test_fuzz_seed1(self):
        self._run_seed(1)

    def test_fuzz_seed2(self):
        self._run_seed(2)


@unittest.skipUnless(torch.cuda.is_available(), "cache fixtures need CUDA")
class TestSegmentLockFuzzWithMamba(TestSegmentLockFuzz):
    """The FULL+SWA+MAMBA (Inkling) shape: the Mamba opt-out receipt, the
    lower-priority cascade on early SWA release, and Mamba evictions all join
    the interleavings."""

    cfg = CacheConfig(
        components=(ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA),
        sliding_window_size=8,
        kv_size=4096,
        max_num_reqs=256,
        mamba_cache_size=512,
    )


class TestStreamingSessionLockLifecycle(
    StreamingSessionLockLifecycleFixtures, CustomTestCase
):
    """A streaming session owns its record and prefix lock from the first
    turn's row allocation; every move of that lock keeps the tree balanced."""

    cfg = CacheConfig(
        page_size=1,
        components=(ComponentType.FULL, ComponentType.SWA),
        sliding_window_size=4,
        kv_size=64,
        max_context_len=64,
    )

    def test_slot_publishes_first_prompt_but_not_its_output(self):
        cache, allocator, pool = build_fixture(self.cfg)
        tokens = list(range(1, 9))
        req = self._admitted_turn(cache, allocator, pool, tokens, self._session("s"))
        cache.maybe_hand_to_session(req)
        slot = cache.session.slots["s"]

        cache.checkpoint(req, up_to=len(tokens))
        match = match_prefix(cache, tokens)
        self.assertEqual(slot.lock.node, match.last_device_node)
        self.assertIs(req.last_node, slot.virtual_node)

        req.output_ids = array("q", range(9, 13))
        req.refresh_fill_ids()
        pool.write((req.kv.req_pool_idx, slice(8, 12)), allocator.alloc(4))
        req.kv.kv_committed_len = 12
        req.kv.kv_allocated_len = 12
        req.finished_reason = FINISH_LENGTH(length=4)
        release_kv_cache(req, cache, checkpoint=True)

        self.assertIs(cache.session.slots["s"], slot)
        match = match_prefix(cache, range(1, 13))
        self.assertEqual(match.device_prefix_len, len(tokens))
        cache.session.release_session("s")
        cache.sanity_check()

    def test_session_turn_releases_slot_swa_prefix_lock_once(self):
        cache, allocator, pool = build_fixture(self.cfg)
        tokens = list(range(1, 9))
        req = self._admitted_turn(cache, allocator, pool, tokens, self._session("s"))
        cache.maybe_hand_to_session(req)
        cache.checkpoint(req, up_to=len(tokens))
        slot = cache.session.slots["s"]

        cache.release_swa_prefix_lock(req)
        self.assertTrue(slot.lock.swa_released)
        self.assertIsNone(req.lock)
        cache.release_swa_prefix_lock(req)

        req.finished_reason = FINISH_LENGTH(length=0)
        self.assertTrue(cache.session.try_cache_finished_req(req))
        cache.session.release_session("s")
        cache.sanity_check()

    def test_later_turn_borrows_without_publishing(self):
        cache, allocator, pool = build_fixture(self.cfg)
        slot, req = self._later_turn(cache, allocator, pool)
        slot_lock = slot.lock

        cache.checkpoint(req, up_to=12)
        self.assertIs(slot.lock, slot_lock)

        req.finished_reason = FINISH_LENGTH(length=0)
        self.assertTrue(cache.session.try_cache_finished_req(req))
        cache.session.release_session("s")
        cache.sanity_check()

    def test_later_turn_abort_releases_like_any_request(self):
        """The record and the slot's lock, SWA part already released, go back
        before the KV is checkpointed."""
        cache, allocator, pool = build_fixture(self.cfg)
        slot, req = self._later_turn(cache, allocator, pool)
        slot_lock = slot.lock
        cache.release_swa_prefix_lock(req)
        self.assertTrue(slot_lock.swa_released)

        req.finished_reason = FINISH_ABORT()
        release_kv_cache(req, cache, checkpoint=True)
        self.assertNotIn("s", cache.session.slots)
        cache.sanity_check()

    def test_bounded_replay_turn_borrows_record_without_hold_back(self):
        """A short turn borrows its whole record; holding back a window would end
        the match inside the published prompt, which the slot cannot rewind."""
        cache, allocator, pool = build_fixture(
            replace(self.cfg, decoder_swa_bounded_replay=True)
        )
        session = self._session("s")
        first = self._admitted_turn(cache, allocator, pool, list(range(1, 9)), session)
        cache.maybe_hand_to_session(first)
        cache.checkpoint(first, up_to=8)
        first.finished_reason = FINISH_LENGTH(length=0)
        self.assertTrue(cache.session.try_cache_finished_req(first))

        req = self._turn(list(range(1, 11)), session)
        req.init_next_round_input(cache)
        self.assertIn("s", cache.session.slots)
        self.assertEqual(req.prefix_len, 8)

        cache.session.release_session("s")
        cache.sanity_check()


if __name__ == "__main__":
    unittest.main()
