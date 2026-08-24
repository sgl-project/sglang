"""Unit tests for the device-only C++ unified TreeCore backend."""

import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.cache_action import (
    FreeDeviceKV,
    RecoverSWAWithLockedFull,
    SWARebuild,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.cpp_unified_tree_core import (
    CppUnifiedTreeCore,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def _params(tree_components=(ComponentType.FULL,), **kwargs) -> CacheInitParams:
    return CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=2,
        tree_components=tree_components,
        **kwargs,
    )


def _key(*tokens: int) -> RadixKey:
    return RadixKey(array("q", tokens))


class CppUnifiedTreeCoreTest(CustomTestCase):
    def setUp(self):
        component = SimpleNamespace(tree_core=None)
        self.core = CppUnifiedTreeCore(_params(), {ComponentType.FULL: component})

    def _new_swa_core(self, sliding_window_size=4):
        components = {
            ComponentType.FULL: SimpleNamespace(tree_core=None),
            ComponentType.SWA: SimpleNamespace(tree_core=None),
        }
        return CppUnifiedTreeCore(
            _params(
                tree_components=(ComponentType.FULL, ComponentType.SWA),
                sliding_window_size=sliding_window_size,
            ),
            components,
        )

    def _apply_swa_actions(self, core, actions):
        """Model the controller's SWA action barrier without a real allocator."""
        for action in actions:
            if isinstance(action, FreeDeviceKV):
                continue
            if isinstance(action, SWARebuild):
                source = action.source_value
            elif isinstance(action, RecoverSWAWithLockedFull):
                source = action.incoming_full
            else:
                self.fail(f"unexpected insert action: {action!r}")
            core.set_component_device_value(
                action.node_id, ComponentType.SWA, source + 1000
            )

    def _insert_swa(
        self,
        core,
        tokens,
        values,
        *,
        prev_prefix_len=0,
        swa_evicted_seqlen=0,
    ):
        step = core.begin_insert(
            InsertParams(
                key=_key(*tokens),
                value=torch.tensor(values, dtype=torch.int64),
                prev_prefix_len=prev_prefix_len,
                swa_evicted_seqlen=swa_evicted_seqlen,
            )
        )
        self.assertIsNotNone(step.result)
        self._apply_swa_actions(core, step.actions)
        core.end_insert()
        return step

    def _insert(self, tokens, values, prev_prefix_len=0):
        step = self.core.begin_insert(
            InsertParams(
                key=_key(*tokens),
                value=torch.tensor(values, dtype=torch.int64),
                prev_prefix_len=prev_prefix_len,
            )
        )
        self.assertIsNotNone(step.result)
        self.core.end_insert()
        return step

    def test_insert_match_split_and_flatten(self):
        step = self._insert([1, 2, 3, 4], [10, 11, 12, 13])
        self.assertEqual(step.result.prefix_len, 0)

        result = self.core.match_prefix(MatchPrefixParams(key=_key(1, 2, 9, 9)))
        self.assertEqual(result.device_indices.tolist(), [10, 11])
        self.assertEqual(result.full_kv_hit_length, 2)
        self.assertNotEqual(result.last_device_node, 0)
        self.assertCountEqual(self.core.all_values_flatten().tolist(), [10, 11, 12, 13])
        self.assertEqual(self.core.total_size(), (4, 0))

    def test_zero_copy_buffer_honors_radix_key_limit_and_page_alignment(self):
        backing = array("q", [1, 2, 3, 4, 90, 91, 92])
        limited = RadixKey(backing, limit=5)
        step = self.core.begin_insert(
            InsertParams(
                key=limited,
                value=torch.tensor([10, 11, 12, 13, 99], dtype=torch.int64),
            )
        )
        self.core.end_insert()
        self.assertEqual(step.result.prefix_len, 0)

        # The binding sees the original seven-token buffer, but the explicit
        # logical length exposes only the four page-aligned tokens.
        result = self.core.match_prefix(MatchPrefixParams(key=limited))
        self.assertEqual(result.device_indices.tolist(), [10, 11, 12, 13])
        self.assertEqual(self.core.total_size(), (4, 0))
        with self.assertRaisesRegex(ValueError, "int64 buffer"):
            self.core.tree.match_prefix(array("i", [1, 2, 3, 4]))

    def test_inserted_node_match_and_lock_avoids_second_tree_walk(self):
        step = self._insert([1, 2, 3, 4], [10, 11, 12, 13])
        before = self.core.tree.debug_stats()
        match, lock = self.core.match_inserted_prefix_and_lock(
            step.result.last_device_node
        )
        after = self.core.tree.debug_stats()

        self.assertEqual(match.device_indices.tolist(), [10, 11, 12, 13])
        self.assertEqual(after["tree_walk_calls"], before["tree_walk_calls"])
        self.assertEqual(after["node_match_calls"], before["node_match_calls"] + 1)
        self.assertEqual(self.core.protected_size(), 4)
        self.core.dec_lock_ref(match.last_device_node, lock.to_dec_params())
        self.assertEqual(self.core.protected_size(), 0)

    def test_inserted_node_match_reuses_value_and_patches_only_overlap(self):
        self._insert([1, 2, 3, 4], [10, 11, 12, 13])
        incoming = torch.tensor([10, 11, 20, 21, 30, 31], dtype=torch.int64)
        step = self.core.begin_insert(
            InsertParams(
                key=_key(1, 2, 3, 4, 5, 6),
                value=incoming,
                prev_prefix_len=2,
            )
        )
        self.core.end_insert()
        self.assertEqual(step.result.prefix_len, 4)

        before = self.core.tree.debug_stats()
        match, lock = self.core.match_inserted_prefix_and_lock(
            step.result.last_device_node,
            inserted_value=incoming,
            existing_prefix_len=step.result.prefix_len,
            reused_prefix_len=2,
        )
        after = self.core.tree.debug_stats()

        # [0, 2) was already protected by the request, [2, 4) is patched from
        # the authoritative tree, and [4, 6) remains the newly inserted suffix.
        self.assertEqual(match.device_indices.tolist(), [10, 11, 12, 13, 30, 31])
        self.assertEqual(match.device_indices.data_ptr(), incoming.data_ptr())
        self.assertEqual(after["tree_walk_calls"], before["tree_walk_calls"])
        self.assertEqual(after["node_match_calls"], before["node_match_calls"] + 1)
        self.core.dec_lock_ref(match.last_device_node, lock.to_dec_params())

    def test_persistent_full_lru_evicts_untouched_branch_first(self):
        self._insert([1, 2, 3, 4], [10, 11, 12, 13])
        self._insert([5, 6, 7, 8], [20, 21, 22, 23])
        self.core.match_prefix(MatchPrefixParams(key=_key(1, 2, 3, 4)))

        self.core.evict_device_start(ComponentType.FULL, 4)
        evicted = self.core.evict_device_next_node(
            ComponentType.FULL, {ComponentType.FULL: 0}
        )
        try:
            self.assertEqual(evicted.tracker[ComponentType.FULL], 4)
            self.assertEqual(
                evicted.device_frees[ComponentType.FULL][0].tolist(),
                [20, 21, 22, 23],
            )
        finally:
            evicted.device_frees.clear()
            evicted.host_frees.clear()
        self.core.evict_device_end(ComponentType.FULL)
        self.assertEqual(
            self.core.match_prefix(
                MatchPrefixParams(key=_key(1, 2, 3, 4))
            ).device_indices.tolist(),
            [10, 11, 12, 13],
        )
        self.assertEqual(
            self.core.match_prefix(
                MatchPrefixParams(key=_key(5, 6, 7, 8))
            ).device_indices.numel(),
            0,
        )

    def test_duplicate_insert_returns_unowned_slots_as_action(self):
        self._insert([1, 2, 3, 4], [10, 11, 12, 13])
        step = self._insert([1, 2, 3, 4], [20, 21, 22, 23], prev_prefix_len=1)
        self.assertEqual(step.result.prefix_len, 4)
        self.assertEqual(len(step.actions), 1)
        self.assertIsInstance(step.actions[0], FreeDeviceKV)
        self.assertEqual(step.actions[0].indices[0].tolist(), [21, 22, 23])

        result = self.core.match_prefix(MatchPrefixParams(key=_key(1, 2, 3, 4)))
        self.assertEqual(result.device_indices.tolist(), [10, 11, 12, 13])

    def test_lock_accounting_and_eager_eviction_bridge(self):
        self._insert([1, 2, 3, 4], [10, 11, 12, 13])
        result = self.core.match_prefix(MatchPrefixParams(key=_key(1, 2)))
        node_id = result.last_device_node

        self.core.inc_lock_ref(node_id)
        self.assertEqual(self.core.protected_size(), 2)
        self.assertEqual(self.core.evictable_size(), 2)
        self.core.dec_lock_ref(node_id)
        self.assertEqual(self.core.protected_size(), 0)

        self.core.evict_device_start(ComponentType.FULL, 3)
        evicted = self.core.evict_device_next_node(
            ComponentType.FULL, {ComponentType.FULL: 0}
        )
        try:
            self.assertIsNone(evicted.node_id)
            self.assertGreaterEqual(evicted.tracker[ComponentType.FULL], 3)
            self.assertEqual(
                sum(len(v) for v in evicted.device_frees[ComponentType.FULL]),
                evicted.tracker[ComponentType.FULL],
            )
        finally:
            # Results assert that allocator-owned values are consumed.
            evicted.device_frees.clear()
            evicted.host_frees.clear()
        self.core.evict_device_end(ComponentType.FULL)
        self.assertEqual(self.core.total_size(), (0, 0))

    def test_reset_clears_tree_and_restores_cpp_root_handle(self):
        self._insert([1, 2], [10, 11])
        self.core.reset()
        self.assertEqual(self.core.root_node_handle(), 0)
        self.assertEqual(self.core.total_size(), (0, 0))
        self.assertEqual(
            self.core.match_prefix(MatchPrefixParams(key=_key(1, 2))).last_device_node,
            0,
        )

    def test_unsupported_namespace_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "extra_key"):
            self.core.match_prefix(
                MatchPrefixParams(key=RadixKey(array("q", [1, 2]), "tenant"))
            )

    def test_hicache_configuration_fails_closed(self):
        with self.assertRaisesRegex(NotImplementedError, "HiCache"):
            CppUnifiedTreeCore(
                _params(enable_hicache=True),
                {ComponentType.FULL: SimpleNamespace(tree_core=None)},
            )

    def test_rejected_insert_does_not_poison_the_next_insert(self):
        with self.assertRaisesRegex(ValueError, "extra_key"):
            self.core.begin_insert(
                InsertParams(
                    key=RadixKey(array("q", [1, 2]), "tenant"),
                    value=torch.tensor([10, 11], dtype=torch.int64),
                )
            )
        self.assertFalse(self.core.has_ongoing_insert())

        step = self._insert([3, 4], [12, 13])
        self.assertEqual(step.result.prefix_len, 0)
        self.assertEqual(
            self.core.match_prefix(
                MatchPrefixParams(key=_key(3, 4))
            ).device_indices.tolist(),
            [12, 13],
        )

    def test_swa_insert_match_boundary_and_component_sizes(self):
        core = self._new_swa_core(sliding_window_size=4)
        step = self._insert_swa(
            core,
            range(1, 9),
            range(10, 18),
            swa_evicted_seqlen=4,
        )
        self.assertEqual(step.result.prefix_len, 0)
        self.assertEqual(
            sum(isinstance(action, SWARebuild) for action in step.actions), 1
        )
        self.assertEqual(core.full_evictable_size(), 8)
        self.assertEqual(core.swa_evictable_size(), 4)

        full_match = core.match_prefix(MatchPrefixParams(key=_key(*range(1, 9))))
        self.assertEqual(full_match.device_indices.tolist(), list(range(10, 18)))
        self.assertEqual(full_match.full_kv_hit_length, 8)

        # FULL still matches six tokens, but the SWA suffix only covers two of
        # the four required window tokens, so the usable cache prefix is empty.
        partial_match = core.match_prefix(MatchPrefixParams(key=_key(*range(1, 7))))
        self.assertEqual(partial_match.device_indices.numel(), 0)
        self.assertEqual(partial_match.full_kv_hit_length, 6)

    def test_swa_native_flat_match_and_many_sibling_splits(self):
        """SWA leaf splitting stays correct with a wide shared-prefix node.

        Every insertion below splits a fresh child below the same parent.  This
        is the DSPARK/shared-system-prompt shape that previously linearly
        scanned every existing sibling in split_node(TreeNode*).
        """
        core = self._new_swa_core(sliding_window_size=4)
        common = [1, 2, 3, 4]
        paths = []
        for branch in range(128):
            path = common + [1000 + branch, 2000 + branch] + [7, 8, 9, 10, 11, 12]
            paths.append(path)
            self._insert_swa(
                core,
                path,
                range(branch * 100, branch * 100 + len(path)),
                prev_prefix_len=len(common),
                swa_evicted_seqlen=len(path) - 4,
            )

        for path in paths:
            result = core.match_prefix(MatchPrefixParams(key=_key(*path)))
            self.assertEqual(result.full_kv_hit_length, len(path))
            self.assertEqual(result.device_indices.numel(), len(path))

        flat, _, full_hit_length = core.tree.match_prefix_swa_flat(
            array("q", paths[-1]), len(paths[-1])
        )
        self.assertIsInstance(flat, torch.Tensor)
        self.assertEqual(flat.numel(), len(paths[-1]))
        self.assertEqual(full_hit_length, len(paths[-1]))

    def test_swa_window_lock_and_early_release(self):
        core = self._new_swa_core(sliding_window_size=4)
        self._insert_swa(
            core,
            range(1, 9),
            range(10, 18),
            swa_evicted_seqlen=4,
        )
        matched = core.match_prefix(MatchPrefixParams(key=_key(*range(1, 9))))
        lock_result = core.inc_lock_ref(matched.last_device_node)
        self.assertIsNotNone(lock_result.swa_uuid_for_lock)
        self.assertEqual(core.full_protected_size(), 8)
        self.assertEqual(core.swa_protected_size(), 4)

        early = core.dec_swa_lock_only(
            matched.last_device_node, lock_result.swa_uuid_for_lock
        )
        try:
            self.assertEqual(core.full_protected_size(), 8)
            self.assertEqual(core.swa_protected_size(), 0)
            self.assertFalse(early.device_frees)
        finally:
            early.device_frees.clear()
            early.host_frees.clear()
        core.dec_lock_ref(
            matched.last_device_node,
            lock_result.to_dec_params(),
            skip_swa=True,
        )
        self.assertEqual(core.full_protected_size(), 0)

    def test_swa_eviction_atomically_removes_an_unlocked_full_leaf(self):
        core = self._new_swa_core(sliding_window_size=4)
        self._insert_swa(
            core,
            range(1, 9),
            range(10, 18),
            swa_evicted_seqlen=4,
        )
        core.evict_device_start(ComponentType.SWA, 1)
        evicted = core.evict_device_next_node(
            ComponentType.SWA,
            {ComponentType.FULL: 0, ComponentType.SWA: 0},
        )
        try:
            self.assertEqual(evicted.tracker[ComponentType.FULL], 4)
            self.assertEqual(evicted.tracker[ComponentType.SWA], 4)
            self.assertEqual(
                sum(len(value) for value in evicted.device_frees[ComponentType.FULL]),
                4,
            )
            # SWA frees deliberately carry FULL logical indices; the allocator
            # uses them to release only the paired SWA slots.
            self.assertEqual(
                sum(len(value) for value in evicted.device_frees[ComponentType.SWA]),
                4,
            )
        finally:
            evicted.device_frees.clear()
            evicted.host_frees.clear()
        core.evict_device_end(ComponentType.SWA)
        self.assertEqual(core.total_size(), (4, 0))
        self.assertEqual(core.swa_evictable_size(), 0)

    def test_swa_tombstone_recovery_respects_owned_prefix_and_full_lock(self):
        core = self._new_swa_core(sliding_window_size=4)
        path_a = list(range(1, 13))
        path_b = list(range(1, 9)) + [90, 91, 92, 93]
        self._insert_swa(
            core,
            path_a,
            range(10, 22),
            swa_evicted_seqlen=4,
        )
        self._insert_swa(
            core,
            path_b,
            range(30, 42),
            prev_prefix_len=8,
            swa_evicted_seqlen=4,
        )

        # Touch the A leaf.  Its four-token SWA value covers the entire window,
        # leaving the shared internal SWA node as the oldest eviction victim.
        match_a = core.match_prefix(MatchPrefixParams(key=_key(*path_a)))
        self.assertEqual(match_a.device_indices.numel(), 12)
        core.evict_device_start(ComponentType.SWA, 1)
        evicted = core.evict_device_next_node(
            ComponentType.SWA,
            {ComponentType.FULL: 0, ComponentType.SWA: 0},
        )
        try:
            self.assertEqual(evicted.tracker.get(ComponentType.FULL, 0), 0)
            self.assertEqual(evicted.tracker[ComponentType.SWA], 4)
        finally:
            evicted.device_frees.clear()
            evicted.host_frees.clear()
        core.evict_device_end(ComponentType.SWA)
        self.assertEqual(core.total_size(), (16, 0))

        # A tombstone wholly inside the request-owned prefix must not consume,
        # replace, or free that FULL value.
        owned = self._insert_swa(
            core,
            path_a,
            range(10, 22),
            prev_prefix_len=12,
            swa_evicted_seqlen=4,
        )
        self.assertFalse(
            any(
                isinstance(action, (SWARebuild, RecoverSWAWithLockedFull))
                for action in owned.actions
            )
        )

        # Once fresh KV covers the tombstone, a concurrent FULL lock requires
        # the recovery action to preserve the locked FULL allocation.
        match_a = core.match_prefix(MatchPrefixParams(key=_key(*path_a)))
        lock_result = core.inc_lock_ref(match_a.last_device_node)
        recovered = self._insert_swa(
            core,
            path_a,
            range(100, 112),
            prev_prefix_len=4,
            swa_evicted_seqlen=4,
        )
        self.assertEqual(
            sum(
                isinstance(action, RecoverSWAWithLockedFull)
                for action in recovered.actions
            ),
            1,
        )
        core.dec_lock_ref(match_a.last_device_node, lock_result.to_dec_params())
        self.assertEqual(core.swa_evictable_size(), 12)


if __name__ == "__main__":
    unittest.main()
