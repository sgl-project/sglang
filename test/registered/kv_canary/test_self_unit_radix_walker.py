from __future__ import annotations

import unittest
from array import array

from sglang.srt.kv_canary.radix_cache_walker import walk_radix_cache_for_canary
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import (
    ComponentType,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedTreeNode
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kv_canary.fixtures import (
    add_unified_child,
    make_unified_radix_cache,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="extra-a", runner_config="1-gpu-small")
register_amd_ci(est_time=30, suite="extra-a-test-1-gpu-small-amd")


class TestSelfUnitRadixWalker(CustomTestCase):
    def test_unified_swa_sweep_gates_on_swa_lock_not_full_lock(self):
        """With unlocked_only + swa_resident_only, the sweep filters on the SWA
        component lock: a FULL-locked node whose SWA lock was already released
        (early dec_swa_lock_only) must still be swept, and a node whose SWA
        lock is still held must not."""
        cache = make_unified_radix_cache((ComponentType.FULL, ComponentType.SWA))
        add_unified_child(cache, [1, 2], lock_ref=1, swa_value=[1, 2])
        held = add_unified_child(cache, [3, 4], swa_value=[3, 4])
        held.component_data[ComponentType.SWA].lock_ref = 1

        result = walk_radix_cache_for_canary(
            radix_cache=cache, unlocked_only=True, swa_resident_only=True
        )
        self.assertEqual(result.slot_indices.tolist(), [1, 2])

    def test_unified_walk_emits_full_component_slots(self):
        """Verify unified radix walking emits the base (full) component slots."""
        cache = make_unified_radix_cache((ComponentType.FULL,))
        add_unified_child(cache, [10, 20, 30])
        result = walk_radix_cache_for_canary(radix_cache=cache)
        self.assertEqual(result.slot_indices.tolist(), [10, 20, 30])
        self.assertEqual(result.positions.tolist(), [0, 1, 2])
        self.assertEqual(result.prev_slot_indices.tolist(), [-1, 10, 20])

    def test_unified_walk_unlocked_only_uses_full_lock_ref(self):
        """Verify unified radix walking honors the base component lock reference."""
        cache = make_unified_radix_cache((ComponentType.FULL,))
        add_unified_child(cache, [1, 2], lock_ref=1)
        add_unified_child(cache, [3, 4])
        result = walk_radix_cache_for_canary(radix_cache=cache, unlocked_only=True)
        self.assertEqual(result.slot_indices.tolist(), [3, 4])

    def test_unified_swa_resident_only_skips_evicted_swa_nodes(self):
        """Verify unified radix walking skips nodes whose SWA storage was evicted."""
        cache = make_unified_radix_cache((ComponentType.FULL, ComponentType.SWA))
        add_unified_child(cache, [1, 2], swa_value=None)
        add_unified_child(cache, [3, 4], swa_value=[3, 4])
        result = walk_radix_cache_for_canary(
            radix_cache=cache,
            swa_resident_only=True,
        )
        self.assertEqual(result.slot_indices.tolist(), [3, 4])

    def test_unified_walk_spans_device_evicted_nodes_without_emitting_them(self):
        """Verify device-evicted (host-only) nodes emit no slots but still advance
        positions by their key length and pass the prev-slot chain through."""
        cache = make_unified_radix_cache((ComponentType.FULL,))
        evicted = UnifiedTreeNode(cache.tree_components)
        evicted.parent = cache.root_node
        evicted.key = RadixKey(array("q", [7, 8]), None)
        cache.root_node.children[evicted.id] = evicted
        grandchild = add_unified_child(cache, [5, 6])
        cache.root_node.children.pop(grandchild.id)
        grandchild.parent = evicted
        evicted.children[grandchild.id] = grandchild
        result = walk_radix_cache_for_canary(radix_cache=cache)
        self.assertEqual(result.slot_indices.tolist(), [5, 6])
        self.assertEqual(result.positions.tolist(), [2, 3])
        self.assertEqual(result.prev_slot_indices.tolist(), [-1, 5])

    def test_unified_swa_resident_only_noop_without_swa_component(self):
        """Verify swa_resident_only is a no-op when SWA is not enabled."""
        cache = make_unified_radix_cache((ComponentType.FULL,))
        add_unified_child(cache, [1, 2])
        add_unified_child(cache, [3, 4])
        result = walk_radix_cache_for_canary(
            radix_cache=cache,
            swa_resident_only=True,
        )
        self.assertEqual(result.slot_indices.tolist(), [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
