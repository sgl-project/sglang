from __future__ import annotations

import unittest

import torch

from sglang.srt.kv_canary.radix_cache_walker import walk_radix_cache_for_canary
from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache, TreeNode
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kv_canary.fixtures import DEFAULT_DEVICE, make_radix_cache
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=9, stage="extra-a", runner_config="1-gpu-small")
register_amd_ci(est_time=30, suite="extra-a-test-1-gpu-small-amd")


class TestSelfUnitRadixWalker(CustomTestCase):
    def setUp(self):
        self.device = DEFAULT_DEVICE

    def test_single_node_chain_positions_increase(self):
        """Verify a single radix chain emits increasing positions."""
        chain = [10, 20, 30, 40]
        cache = make_radix_cache([[], chain], device=self.device)
        result = walk_radix_cache_for_canary(radix_cache=cache)
        self.assertEqual(result.slot_indices.tolist(), chain)
        self.assertEqual(result.positions.tolist(), [0, 1, 2, 3])
        self.assertEqual(result.prev_slot_indices.tolist(), [-1, 10, 20, 30])

    def test_child_node_first_slot_prev_is_parent_last(self):
        """Verify child chains link their first slot to the parent tail."""
        parent = [7, 8]
        child = [9, 10]
        cache = make_radix_cache([[], parent, child], device=self.device)
        result = walk_radix_cache_for_canary(radix_cache=cache)
        self.assertEqual(result.slot_indices.tolist(), parent + child)
        self.assertEqual(result.prev_slot_indices.tolist()[len(parent)], parent[-1])

    def test_root_child_first_slot_prev_minus_one(self):
        """Verify root child chains use -1 as the initial previous slot."""
        cache = make_radix_cache([[], [42, 43]], device=self.device)
        result = walk_radix_cache_for_canary(radix_cache=cache)
        self.assertEqual(int(result.prev_slot_indices[0]), -1)

    def test_position_equals_depth_from_root(self):
        """Verify emitted positions match depth from the radix root."""
        cache = make_radix_cache([[], [1, 2], [3], [4, 5]], device=self.device)
        result = walk_radix_cache_for_canary(radix_cache=cache)
        self.assertEqual(result.positions.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(result.slot_indices.tolist(), [1, 2, 3, 4, 5])

    def test_walk_includes_locked_nodes_by_default(self):
        """Verify radix walking includes locked nodes by default."""
        cache = make_radix_cache([[], [1, 2], [3, 4]], device=self.device)
        locked_node = next(iter(cache.root_node.children.values()))
        locked_node.lock_ref = 1
        result = walk_radix_cache_for_canary(radix_cache=cache)
        self.assertEqual(result.slot_indices.tolist(), [1, 2, 3, 4])

    def test_walk_unlocked_only_skips_locked(self):
        """Verify unlocked-only radix walking skips locked nodes."""
        cache = make_radix_cache([[], [1, 2], [3, 4]], device=self.device)
        locked_node = next(iter(cache.root_node.children.values()))
        locked_node.lock_ref = 1
        result = walk_radix_cache_for_canary(radix_cache=cache, unlocked_only=True)
        self.assertEqual(result.slot_indices.tolist(), [3, 4])

    def test_walk_unlocked_only_uses_swa_full_lock_ref(self):
        """Verify SWA radix walking honors full-pool lock references."""
        cache = SWARadixCache.__new__(SWARadixCache)
        cache.device = self.device
        cache.page_size = 1
        cache.disable = False

        root = TreeNode()
        root.value = torch.tensor([], dtype=torch.int32, device=self.device)
        cache.root_node = root

        locked_child = TreeNode()
        locked_child.value = torch.tensor([1, 2], dtype=torch.int32, device=self.device)
        locked_child.parent = root
        locked_child.full_lock_ref = 1
        root.children[locked_child.id] = locked_child

        unlocked_child = TreeNode()
        unlocked_child.value = torch.tensor(
            [3, 4], dtype=torch.int32, device=self.device
        )
        unlocked_child.parent = root
        root.children[unlocked_child.id] = unlocked_child

        result = walk_radix_cache_for_canary(radix_cache=cache, unlocked_only=True)
        self.assertEqual(result.slot_indices.tolist(), [3, 4])

    def test_swa_resident_only_skips_tombstoned_nodes(self):
        """Verify SWA radix walking skips nodes whose SWA storage was evicted."""
        cache = SWARadixCache.__new__(SWARadixCache)
        cache.device = self.device
        cache.page_size = 1
        cache.disable = False

        root = TreeNode()
        root.value = torch.tensor([], dtype=torch.int32, device=self.device)
        cache.root_node = root

        tombstoned_child = TreeNode()
        tombstoned_child.value = torch.tensor(
            [1, 2], dtype=torch.int32, device=self.device
        )
        tombstoned_child.parent = root
        tombstoned_child.swa_tombstone = True
        root.children[tombstoned_child.id] = tombstoned_child

        resident_child = TreeNode()
        resident_child.value = torch.tensor(
            [3, 4], dtype=torch.int32, device=self.device
        )
        resident_child.parent = root
        root.children[resident_child.id] = resident_child

        result = walk_radix_cache_for_canary(
            radix_cache=cache,
            swa_resident_only=True,
        )
        self.assertEqual(result.slot_indices.tolist(), [3, 4])


if __name__ == "__main__":
    unittest.main()
