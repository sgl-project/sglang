from __future__ import annotations

import unittest

import torch

from sglang.srt.kv_canary.plan_input import walk_radix_cache_for_canary
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache, TreeNode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.fixtures import DEFAULT_DEVICE, make_radix_cache
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="extra-a", runner_config="1-gpu-large")


class TestSelfUnitRadixWalker(CustomTestCase):
    def setUp(self):
        self.device = DEFAULT_DEVICE

    def test_single_node_chain_positions_increase(self):
        chain = [10, 20, 30, 40]
        cache = make_radix_cache([[], chain], device=self.device)
        slots, positions, prev_slots = walk_radix_cache_for_canary(radix_cache=cache)
        self.assertEqual(slots.tolist(), chain)
        self.assertEqual(positions.tolist(), [0, 1, 2, 3])
        self.assertEqual(prev_slots.tolist(), [-1, 10, 20, 30])

    def test_child_node_first_slot_prev_is_parent_last(self):
        parent = [7, 8]
        child = [9, 10]
        cache = make_radix_cache([[], parent, child], device=self.device)
        slots, positions, prev_slots = walk_radix_cache_for_canary(radix_cache=cache)
        self.assertEqual(slots.tolist(), parent + child)
        self.assertEqual(prev_slots.tolist()[len(parent)], parent[-1])

    def test_root_child_first_slot_prev_minus_one(self):
        cache = make_radix_cache([[], [42, 43]], device=self.device)
        _, _, prev_slots = walk_radix_cache_for_canary(radix_cache=cache)
        self.assertEqual(int(prev_slots[0]), -1)

    def test_position_equals_depth_from_root(self):
        cache = make_radix_cache([[], [1, 2], [3], [4, 5]], device=self.device)
        slots, positions, _ = walk_radix_cache_for_canary(radix_cache=cache)
        self.assertEqual(positions.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(slots.tolist(), [1, 2, 3, 4, 5])

    def test_walk_includes_locked_nodes_by_default(self):
        cache = make_radix_cache([[], [1, 2], [3, 4]], device=self.device)
        locked_node = next(iter(cache.root_node.children.values()))
        locked_node.lock_ref = 1
        slots, _, _ = walk_radix_cache_for_canary(radix_cache=cache)
        self.assertEqual(slots.tolist(), [1, 2, 3, 4])

    def test_walk_unlocked_only_skips_locked(self):
        cache = make_radix_cache([[], [1, 2], [3, 4]], device=self.device)
        locked_node = next(iter(cache.root_node.children.values()))
        locked_node.lock_ref = 1
        slots, _, _ = walk_radix_cache_for_canary(radix_cache=cache, unlocked_only=True)
        self.assertEqual(slots.tolist(), [3, 4])

    def test_walk_unlocked_only_uses_swa_full_lock_ref(self):
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

        slots, _, _ = walk_radix_cache_for_canary(radix_cache=cache, unlocked_only=True)
        self.assertEqual(slots.tolist(), [3, 4])

    def test_chunk_cache_returns_empty_sweep_entries(self):
        cache = ChunkCache.__new__(ChunkCache)

        slots, positions, prev_slots = walk_radix_cache_for_canary(radix_cache=cache)

        self.assertEqual(slots.tolist(), [])
        self.assertEqual(positions.tolist(), [])
        self.assertEqual(prev_slots.tolist(), [])


if __name__ == "__main__":
    unittest.main()
