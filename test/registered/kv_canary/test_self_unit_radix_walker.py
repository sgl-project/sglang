from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from _fixtures import CPU_DEVICE, make_radix_cache  # noqa: E402

from sglang.srt.kv_canary.plan_input import walk_radix_cache_for_canary  # noqa: E402
from sglang.test.ci.ci_register import register_cuda_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cuda_ci(est_time=30, stage="extra-a", runner_config="1-gpu-large")


class TestSelfUnitRadixWalker(CustomTestCase):
    def setUp(self):
        self.device = CPU_DEVICE

    def test_empty_radix_returns_zero_extras(self):
        cache = make_radix_cache([[]], device=self.device)
        slots, positions, prev_slots = walk_radix_cache_for_canary(radix_cache=cache)
        self.assertEqual(slots.numel(), 0)
        self.assertEqual(positions.numel(), 0)
        self.assertEqual(prev_slots.numel(), 0)

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

    def test_skips_slots_owned_by_running_reqs(self):
        cache = make_radix_cache([[], [100, 101]], device=self.device)
        slots, _, _ = walk_radix_cache_for_canary(radix_cache=cache)
        self.assertEqual(set(slots.tolist()), {100, 101})
        self.assertEqual(slots.dtype, torch.int32)


if __name__ == "__main__":
    unittest.main()
