"""CPU-only stable-slot and validation checks for hot expert-set replacement."""

import random
import unittest

from sglang.srt.eplb.expert_swap import plan_expert_swap
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestExpertSwap(CustomTestCase):
    def test_retained_experts_keep_their_slots(self):
        plan = plan_expert_swap([[7, 1, 5, 3]], [[5, 6, 7, 0]], 8)
        self.assertEqual(plan.physical_to_logical, ((7, 6, 5, 0),))
        self.assertEqual(
            [(c.slot, c.old_expert, c.new_expert) for c in plan.changes],
            [(1, 1, 6), (3, 3, 0)],
        )

    def test_reordered_same_set_needs_no_weight_copy(self):
        plan = plan_expert_swap([[7, 1, 5, 3]], [[3, 5, 1, 7]], 8)
        self.assertEqual(plan.physical_to_logical, ((7, 1, 5, 3),))
        self.assertEqual(plan.changes, ())

    def test_layer_changes_are_independent(self):
        plan = plan_expert_swap([[0, 2], [7, 3]], [[1, 2], [7, 6]], 8)
        self.assertEqual(plan.physical_to_logical, ((1, 2), (7, 6)))
        self.assertEqual([c.layer for c in plan.changes], [0, 1])

    def test_rejects_capacity_layer_and_id_errors(self):
        for target in ([[0]], [[0, 1], [2, 3]], [[0, 0]], [[0, 8]], [[True, 1]], []):
            with self.subTest(target=target), self.assertRaises(ValueError):
                plan_expert_swap([[0, 1]], target, 8)
        with self.assertRaises(ValueError):
            plan_expert_swap([[0, 0]], [[0, 1]], 8)

    def test_random_sets_preserve_slots_and_change_only_the_difference(self):
        rng = random.Random(417)
        for _ in range(100):
            before = rng.sample(range(32), 16)
            requested = rng.sample(range(32), 16)
            plan = plan_expert_swap([before], [requested], 32)
            actual = plan.physical_to_logical[0]
            self.assertEqual(set(actual), set(requested))
            self.assertEqual(len(plan.changes), len(set(requested) - set(before)))
            for slot, expert in enumerate(before):
                if expert in requested:
                    self.assertEqual(actual[slot], expert)
            for change in plan.changes:
                self.assertEqual(before[change.slot], change.old_expert)
                self.assertEqual(actual[change.slot], change.new_expert)


if __name__ == "__main__":
    unittest.main()
