"""Unit tests for the elastic staged-grow planner — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.managers.tokenizer_manager import elastic_stage_plan
from sglang.test.test_utils import CustomTestCase

LAUNCH = 4


class TestElasticStagePlan(CustomTestCase):
    def test_partial_regrow_below_the_launch_width_is_staged(self):
        # Direct is what a reader expects and what MC17 measured failing: the joiner
        # rebuilds its expert map against a width the survivors do not share.
        self.assertEqual(elastic_stage_plan(2, LAUNCH, 3), [LAUNCH, 3])
        self.assertEqual(elastic_stage_plan(1, LAUNCH, 2), [LAUNCH, 2])

    def test_grow_back_to_the_launch_width_is_direct(self):
        self.assertEqual(elastic_stage_plan(2, LAUNCH, LAUNCH), [LAUNCH])

    def test_grow_above_the_launch_width_from_below_is_staged(self):
        """Mixing retired slots and fresh slots in one grow is rejected downstream."""
        self.assertEqual(elastic_stage_plan(2, LAUNCH, 5), [LAUNCH, 5])
        self.assertEqual(elastic_stage_plan(3, LAUNCH, 8), [LAUNCH, 8])

    def test_grow_above_the_launch_width_from_at_it_is_direct(self):
        # Nothing is retired, so every new slot is an append.
        self.assertEqual(elastic_stage_plan(LAUNCH, LAUNCH, 6), [6])
        self.assertEqual(elastic_stage_plan(5, LAUNCH, 6), [6])

    def test_shrink_is_never_staged(self):
        self.assertEqual(elastic_stage_plan(LAUNCH, LAUNCH, 2), [2])
        self.assertEqual(elastic_stage_plan(5, LAUNCH, 3), [3])
        self.assertEqual(elastic_stage_plan(2, LAUNCH, 1), [1])

    def test_noop_target_is_returned_as_itself(self):
        self.assertEqual(elastic_stage_plan(3, LAUNCH, 3), [3])

    def test_a_descending_plan_is_legal(self):
        """Stage 2 below stage 1: the partial regrow recovers, then retires again."""
        self.assertEqual(elastic_stage_plan(2, LAUNCH, 3), [LAUNCH, 3])

    def test_every_plan_ends_on_the_requested_target(self):
        for current in range(1, 9):
            for target in range(1, 9):
                with self.subTest(current=current, target=target):
                    plan = elastic_stage_plan(current, LAUNCH, target)
                    self.assertTrue(plan, "plan must not be empty")
                    self.assertEqual(plan[-1], target)
                    self.assertLessEqual(len(plan), 2)

    def test_staging_happens_exactly_for_a_grow_starting_below_the_launch_width(self):
        for current in range(1, 9):
            for target in range(1, 9):
                with self.subTest(current=current, target=target):
                    # Any grow that starts below the launch width, except one landing
                    # exactly on it, which needs no intermediate.
                    staged = current < LAUNCH and current < target and target != LAUNCH
                    plan = elastic_stage_plan(current, LAUNCH, target)
                    self.assertEqual(
                        len(plan) == 2,
                        staged,
                        f"{current}->{target} staged={len(plan) == 2} want={staged}",
                    )
                    if len(plan) == 2:
                        self.assertEqual(plan[0], LAUNCH)


if __name__ == "__main__":
    unittest.main()
