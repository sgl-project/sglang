"""Unit tests for the elastic staged-grow planner — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.managers.tokenizer_manager import elastic_stage_plan
from sglang.test.test_utils import CustomTestCase

LAUNCH = 4


class TestElasticStagePlan(CustomTestCase):
    def test_only_a_grow_crossing_the_launch_width_is_staged(self):
        """It stops at the launch width; partial regrows, shrinks, no-ops go direct."""
        for current in range(1, 9):
            for target in range(1, 9):
                with self.subTest(current=current, target=target):
                    staged = current < LAUNCH < target
                    self.assertEqual(
                        elastic_stage_plan(current, LAUNCH, target),
                        [LAUNCH, target] if staged else [target],
                    )


if __name__ == "__main__":
    unittest.main()
