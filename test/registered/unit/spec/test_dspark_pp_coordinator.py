"""Coordinated DSpark phases use aligned proposals and local physical KV slots."""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dspark_pp_coordinator_utils import launch_coordinator_test
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


class TestDSparkPPCoordinator(CustomTestCase):
    def test_tp2_pp2(self):
        launch_coordinator_test(self, tp_size=2, pp_size=2)

    def test_pp4(self):
        launch_coordinator_test(self, tp_size=1, pp_size=4)


if __name__ == "__main__":
    unittest.main()
