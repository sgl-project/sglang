"""Reject incompatible peer strides before a C2 state transfer is issued."""

import unittest

from sglang.srt.disaggregation.utils import validate_dsv41_c2_state_layout
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestC2Layout(unittest.TestCase):
    def test_matching_layers(self):
        validate_dsv41_c2_state_layout([32768, 32768, 32768], [32768, 32768, 32768])

    def test_different_ring_strides(self):
        with self.assertRaisesRegex(ValueError, "matching C2 state layouts"):
            validate_dsv41_c2_state_layout([8192] * 3, [32768] * 3)


if __name__ == "__main__":
    unittest.main()
