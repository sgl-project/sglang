"""Stress test for Qwen3-235B model."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.ci.ci_stress_utils import StressTestBase

# Register for CI - estimated 30 minutes for throughput benchmarking
register_cuda_ci(est_time=1800, suite="stress")


class TestStressQwen3235B(StressTestBase):
    MODEL_PATH = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    OUTPUT_FILE = "stress_test_qwen3_235b.jsonl"
    TEST_NAME = "Qwen3-235B Stress Test"

    def test_stress_qwen3_235b(self):
        self.run_stress_test()


if __name__ == "__main__":
    unittest.main()
