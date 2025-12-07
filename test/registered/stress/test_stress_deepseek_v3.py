"""Stress test for DeepSeek-V3 model."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.ci.ci_stress_utils import StressTestBase

# Register for CI - estimated 30 minutes for throughput benchmarking
register_cuda_ci(est_time=1800, suite="stress")


class TestStressDeepSeekV3(StressTestBase):
    MODEL_PATH = "deepseek-ai/DeepSeek-V3"
    OUTPUT_FILE = "stress_test_deepseek_v3.jsonl"
    TEST_NAME = "DeepSeek-V3 Stress Test"

    def test_stress_deepseek_v3(self):
        self.run_stress_test()


if __name__ == "__main__":
    unittest.main()
