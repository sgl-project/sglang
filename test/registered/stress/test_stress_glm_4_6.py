"""Stress test for GLM-4.6 model."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.ci.ci_stress_utils import StressTestBase

# Register for CI - estimated 30 minutes for throughput benchmarking
register_cuda_ci(est_time=1800, suite="stress")


class TestStressGLM46(StressTestBase):
    MODEL_PATH = "zai-org/GLM-4.6"
    OUTPUT_FILE = "stress_test_glm_4_6.jsonl"
    TEST_NAME = "GLM-4.6 Stress Test"

    def test_stress_glm_4_6(self):
        self.run_stress_test()


if __name__ == "__main__":
    unittest.main()
