"""Stress test for Kimi-K2-Thinking model."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.ci.ci_stress_utils import StressTestBase

# Register for CI - estimated 30 minutes for throughput benchmarking
register_cuda_ci(est_time=1800, suite="stress")


class TestStressKimiK2(StressTestBase):
    MODEL_PATH = "moonshotai/Kimi-K2-Thinking"
    OUTPUT_FILE = "stress_test_kimi_k2.jsonl"
    TEST_NAME = "Kimi-K2-Thinking Stress Test"
    SERVER_ARGS = [
        "--tp",
        "8",
        "--trust-remote-code",
        "--tool-call-parser",
        "kimi_k2",
        "--reasoning-parser",
        "kimi_k2",
        "--mem-fraction-static",
        "0.90",
    ]

    def test_stress_kimi_k2(self):
        self.run_stress_test()


if __name__ == "__main__":
    unittest.main()
