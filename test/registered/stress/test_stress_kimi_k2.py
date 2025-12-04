"""Stress test for Kimi-K2-Thinking model."""

import os
import unittest

from sglang.test.ci.ci_stress_utils import StressTestRunner
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST

MODEL_PATH = "moonshotai/Kimi-K2-Thinking"
RANDOM_INPUT_LEN = 16384
RANDOM_OUTPUT_LEN = 1024
OUTPUT_FILE = "stress_test_kimi_k2.jsonl"

# Register for CI - estimated 30 minutes for throughput benchmarking
# TEMP: Disabled for debugging - testing only Qwen3
# register_cuda_ci(est_time=1800, suite="stress")


class TestStressKimiK2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_prompts = int(os.environ.get("NUM_PROMPTS", "20000"))
        cls.duration_minutes = int(os.environ.get("DURATION_MINUTES", "5"))

        cls.runner = StressTestRunner(
            test_name="Kimi-K2-Thinking Stress Test",
            base_url=cls.base_url,
            num_prompts=cls.num_prompts,
            duration_minutes=cls.duration_minutes,
        )

    def test_stress_kimi_k2(self):
        try:
            success = self.runner.run_stress_test_for_model(
                model_path=self.model,
                random_input_len=RANDOM_INPUT_LEN,
                random_output_len=RANDOM_OUTPUT_LEN,
                output_file=OUTPUT_FILE,
                server_args=[
                    "--tp",
                    "8",
                    "--trust-remote-code",
                    "--tool-call-parser",
                    "kimi_k2",
                    "--reasoning-parser",
                    "kimi_k2",
                    "--mem-fraction-static",
                    "0.90",
                ],
            )

            self.assertTrue(success, f"Stress test failed for {self.model}")

        finally:
            self.runner.write_final_report()


if __name__ == "__main__":
    unittest.main()
