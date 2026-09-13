"""Stress test for GLM-4.6 model."""

import os
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.ci.ci_stress_utils import StressTestRunner
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST

MODEL_PATH = "zai-org/GLM-4.6"
RANDOM_INPUT_LEN = 4096
RANDOM_OUTPUT_LEN = 512
OUTPUT_FILE = "stress_test_glm_4_6.jsonl"

# Register for CI - estimated 45 minutes
register_cuda_ci(est_time=2700, suite="stress")


class TestStressGLM46(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_prompts = int(os.environ.get("NUM_PROMPTS", "50000"))
        cls.duration_minutes = int(os.environ.get("DURATION_MINUTES", "45"))

        cls.runner = StressTestRunner(
            test_name="GLM-4.6 Stress Test",
            base_url=cls.base_url,
            num_prompts=cls.num_prompts,
            duration_minutes=cls.duration_minutes,
        )

    def test_stress_glm_4_6(self):
        try:
            success = self.runner.run_stress_test_for_model(
                model_path=self.model,
                random_input_len=RANDOM_INPUT_LEN,
                random_output_len=RANDOM_OUTPUT_LEN,
                output_file=OUTPUT_FILE,
                server_args=["--tp", "8", "--trust-remote-code"],
            )

            self.assertTrue(success, f"Stress test failed for {self.model}")

        finally:
            self.runner.write_final_report()


if __name__ == "__main__":
    unittest.main()
