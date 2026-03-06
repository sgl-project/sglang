import os
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.ci.ci_stress_utils import StressTestRunner
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, ModelLaunchSettings

# Register for CI - estimated 45 minutes
register_cuda_ci(est_time=2700, suite="stress")

QWEN35_MODEL_PATH = "Qwen/Qwen3.5-397B-A17B"


class TestStressQwen35(unittest.TestCase):
    """Stress test for Qwen3.5-397B-A17B.

    Qwen3.5 is a 397B MoE VLM with 17B active params.
    Sends 50K prompts over 45 minutes to validate stability.
    """

    def test_stress_qwen35(self):
        """Run stress test for Qwen3.5-397B-A17B."""
        model = ModelLaunchSettings(
            QWEN35_MODEL_PATH,
            tp_size=8,
            extra_args=[
                "--trust-remote-code",
                "--reasoning-parser=qwen3",
                "--tool-call-parser=qwen3_coder",
                "--mem-fraction-static=0.8",
            ],
            variant="TP8",
        )

        runner = StressTestRunner(
            test_name="Qwen3.5-397B-A17B Stress Test",
            base_url=DEFAULT_URL_FOR_TEST,
            num_prompts=int(os.environ.get("NUM_PROMPTS", "50000")),
            duration_minutes=int(os.environ.get("DURATION_MINUTES", "45")),
        )

        try:
            success = runner.run_stress_test_for_model(
                model_path=model.model_path,
                random_input_len=4096,
                random_output_len=512,
                output_file="stress_test_qwen35.jsonl",
                server_args=model.extra_args,
            )

            self.assertTrue(success, f"Stress test failed for {model.model_path}")

        finally:
            runner.write_final_report()


if __name__ == "__main__":
    unittest.main()
