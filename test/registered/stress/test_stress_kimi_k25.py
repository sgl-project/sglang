import os
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.ci.ci_stress_utils import StressTestRunner
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, ModelLaunchSettings

# Register for CI - estimated 45 minutes
register_cuda_ci(est_time=2700, suite="stress")

KIMI_K25_MODEL_PATH = "moonshotai/Kimi-K2.5"


class TestStressKimiK25(unittest.TestCase):
    """Stress test for Kimi-K2.5.

    Kimi-K2.5 is a multimodal agentic model with reasoning capabilities.
    Sends 50K prompts over 45 minutes to validate stability.
    """

    def test_stress_kimi_k25(self):
        """Run stress test for Kimi-K2.5."""
        model = ModelLaunchSettings(
            KIMI_K25_MODEL_PATH,
            tp_size=8,
            extra_args=[
                "--trust-remote-code",
                "--tool-call-parser=kimi_k2",
                "--reasoning-parser=kimi_k2",
            ],
            variant="TP8",
        )

        runner = StressTestRunner(
            test_name="Kimi-K2.5 Stress Test",
            base_url=DEFAULT_URL_FOR_TEST,
            num_prompts=int(os.environ.get("NUM_PROMPTS", "50000")),
            duration_minutes=int(os.environ.get("DURATION_MINUTES", "45")),
        )

        try:
            success = runner.run_stress_test_for_model(
                model_path=model.model_path,
                random_input_len=4096,
                random_output_len=512,
                output_file="stress_test_kimi_k25.jsonl",
                server_args=model.extra_args,
            )

            self.assertTrue(success, f"Stress test failed for {model.model_path}")

        finally:
            runner.write_final_report()


if __name__ == "__main__":
    unittest.main()
