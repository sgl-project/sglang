"""Unified GLM-4.6 performance and accuracy tests using nightly_metrics.

This file replaces test_glm_4_6_perf.py and adds accuracy testing.
Simple configuration: TP=8 only.
GLM-4.6 is a 357B MoE model.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nightly_metrics import run_metrics

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, ModelLaunchSettings

register_cuda_ci(est_time=12000, suite="nightly-8-gpu-temp", nightly=True)

GLM_4_6_MODEL_PATH = "zai-org/GLM-4.6"


class TestGLM46Unified(unittest.TestCase):
    """Unified test class for GLM-4.6 performance and accuracy.

    Single variant with simple TP=8 configuration.
    GLM-4.6 is a 357B MoE model.
    Runs BOTH:
    - Performance test (using NightlyBenchmarkRunner)
    - Accuracy test (using run_eval with mgsm_en)
    """

    def test_glm_46(self):
        """Run performance and accuracy for GLM-4.6."""
        print("\n" + "=" * 80)
        print("RUNNING: TestGLM46Unified.test_glm_46")
        print("=" * 80)

        variants = [
            ModelLaunchSettings(
                GLM_4_6_MODEL_PATH,
                tp_size=8,
                extra_args=[
                    "--trust-remote-code",
                    "--tp=8",
                ],
            ),
        ]

        # Run both performance and accuracy
        result = run_metrics(
            models=variants,
            run_perf=True,
            run_accuracy=True,
            is_vlm=False,
            base_url=DEFAULT_URL_FOR_TEST,
            profile_dir="performance_profiles_glm_4_6",
            test_name="TestGLM46Unified",
            batch_sizes=[1, 1, 8, 16, 64],
            eval_name="mgsm_en",
        )

        # Check results
        self.assertTrue(
            result["all_passed"], f"Test failed. Results: {result['results']}"
        )

        # Print summary
        print("\n" + "=" * 60)
        print("GLM-4.6 Unified Test Results")
        print("=" * 60)
        model_result = result["results"][0]
        print(f"Performance: {'✓' if model_result['perf_passed'] else '✗'}")
        print(f"Accuracy: {'✓' if model_result['accuracy_passed'] else '✗'}")
        if model_result["accuracy_metrics"]:
            print(f"Score: {model_result['accuracy_metrics'].get('score', 'N/A')}")
        if model_result["errors"]:
            print(f"Errors: {model_result['errors']}")


if __name__ == "__main__":
    unittest.main()
