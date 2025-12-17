"""Unified Qwen3-235B performance and accuracy tests using nightly_metrics.

This file replaces test_qwen3_235b_perf.py and adds accuracy testing.
Simple configuration: TP=8 only.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nightly_metrics import run_metrics

from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, ModelLaunchSettings

# Registered to nightly-8-gpu-temp suite for testing
# This suite should be run with --timeout-per-file=12000 (200 minutes)
# register_cuda_ci(est_time=12000, suite="nightly-8-gpu-temp", nightly=True)

QWEN3_235B_MODEL_PATH = "Qwen/Qwen3-235B-A22B-Instruct-2507"


class TestQwen3235BUnified(unittest.TestCase):
    """Unified test class for Qwen3-235B performance and accuracy.

    Single variant with simple TP=8 configuration.
    Runs BOTH:
    - Performance test (using NightlyBenchmarkRunner)
    - Accuracy test (using run_eval with mgsm_en)
    """

    def test_qwen3_235b(self):
        """Run performance and accuracy for Qwen3-235B."""
        print("\n" + "=" * 80)
        print("RUNNING: TestQwen3235BUnified.test_qwen3_235b")
        print("=" * 80)

        variants = [
            ModelLaunchSettings(
                QWEN3_235B_MODEL_PATH,
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
            profile_dir="performance_profiles_qwen3_235b",
            test_name="TestQwen3235BUnified",
            batch_sizes=[1, 1, 8, 16, 64],
            eval_name="mgsm_en",
        )

        # Check results
        self.assertTrue(
            result["all_passed"], f"Test failed. Results: {result['results']}"
        )

        # Print summary
        print("\n" + "=" * 60)
        print("Qwen3-235B Unified Test Results")
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
