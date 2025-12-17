"""Unified MiniMax-M2 performance and accuracy tests using nightly_metrics.

This file replaces test_minimax_m2_perf.py and adds accuracy testing.
Configuration: TP=8 + EP=8 (expert parallelism) for MoE.
MiniMax-M2 is a 230B MoE model with 10B active params.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nightly_metrics import run_metrics

from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, ModelLaunchSettings

# NOTE: This test is NOT registered via register_cuda_ci() decorator.
# It must be called directly from the YML workflow with appropriate timeout (180min).

MINIMAX_M2_MODEL_PATH = "MiniMaxAI/MiniMax-M2"


class TestMiniMaxM2Unified(unittest.TestCase):
    """Unified test class for MiniMax-M2 performance and accuracy.

    Single variant with TP=8 + EP=8 configuration.
    MiniMax-M2 is a 230B MoE model with 10B active params.
    Runs BOTH:
    - Performance test (using NightlyBenchmarkRunner with extra_bench_args)
    - Accuracy test (using run_eval with mgsm_en)
    """

    def test_minimax_m2(self):
        """Run performance and accuracy for MiniMax-M2."""

        variants = [
            ModelLaunchSettings(
                MINIMAX_M2_MODEL_PATH,
                tp_size=8,
                extra_args=[
                    "--trust-remote-code",
                    "--tp=8",
                    "--ep=8",
                    "--model-loader-extra-config",
                    '{"enable_multithread_load": true}',
                ],
            ),
        ]

        # Run both performance and accuracy
        # Note: Original test has extra_bench_args=["--trust-remote-code"]
        # but this is already in extra_args, so it should be passed through
        result = run_metrics(
            models=variants,
            run_perf=True,
            run_accuracy=True,
            is_vlm=False,
            base_url=DEFAULT_URL_FOR_TEST,
            profile_dir="performance_profiles_minimax_m2",
            test_name="TestMiniMaxM2Unified",
            batch_sizes=[1, 1, 8, 16, 64],
            eval_name="mgsm_en",
        )

        # Check results
        self.assertTrue(
            result["all_passed"], f"Test failed. Results: {result['results']}"
        )

        # Print summary
        print("\n" + "=" * 60)
        print("MiniMax-M2 Unified Test Results")
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
