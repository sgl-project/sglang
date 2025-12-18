"""Unified MiniMax-M2 performance and accuracy tests using nightly_metrics.

This file replaces test_minimax_m2_perf.py and adds accuracy testing.
Configuration: TP=8 + EP=8 (expert parallelism) for MoE.
MiniMax-M2 is a 230B MoE model with 10B active params.
"""

import sys
import unittest
from pathlib import Path

# Add nightly directory to path for run_combined_tests import
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "nightly"))

from run_combined_tests import run_metrics

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, ModelLaunchSettings

# Registered to nightly-8-gpu-h200-basic suite
# This suite should be run with --timeout-per-file=12000 (200 minutes)
register_cuda_ci(est_time=12000, suite="nightly-8-gpu-h200-basic", nightly=True)
register_cuda_ci(est_time=12000, suite="nightly-8-gpu-b200-basic", nightly=True)

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
        print("\n" + "=" * 80)
        print("RUNNING: TestMiniMaxM2Unified.test_minimax_m2")
        print("=" * 80)

        variants = [
            ModelLaunchSettings(
                MINIMAX_M2_MODEL_PATH,
                tp_size=8,
                extra_args=[
                    "--trust-remote-code",
                    "--tp",
                    "8",
                    "--ep",
                    "8",
                    "--model-loader-extra-config",
                    '{"enable_multithread_load": true}',
                ],
            ),
        ]

        # Run both performance and accuracy
        # run_metrics() handles summary printing and raises AssertionError on failure
        run_metrics(
            models=variants,
            run_perf=True,
            run_accuracy=True,
            is_vlm=False,
            base_url=DEFAULT_URL_FOR_TEST,
            profile_dir="performance_profiles_minimax_m2",
            test_name="MiniMax-M2 Unified",
            batch_sizes=[1, 1, 8, 16, 64],
            eval_name="mgsm_en",
        )


if __name__ == "__main__":
    unittest.main()
