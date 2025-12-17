"""Unified Qwen3-235B performance and accuracy tests using nightly_metrics.

This file replaces test_qwen3_235b_perf.py and adds accuracy testing.
Simple configuration: TP=8 only.
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
        # run_metrics() handles summary printing and raises AssertionError on failure
        run_metrics(
            models=variants,
            run_perf=True,
            run_accuracy=True,
            is_vlm=False,
            base_url=DEFAULT_URL_FOR_TEST,
            profile_dir="performance_profiles_qwen3_235b",
            test_name="Qwen3-235B Unified",
            batch_sizes=[1, 1, 8, 16, 64],
            eval_name="mgsm_en",
        )


if __name__ == "__main__":
    unittest.main()
