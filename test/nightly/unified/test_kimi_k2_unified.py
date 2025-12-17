"""Unified Kimi-K2-Thinking performance and accuracy tests using nightly_metrics.

This file replaces test_kimi_k2_thinking_perf.py and adds accuracy testing.
Configuration: TP=8 with tool-call-parser and reasoning-parser.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nightly_metrics import run_metrics

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, ModelLaunchSettings

# Registered to nightly-8-gpu-temp suite for testing
# This suite should be run with --timeout-per-file=12000 (200 minutes)
register_cuda_ci(est_time=12000, suite="nightly-8-gpu-temp", nightly=True)

KIMI_K2_THINKING_MODEL_PATH = "moonshotai/Kimi-K2-Thinking"


class TestKimiK2Unified(unittest.TestCase):
    """Unified test class for Kimi-K2-Thinking performance and accuracy.

    Single variant with TP=8 + tool/reasoning parsers.
    Runs BOTH:
    - Performance test (using NightlyBenchmarkRunner with extra_bench_args)
    - Accuracy test (using run_eval with mgsm_en)
    """

    def test_kimi_k2(self):
        """Run performance and accuracy for Kimi-K2-Thinking."""
        print("\n" + "=" * 80)
        print("RUNNING: TestKimiK2Unified.test_kimi_k2")
        print("=" * 80)

        variants = [
            ModelLaunchSettings(
                KIMI_K2_THINKING_MODEL_PATH,
                tp_size=8,
                extra_args=[
                    "--trust-remote-code",
                    "--tp=8",
                    "--tool-call-parser=kimi_k2",
                    "--reasoning-parser=kimi_k2",
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
            profile_dir="performance_profiles_kimi_k2_thinking",
            test_name="Kimi-K2-Thinking Unified",
            batch_sizes=[1, 1, 8, 16, 64],
            eval_name="mgsm_en",
        )


if __name__ == "__main__":
    unittest.main()
