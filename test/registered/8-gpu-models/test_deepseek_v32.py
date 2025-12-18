"""Unified DeepSeek V3.2 performance and accuracy tests using run_combined_tests.

This file tests the 4 variants from test_deepseek_v32_perf.py with both
performance and accuracy tests.

Custom backend tests remain separate:
- test_deepseek_v32_nsabackend.py (NSA backend variants with custom eval)
- test_deepseek_v32_gpqa.py (GPQA evaluation with thinking mode)
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
# This suite should be run with --timeout-per-file=8000 (133 minutes)
# because each test runs 4 variants with both perf + accuracy
register_cuda_ci(est_time=8000, suite="nightly-8-gpu-h200-basic", nightly=True)

DEEPSEEK_V32_MODEL_PATH = "deepseek-ai/DeepSeek-V3.2-Exp"


class TestDeepseekV32Unified(unittest.TestCase):
    """Unified test class for DeepSeek V3.2 performance and accuracy.

    Tests 4 variants (matching test_deepseek_v32_perf.py on main):
    - dp: Standard TP=8 + DP=8 with dp-attention
    - dp+mtp: DP + EAGLE speculative decoding
    - tp: Pure TP=8 only
    - tp+mtp: Pure TP=8 + EAGLE speculative decoding

    Each variant runs BOTH:
    - Performance test (using NightlyBenchmarkRunner)
    - Accuracy test (using run_eval with mgsm_en)
    """

    def test_deepseek_v32_all_variants(self):
        """Run performance and accuracy for all DeepSeek V3.2 variants."""
        print("\n" + "=" * 80)
        print("RUNNING: TestDeepseekV32Unified.test_deepseek_v32_all_variants")
        print("=" * 80)

        # Define all model variants (matching test_deepseek_v32_perf.py on main)
        variants = [
            # Variant: "dp"
            # Standard TP=8 + DP=8 with dp-attention
            ModelLaunchSettings(
                DEEPSEEK_V32_MODEL_PATH,
                tp_size=8,
                extra_args=[
                    "--trust-remote-code",
                    "--tp",
                    "8",
                    "--dp",
                    "8",
                    "--enable-dp-attention",
                    "--model-loader-extra-config",
                    '{"enable_multithread_load": true}',
                ],
            ),
            # Variant: "dp+mtp"
            # DP + EAGLE speculative decoding
            ModelLaunchSettings(
                DEEPSEEK_V32_MODEL_PATH,
                tp_size=8,
                extra_args=[
                    "--trust-remote-code",
                    "--tp",
                    "8",
                    "--dp",
                    "8",
                    "--enable-dp-attention",
                    "--speculative-algorithm",
                    "EAGLE",
                    "--speculative-num-steps",
                    "3",
                    "--speculative-eagle-topk",
                    "1",
                    "--speculative-num-draft-tokens",
                    "4",
                    "--mem-frac",
                    "0.7",
                    "--model-loader-extra-config",
                    '{"enable_multithread_load": true}',
                ],
            ),
            # Variant: "tp"
            # Pure TP=8 only
            ModelLaunchSettings(
                DEEPSEEK_V32_MODEL_PATH,
                tp_size=8,
                extra_args=[
                    "--trust-remote-code",
                    "--tp",
                    "8",
                    "--model-loader-extra-config",
                    '{"enable_multithread_load": true}',
                ],
            ),
            # Variant: "tp+mtp"
            # Pure TP=8 + EAGLE speculative decoding
            ModelLaunchSettings(
                DEEPSEEK_V32_MODEL_PATH,
                tp_size=8,
                extra_args=[
                    "--trust-remote-code",
                    "--tp",
                    "8",
                    "--speculative-algorithm",
                    "EAGLE",
                    "--speculative-num-steps",
                    "3",
                    "--speculative-eagle-topk",
                    "1",
                    "--speculative-num-draft-tokens",
                    "4",
                    "--mem-frac",
                    "0.7",
                    "--model-loader-extra-config",
                    '{"enable_multithread_load": true}',
                ],
            ),
        ]

        # Run both performance and accuracy for all variants
        # run_metrics() handles summary printing and raises AssertionError on failure
        run_metrics(
            models=variants,
            run_perf=True,
            run_accuracy=True,
            is_vlm=False,
            base_url=DEFAULT_URL_FOR_TEST,
            profile_dir="performance_profiles_deepseek_v32",
            test_name="DeepSeek-V3.2 Unified",
            batch_sizes=[1, 8, 16, 64],
            eval_name="mgsm_en",
        )


if __name__ == "__main__":
    unittest.main()
