"""Unified DeepSeek-V3.1 performance and accuracy tests using nightly_metrics.

This file replaces test_deepseek_v31_perf.py and adds accuracy testing.
Two variants: basic (TP=8) and mtp (TP=8 + EAGLE speculative decoding).
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nightly_metrics import run_metrics

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, ModelLaunchSettings

register_cuda_ci(est_time=12000, suite="nightly-8-gpu-temp-b200", nightly=True)

DEEPSEEK_V31_MODEL_PATH = "deepseek-ai/DeepSeek-V3.1"


class TestDeepseekV31Unified(unittest.TestCase):
    """Unified test class for DeepSeek-V3.1 performance and accuracy.

    Two variants:
    - basic: Standard TP=8
    - mtp: TP=8 + EAGLE speculative decoding

    Each variant runs BOTH:
    - Performance test (using NightlyBenchmarkRunner)
    - Accuracy test (using run_eval with mgsm_en)
    """

    def test_deepseek_v31_all_variants(self):
        """Run performance and accuracy for all DeepSeek-V3.1 variants."""
        print("\n" + "=" * 80)
        print("RUNNING: TestDeepseekV31Unified.test_deepseek_v31_all_variants")
        print("=" * 80)

        variants = [
            # Variant: "basic" (from test_deepseek_v31_perf.py)
            # Standard TP=8
            ModelLaunchSettings(
                DEEPSEEK_V31_MODEL_PATH,
                tp_size=8,
                extra_args=[
                    "--trust-remote-code",
                    "--tp=8",
                    "--model-loader-extra-config",
                    '{"enable_multithread_load": true}',
                ],
            ),
            # Variant: "mtp" (from test_deepseek_v31_perf.py)
            # TP=8 + EAGLE speculative decoding
            ModelLaunchSettings(
                DEEPSEEK_V31_MODEL_PATH,
                tp_size=8,
                extra_args=[
                    "--trust-remote-code",
                    "--tp=8",
                    "--speculative-algorithm=EAGLE",
                    "--speculative-num-steps=3",
                    "--speculative-eagle-topk=1",
                    "--speculative-num-draft-tokens=4",
                    "--mem-frac=0.7",
                    "--model-loader-extra-config",
                    '{"enable_multithread_load": true}',
                ],
            ),
        ]

        # Run both performance and accuracy for all variants
        result = run_metrics(
            models=variants,
            run_perf=True,
            run_accuracy=True,
            is_vlm=False,
            base_url=DEFAULT_URL_FOR_TEST,
            profile_dir="performance_profiles_deepseek_v31",
            test_name="TestDeepseekV31Unified",
            batch_sizes=[1, 1, 8, 16, 64],
            eval_name="mgsm_en",
        )

        # Check results
        self.assertTrue(
            result["all_passed"], f"Some variants failed. Results: {result['results']}"
        )

        # Print summary
        print("\n" + "=" * 60)
        print("DeepSeek-V3.1 Unified Test Results")
        print("=" * 60)
        for i, model_result in enumerate(result["results"]):
            variant_name = ["basic", "mtp"][i]
            print(f"\nVariant: {variant_name}")
            print(f"  Performance: {'✓' if model_result['perf_passed'] else '✗'}")
            print(f"  Accuracy: {'✓' if model_result['accuracy_passed'] else '✗'}")
            if model_result["accuracy_metrics"]:
                print(
                    f"  Score: {model_result['accuracy_metrics'].get('score', 'N/A')}"
                )
            if model_result["errors"]:
                print(f"  Errors: {model_result['errors']}")


if __name__ == "__main__":
    unittest.main()
