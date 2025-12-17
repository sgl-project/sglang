"""Unified DeepSeek V3.2 performance and accuracy tests using nightly_metrics.

This file combines:
- test_deepseek_v32_perf.py (performance with 4 variants)
- test_deepseek_v32_tp.py (accuracy for pure_tp and partial_tp variants)

It uses nightly_metrics.run_metrics() to run both performance and accuracy
for standard model configurations.

Custom backend tests remain separate:
- test_deepseek_v32_nsabackend.py (NSA backend variants with custom eval)
- test_deepseek_v32_gpqa.py (GPQA evaluation with thinking mode)
"""

import unittest

from nightly_metrics import run_metrics

from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, ModelLaunchSettings

# NOTE: This test is NOT registered via register_cuda_ci() decorator.
# It must be called directly from the YML workflow with appropriate timeout (180min).
# This is because running 5 variants with both perf + accuracy takes ~100+ minutes.

DEEPSEEK_V32_MODEL_PATH = "deepseek-ai/DeepSeek-V3.2-Exp"


class TestDeepseekV32Unified(unittest.TestCase):
    """Unified test class for DeepSeek V3.2 performance and accuracy.

    Tests 5 variants:
    - basic: Standard TP=8 + DP=8 with dp-attention
    - mtp: Basic + EAGLE speculative decoding
    - nsa: NSA backend with flashmla + DP=8
    - pure_tp: Pure TP=8 without DP
    - partial_tp: Partial TP=8 + DP=4 (hybrid parallelism)

    Each variant runs BOTH:
    - Performance test (using NightlyBenchmarkRunner)
    - Accuracy test (using run_eval with mgsm_en)
    """

    def test_deepseek_v32_all_variants(self):
        """Run performance and accuracy for all DeepSeek V3.2 variants."""

        # Define all model variants
        variants = [
            # Variant: "basic" (from test_deepseek_v32_perf.py)
            # Standard TP=8 + DP=8 with dp-attention
            ModelLaunchSettings(
                DEEPSEEK_V32_MODEL_PATH,
                tp_size=8,
                extra_args=[
                    "--trust-remote-code",
                    "--tp=8",
                    "--dp=8",
                    "--enable-dp-attention",
                    "--model-loader-extra-config",
                    '{"enable_multithread_load": true}',
                ],
            ),
            # Variant: "mtp" (from test_deepseek_v32_perf.py)
            # Basic + EAGLE speculative decoding
            ModelLaunchSettings(
                DEEPSEEK_V32_MODEL_PATH,
                tp_size=8,
                extra_args=[
                    "--trust-remote-code",
                    "--tp=8",
                    "--dp=8",
                    "--enable-dp-attention",
                    "--speculative-algorithm=EAGLE",
                    "--speculative-num-steps=3",
                    "--speculative-eagle-topk=1",
                    "--speculative-num-draft-tokens=4",
                    "--mem-frac=0.7",
                    "--model-loader-extra-config",
                    '{"enable_multithread_load": true}',
                ],
            ),
            # Variant: "nsa" (from test_deepseek_v32_perf.py)
            # NSA backend with flashmla + DP=8
            ModelLaunchSettings(
                DEEPSEEK_V32_MODEL_PATH,
                tp_size=8,
                extra_args=[
                    "--trust-remote-code",
                    "--tp=8",
                    "--dp=8",
                    "--enable-dp-attention",
                    "--attention-backend=nsa",
                    "--nsa-prefill-backend=flashmla_sparse",
                    "--nsa-decode-backend=flashmla_kv",
                    "--model-loader-extra-config",
                    '{"enable_multithread_load": true}',
                ],
            ),
            # Variant: "pure_tp" (from test_deepseek_v32_perf.py + test_deepseek_v32_tp.py)
            # Pure TP=8 without DP (NSA backend with flashmla)
            ModelLaunchSettings(
                DEEPSEEK_V32_MODEL_PATH,
                tp_size=8,
                extra_args=[
                    "--trust-remote-code",
                    "--tp=8",
                    "--attention-backend=nsa",
                    "--nsa-prefill-backend=flashmla_sparse",
                    "--nsa-decode-backend=flashmla_kv",
                    "--model-loader-extra-config",
                    '{"enable_multithread_load": true}',
                ],
            ),
            # Variant: "partial_tp" (from test_deepseek_v32_tp.py)
            # Partial TP=8 + DP=4 with dp-attention (hybrid parallelism)
            ModelLaunchSettings(
                DEEPSEEK_V32_MODEL_PATH,
                tp_size=8,
                extra_args=[
                    "--trust-remote-code",
                    "--tp=8",
                    "--dp=4",
                    "--enable-dp-attention",
                    "--attention-backend=nsa",
                    "--nsa-prefill-backend=flashmla_sparse",
                    "--nsa-decode-backend=flashmla_kv",
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
            profile_dir="performance_profiles_deepseek_v32",
            test_name="TestDeepseekV32Unified",
            batch_sizes=[1, 8, 16, 64],
            eval_name="mgsm_en",
        )

        # Check results
        self.assertTrue(
            result["all_passed"], f"Some variants failed. Results: {result['results']}"
        )

        # Print summary
        print("\n" + "=" * 60)
        print("DeepSeek V3.2 Unified Test Results")
        print("=" * 60)
        for i, model_result in enumerate(result["results"]):
            variant_name = ["basic", "mtp", "nsa", "pure_tp", "partial_tp"][i]
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
