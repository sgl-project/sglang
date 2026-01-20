import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings, is_blackwell_system

register_cuda_ci(est_time=5400, suite="nightly-8-gpu-common", nightly=True)

DEEPSEEK_V32_MODEL_PATH = "deepseek-ai/DeepSeek-V3.2"

BASE_ARGS = [
    "--trust-remote-code",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
]

DP_ARGS = [
    "--tp=8",
    "--dp=8",
    "--enable-dp-attention",
]

# Accuracy thresholds
GSM8K_BASELINE = 0.935
GPQA_BASELINE = 0.835


class TestDeepseekV32(unittest.TestCase):
    """Unified test class for DeepSeek V3.2 performance and accuracy.

    Tests multiple variants with both performance and accuracy tests:
    - dp: Standard TP=8 + DP=8 with dp-attention
    - dp+mtp: DP + EAGLE speculative decoding
    - tp: Pure TP=8 only
    - tp+mtp: Pure TP=8 + EAGLE speculative decoding
    """

    def test_deepseek_v32_all_variants(self):
        """Run performance and accuracy for all DeepSeek V3.2 variants."""
        TP_ARGS = [
            "--tp=8",
        ]
        MTP_ARGS = [
            "--speculative-algorithm=EAGLE",
            "--speculative-num-steps=3",
            "--speculative-eagle-topk=1",
            "--speculative-num-draft-tokens=4",
            "--mem-frac=0.7",
        ]
        variants = [
            # Variant: "dp" - Standard TP=8 + DP=8 with dp-attention
            ModelLaunchSettings(
                DEEPSEEK_V32_MODEL_PATH,
                tp_size=8,
                extra_args=BASE_ARGS + DP_ARGS,
                variant="DP8",
            ),
            # Variant: "dp+mtp" - DP + EAGLE speculative decoding
            ModelLaunchSettings(
                DEEPSEEK_V32_MODEL_PATH,
                tp_size=8,
                extra_args=BASE_ARGS + DP_ARGS + MTP_ARGS,
                variant="DP8+MTP",
            ),
            # Variant: "tp" - Pure TP=8 only
            ModelLaunchSettings(
                DEEPSEEK_V32_MODEL_PATH,
                tp_size=8,
                extra_args=BASE_ARGS + TP_ARGS,
                variant="TP8",
            ),
            # Variant: "tp+mtp" - Pure TP=8 + EAGLE speculative decoding
            ModelLaunchSettings(
                DEEPSEEK_V32_MODEL_PATH,
                tp_size=8,
                extra_args=BASE_ARGS + TP_ARGS + MTP_ARGS,
                variant="TP8+MTP",
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="DeepSeek-V3.2 Unified",
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k", baseline_accuracy=GSM8K_BASELINE
            ),
            performance_params=PerformanceTestParams(
                batch_sizes=[1, 8, 16, 64],
                profile_dir="performance_profiles_deepseek_v32",
            ),
        )

    @unittest.skipIf(is_blackwell_system(), "Requires H200 system")
    def test_deepseek_v32_nsa_backends(self):
        """Test NSA attention backend variants (H200 only).

        Tests three NSA backend configurations:
        - flashmla: flashmla_sparse prefill + flashmla_kv decode
        - fa3: FA3 prefill + FA3 decode
        - fp8kvcache: default backends with FP8 KV cache
        """
        NSA_FLASHMLA_ARGS = [
            "--attention-backend=nsa",
            "--nsa-prefill-backend=flashmla_sparse",
            "--nsa-decode-backend=flashmla_kv",
        ]

        NSA_FA3_ARGS = [
            "--attention-backend=nsa",
            "--nsa-prefill-backend=fa3",
            "--nsa-decode-backend=fa3",
        ]

        NSA_FP8KV_ARGS = [
            "--attention-backend=nsa",
            "--kv-cache-dtype=fp8_e4m3",
        ]

        nsa_variants = [
            # flashmla backend
            ModelLaunchSettings(
                DEEPSEEK_V32_MODEL_PATH,
                tp_size=8,
                extra_args=BASE_ARGS + DP_ARGS + NSA_FLASHMLA_ARGS,
            ),
            # fa3 backend
            ModelLaunchSettings(
                DEEPSEEK_V32_MODEL_PATH,
                tp_size=8,
                extra_args=BASE_ARGS + DP_ARGS + NSA_FA3_ARGS,
            ),
            # fp8 kv cache
            ModelLaunchSettings(
                DEEPSEEK_V32_MODEL_PATH,
                tp_size=8,
                extra_args=BASE_ARGS + DP_ARGS + NSA_FP8KV_ARGS,
            ),
        ]

        run_combined_tests(
            models=nsa_variants,
            test_name="DeepSeek-V3.2 NSA Backends",
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k", baseline_accuracy=GSM8K_BASELINE
            ),
            performance_params=None,
        )

    @unittest.skipIf(
        not is_blackwell_system(),
        "Hardware agnostic - just using B200 for efficiency reasons",
    )
    def test_deepseek_v32_b200(self):
        """Test DeepSeek V3.2 with GPQA evaluation using thinking mode (B200 only).

        This test runs GPQA evaluation with the reasoning parser enabled.
        """
        B200_REASONING_ARGS = [
            "--tool-call-parser=deepseekv32",
            "--reasoning-parser=deepseek-v3",
        ]

        variants = [
            ModelLaunchSettings(
                DEEPSEEK_V32_MODEL_PATH,
                tp_size=8,
                extra_args=BASE_ARGS + DP_ARGS + B200_REASONING_ARGS,
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="DeepSeek-V3.2 GPQA (B200)",
            accuracy_params=AccuracyTestParams(
                dataset="gpqa",
                baseline_accuracy=GPQA_BASELINE,
                num_examples=198,
                num_threads=198,
                max_tokens=120000,
                thinking_mode="deepseek-v3",
                temperature=0.1,
                repeat=4,
            ),
            performance_params=None,  # Skip performance test for GPQA
        )


if __name__ == "__main__":
    unittest.main()
