import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

# Runs on B200 via nightly-4-gpu-b200 suite
register_cuda_ci(est_time=2000, suite="nightly-4-gpu-b200", nightly=True)

FULL_DEEPSEEK_V3_FP4_MODEL_PATH = "nvidia/DeepSeek-V3-0324-FP4"


class TestDeepseekR1FP4Unified(unittest.TestCase):
    """Unified test class for DeepSeek-V3-0324-FP4 performance and accuracy.

    Two variants:
    - basic: Standard TP=4
    - mtp: TP=4 + EAGLE speculative decoding

    Each variant runs BOTH:
    - Performance test (using NightlyBenchmarkRunner)
    - Accuracy test (using run_eval with mgsm_en)
    """

    def test_deepseek_r1_fp4_all_variants(self):
        """Run performance and accuracy for all DeepSeek-R1-0528-NVFP4-v2 variants."""
        # Define base arguments shared by most variants
        base_args = [
            "--tp=4",
            "--trust-remote-code",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true}',
        ]
        mtp_args = [
            "--speculative-algorithm=EAGLE",
            "--speculative-num-steps=3",
            "--speculative-eagle-topk=1",
            "--speculative-num-draft-tokens=4",
            "--mem-frac=0.7",
        ]

        variants = [
            # Variant: "basic" - Standard TP=4
            ModelLaunchSettings(
                FULL_DEEPSEEK_V3_FP4_MODEL_PATH,
                tp_size=4,
                extra_args=base_args,
                variant="TP4",
            ),
            # Variant: "mtp" - TP=4 + EAGLE speculative decoding
            ModelLaunchSettings(
                FULL_DEEPSEEK_V3_FP4_MODEL_PATH,
                tp_size=4,
                extra_args=base_args + mtp_args,
                variant="TP4+MTP",
                env={"SGLANG_ENABLE_SPEC_V2": "1"},
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="DeepSeek-V3-0324-FP4 Unified",
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k", baseline_accuracy=0.935
            ),
            performance_params=PerformanceTestParams(
                profile_dir="performance_profiles_deepseek_v3_fp4",
            ),
        )


if __name__ == "__main__":
    unittest.main()
