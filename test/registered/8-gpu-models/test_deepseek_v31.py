import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings, is_blackwell_system

# Runs on both H200 and B200 via nightly-8-gpu-common suite
register_cuda_ci(est_time=12000, suite="nightly-8-gpu-common", nightly=True)

DEEPSEEK_V31_MODEL_PATH = "deepseek-ai/DeepSeek-V3.1"


@unittest.skipIf(not is_blackwell_system(), "Requires B200")
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
        # Define base arguments shared by most variants
        base_args = [
            "--tp=8",
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
            # Variant: "basic" - Standard TP=8
            ModelLaunchSettings(
                DEEPSEEK_V31_MODEL_PATH,
                tp_size=8,
                extra_args=base_args,
            ),
            # Variant: "mtp" - TP=8 + EAGLE speculative decoding
            ModelLaunchSettings(
                DEEPSEEK_V31_MODEL_PATH,
                tp_size=8,
                extra_args=base_args + mtp_args,
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="DeepSeek-V3.1 Unified",
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k", baseline_accuracy=0.935
            ),
            performance_params=PerformanceTestParams(
                profile_dir="performance_profiles_deepseek_v31",
            ),
        )


if __name__ == "__main__":
    unittest.main()
