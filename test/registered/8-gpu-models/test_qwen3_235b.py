import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

# Runs on both H200 and B200 via nightly-8-gpu-common suite
register_cuda_ci(est_time=1800, suite="nightly-8-gpu-common", nightly=True)

QWEN3_235B_FP8_MODEL_PATH = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
QWEN3_235B_EAGLE3_MODEL_PATH = (
    "lmsys/SGLang-EAGLE3-Qwen3-235B-A22B-Instruct-2507-SpecForge-Meituan"
)


class TestQwen3235BFP8(unittest.TestCase):
    """Test class for Qwen3-235B-FP8 performance and accuracy.

    Two variants:
    - basic: TP=8
    - eagle3: TP=8 + EP=2 + EAGLE3 speculative decoding

    Each variant runs BOTH:
    - Performance test (using NightlyBenchmarkRunner)
    - Accuracy test (using run_eval with gsm8k)
    """

    def test_qwen3_235b_fp8_all_variants(self):
        """Run performance and accuracy for Qwen3-235B-FP8."""
        base_args = [
            "--tp=8",
            "--ep=2",
            "--trust-remote-code",
        ]
        eagle3_args = [
            "--speculative-algorithm=EAGLE3",
            f"--speculative-draft-model-path={QWEN3_235B_EAGLE3_MODEL_PATH}",
            "--speculative-num-steps=3",
            "--speculative-eagle-topk=1",
            "--speculative-num-draft-tokens=4",
        ]

        variants = [
            # Variant: "basic" - TP=8
            ModelLaunchSettings(
                QWEN3_235B_FP8_MODEL_PATH,
                tp_size=8,
                extra_args=base_args,
                variant="TP8",
            ),
            # Variant: "eagle3" - TP=8 + EP=2 + EAGLE3 speculative decoding
            ModelLaunchSettings(
                QWEN3_235B_FP8_MODEL_PATH,
                tp_size=8,
                extra_args=base_args + eagle3_args,
                variant="TP8+EP2+EAGLE3",
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="Qwen3-235B-FP8",
            accuracy_params=AccuracyTestParams(dataset="gsm8k", baseline_accuracy=0.88),
            performance_params=PerformanceTestParams(
                profile_dir="performance_profiles_qwen3_235b_fp8",
            ),
        )


if __name__ == "__main__":
    unittest.main()
