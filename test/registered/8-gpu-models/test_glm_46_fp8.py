import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

# Runs on both H200 and B200 via nightly-8-gpu-common suite
register_cuda_ci(est_time=1800, suite="nightly-8-gpu-common", nightly=True)

GLM_4_6_FP8_MODEL_PATH = "zai-org/GLM-4.6-FP8"


class TestGLM46FP8(unittest.TestCase):
    """Unified test class for GLM-4.6-FP8 performance and accuracy.

    Single variant with simple TP=8 configuration.
    Runs BOTH:
    - Performance test (using NightlyBenchmarkRunner)
    - Accuracy test (using run_eval with mgsm_en)
    """

    def test_glm_46_fp8_all_variants(self):
        """Run performance and accuracy for GLM-4.6-FP8."""
        base_args = [
            "--tp=8",
            "--trust-remote-code",
        ]
        mtp_args = [
            "--speculative-algorithm=EAGLE",
            "--speculative-num-steps=3",
            "--speculative-eagle-topk=1",
            "--speculative-num-draft-tokens=4",
        ]

        variants = [
            ModelLaunchSettings(
                GLM_4_6_FP8_MODEL_PATH,
                tp_size=8,
                extra_args=base_args,
                variant="TP8",
            ),
            ModelLaunchSettings(
                GLM_4_6_FP8_MODEL_PATH,
                tp_size=8,
                extra_args=base_args + mtp_args,
                variant="TP8+MTP",
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="GLM-4.6-FP8",
            accuracy_params=AccuracyTestParams(dataset="gsm8k", baseline_accuracy=0.80),
            performance_params=PerformanceTestParams(
                profile_dir="performance_profiles_glm_4_6_fp8",
            ),
        )


if __name__ == "__main__":
    unittest.main()
