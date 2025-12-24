import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings, is_blackwell_system

# Runs on both H200 and B200 via nightly-8-gpu-common suite
register_cuda_ci(est_time=12000, suite="nightly-8-gpu-common", nightly=True)

QWEN3_235B_MODEL_PATH = "Qwen/Qwen3-235B-A22B-Instruct-2507"


@unittest.skipIf(not is_blackwell_system(), "Requires B200")
class TestQwen3235BUnified(unittest.TestCase):
    """Unified test class for Qwen3-235B performance and accuracy.

    Single variant with simple TP=8 configuration.
    Runs BOTH:
    - Performance test (using NightlyBenchmarkRunner)
    - Accuracy test (using run_eval with mgsm_en)
    """

    def test_qwen3_235b(self):
        """Run performance and accuracy for Qwen3-235B."""
        base_args = [
            "--tp=8",
            "--trust-remote-code",
        ]

        variants = [
            ModelLaunchSettings(
                QWEN3_235B_MODEL_PATH,
                tp_size=8,
                extra_args=base_args,
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="Qwen3-235B Unified",
            accuracy_params=AccuracyTestParams(dataset="gsm8k", baseline_accuracy=0.88),
            performance_params=PerformanceTestParams(
                profile_dir="performance_profiles_qwen3_235b",
            ),
        )


if __name__ == "__main__":
    unittest.main()
