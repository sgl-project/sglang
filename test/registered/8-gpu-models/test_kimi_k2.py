import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

# Runs on both H200 and B200 via nightly-8-gpu-common suite
register_cuda_ci(est_time=1800, suite="nightly-8-gpu-common", nightly=True)

KIMI_K2_THINKING_MODEL_PATH = "moonshotai/Kimi-K2-Thinking"


class TestKimiK2(unittest.TestCase):
    """Unified test class for Kimi-K2-Thinking performance and accuracy.

    Single variant with TP=8 + tool/reasoning parsers.
    Runs BOTH:
    - Performance test (using NightlyBenchmarkRunner with extra_bench_args)
    - Accuracy test (using run_eval with mgsm_en)
    """

    def test_kimi_k2(self):
        """Run performance and accuracy for Kimi-K2-Thinking."""
        base_args = [
            "--tp=8",
            "--trust-remote-code",
            "--tool-call-parser=kimi_k2",
            "--reasoning-parser=kimi_k2",
        ]

        variants = [
            ModelLaunchSettings(
                KIMI_K2_THINKING_MODEL_PATH,
                tp_size=8,
                extra_args=base_args,
                variant="TP8",
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="Kimi-K2-Thinking Unified",
            accuracy_params=AccuracyTestParams(dataset="gsm8k", baseline_accuracy=0.94),
            performance_params=PerformanceTestParams(
                profile_dir="performance_profiles_kimi_k2_thinking",
            ),
        )


if __name__ == "__main__":
    unittest.main()
