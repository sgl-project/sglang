import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

# Runs on both H200 and B200 via nightly-8-gpu-common suite
register_cuda_ci(est_time=1800, suite="nightly-8-gpu-common", nightly=True)

MINIMAX_M2_MODEL_PATH = "MiniMaxAI/MiniMax-M2"


class TestMiniMaxM2(unittest.TestCase):
    """Unified test class for MiniMax-M2 performance and accuracy.

    Single variant with TP=8 + EP=8 configuration.
    MiniMax-M2 is a 230B MoE model with 10B active params.
    Runs BOTH:
    - Performance test (using NightlyBenchmarkRunner with extra_bench_args)
    - Accuracy test (using run_eval with mgsm_en)
    """

    def test_minimax_m2(self):
        """Run performance and accuracy for MiniMax-M2."""
        base_args = [
            "--tp=8",
            "--ep=8",
            "--trust-remote-code",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true}',
        ]

        variants = [
            ModelLaunchSettings(
                MINIMAX_M2_MODEL_PATH,
                tp_size=8,
                extra_args=base_args,
                variant="TP8+EP8",
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="MiniMax-M2",
            accuracy_params=AccuracyTestParams(dataset="gsm8k", baseline_accuracy=0.80),
            performance_params=PerformanceTestParams(
                profile_dir="performance_profiles_minimax_m2",
            ),
        )


if __name__ == "__main__":
    unittest.main()
