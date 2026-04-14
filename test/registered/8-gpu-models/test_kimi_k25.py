import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

# Runs on both H200 and B200 via nightly-8-gpu-common suite
register_cuda_ci(est_time=3600, suite="nightly-8-gpu-common", nightly=True)

KIMI_K25_MODEL_PATH = "moonshotai/Kimi-K2.5"


class TestKimiK25(unittest.TestCase):
    """Unified test class for Kimi-K2.5 performance and accuracy.

    Runs TP=8 with tool/reasoning parsers.
    Runs BOTH performance test and accuracy test (gsm8k).
    """

    def test_kimi_k25(self):
        """Run performance and accuracy for all Kimi-K2.5 variants."""
        base_args = [
            "--trust-remote-code",
            "--tool-call-parser=kimi_k2",
            "--reasoning-parser=kimi_k2",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true, "num_threads": 64}',
        ]

        dp_attn_args = [
            "--dp=8",
            "--enable-dp-attention",
        ]

        variants = [
            ModelLaunchSettings(
                KIMI_K25_MODEL_PATH,
                tp_size=8,
                extra_args=base_args,
                variant="TP8",
            ),
            ModelLaunchSettings(
                KIMI_K25_MODEL_PATH,
                tp_size=8,
                extra_args=base_args + dp_attn_args,
                variant="TP8+DP8",
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="Kimi-K2.5",
            accuracy_params=AccuracyTestParams(dataset="gsm8k", baseline_accuracy=0.92),
            performance_params=PerformanceTestParams(
                profile_dir="performance_profiles_kimi_k25",
            ),
        )


if __name__ == "__main__":
    unittest.main()
