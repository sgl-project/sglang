import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

# Runs on both H200 and B200 via nightly-8-gpu-common suite
register_cuda_ci(est_time=1800, suite="nightly-8-gpu-common", nightly=True)

LLAMA4_MODEL_PATH = "meta-llama/Llama-4-Scout-17B-16E-Instruct"


@unittest.skip("Blocked: Missing HF token permission for Llama 4 model")
class TestLlama4(unittest.TestCase):
    """Unified test class for Llama-4-Scout performance and accuracy.

    Llama4 has local attention mechanism with hybrid sliding window attention.
    Single variant with TP=8 configuration.
    Runs BOTH:
    - Performance test (using NightlyBenchmarkRunner)
    - Accuracy test (using run_eval with gsm8k)
    """

    def test_llama4(self):
        """Run performance and accuracy for Llama-4-Scout."""
        base_args = [
            "--tp=8",
            "--trust-remote-code",
            "--chat-template=llama-4",
            "--mem-fraction-static=0.8",
            "--context-length=1000000",
        ]

        variants = [
            ModelLaunchSettings(
                LLAMA4_MODEL_PATH,
                tp_size=8,
                extra_args=base_args,
                variant="TP8",
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="Llama-4-Scout Unified",
            accuracy_params=AccuracyTestParams(dataset="gsm8k", baseline_accuracy=0.9),
            performance_params=PerformanceTestParams(
                profile_dir="performance_profiles_llama4",
            ),
        )


if __name__ == "__main__":
    unittest.main()
