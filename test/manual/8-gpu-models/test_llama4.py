"""Moved out of test/registered/8-gpu-models/.

Originally registered with `register_cuda_ci(...)` on the nightly 8-gpu-h200 and
8-gpu-b200 suites. Moved here because nobody serves Llama 4 any more, and the CI
HF account has no access to meta-llama/Llama-4-Scout-17B-16E-Instruct either, so
it had been skipping for a while. Run with
`python3 test/manual/8-gpu-models/test_llama4.py`.
"""

import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

LLAMA4_MODEL_PATH = "meta-llama/Llama-4-Scout-17B-16E-Instruct"


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
            test_name="Llama-4-Scout",
            accuracy_params=AccuracyTestParams(dataset="gsm8k", baseline_accuracy=0.9),
            performance_params=PerformanceTestParams(
                result_dir="performance_results_llama4",
            ),
        )


if __name__ == "__main__":
    unittest.main()
