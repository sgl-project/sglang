"""Nightly accuracy + performance test for Qwen3-1.7B-FP8 on CPU.

1.7B dense model with FP8 weight quantization, TP=6 on Xeon.

Registry: nightly-xeon-models suite"""

import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.cpu_test_utils import CPU_BASE_ARGS, CPU_LAUNCH_TIMEOUT
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

register_cpu_ci(est_time=3600, suite="nightly-xeon-models", nightly=True)

MODEL_PATH = "Qwen/Qwen3-1.7B-FP8"

BASE_ARGS = CPU_BASE_ARGS


class TestQwen3_1B7FP8Xeon(unittest.TestCase):
    """Qwen3-1.7B-FP8 on Xeon."""

    def test_qwen3_1b7_fp8(self):
        models = [
            ModelLaunchSettings(
                MODEL_PATH,
                extra_args=BASE_ARGS,
                tp_size=6,
                launch_timeout=CPU_LAUNCH_TIMEOUT,
            ),
        ]
        result = run_combined_tests(
            models=models,
            test_name="Qwen3-1.7B-FP8 (Xeon)",
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k",
                baseline_accuracy=0.67,
                api="completion",
                num_threads=64,
                return_latency=True,
            ),
            performance_params=PerformanceTestParams(
                batch_sizes=[16],
                input_lens=(1024,),
                output_lens=(1024,),
                baseline_ftl_s=1.2,
                baseline_itl_ms=35.0,
                include_latency_breakdown=True,
            ),
            share_server=True,
        )
        self.assertTrue(result["all_passed"], f"Test failed: {result}")


if __name__ == "__main__":
    unittest.main()
