"""Nightly accuracy + performance test for Google Gemma-4-26B-A4B on CPU.

26B MoE model with 4B active params, TP=6 on Xeon.

Registry: nightly-xeon-models suite"""

import unittest

from utils import CPU_BASE_ARGS

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

register_cpu_ci(est_time=3600, suite="nightly-xeon-models", nightly=True)

MODEL_PATH = "google/gemma-4-26B-A4B"

BASE_ARGS = CPU_BASE_ARGS + ["--skip-server-warmup"]


class TestGemma4_26BCPU(unittest.TestCase):
    """Google Gemma-4-26B-A4B on Xeon."""

    def test_gemma4_26b(self):
        models = [
            ModelLaunchSettings(
                MODEL_PATH,
                extra_args=BASE_ARGS,
                tp_size=6,
            ),
        ]
        result = run_combined_tests(
            models=models,
            test_name="Gemma-4-26B-A4B (Xeon)",
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k",
                baseline_accuracy=0.71,
                api="completion",
                num_threads=64,
            ),
            performance_params=PerformanceTestParams(
                batch_sizes=[16],
                input_lens=(1024,),
                output_lens=(1024,),
                enable_profile=False,
                baseline_output_throughput=80.0,
            ),
            share_server=True,
        )
        self.assertTrue(result["all_passed"], f"Test failed: {result}")


if __name__ == "__main__":
    unittest.main()
