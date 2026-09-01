"""Nightly accuracy + performance test for Google Gemma-4-26B-A4B-it on CPU.

26B MoE model with 4B active params, TP=6 on Xeon.

Registry: nightly-xeon-models suite"""

import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.cpu_test_utils import CPU_BASE_ARGS, CPU_LAUNCH_TIMEOUT
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

register_cpu_ci(est_time=3600, suite="nightly-xeon-models", nightly=True)

MODEL_PATH = "google/gemma-4-26B-A4B-it"

BASE_ARGS = CPU_BASE_ARGS + [
    "--context-length",
    "4096",
    "--skip-server-warmup",
]


class TestGemma4_26BCPU(unittest.TestCase):
    """Google Gemma-4-26B-A4B-it on Xeon."""

    def test_gemma4_26b(self):
        models = [
            ModelLaunchSettings(
                MODEL_PATH,
                extra_args=BASE_ARGS,
                # 16 attention heads are not divisible by 6, so tp=6 fails the
                # ensure_divisibility check on GNR; 16 % 2 == 0.
                tp_size=2,
                launch_timeout=CPU_LAUNCH_TIMEOUT,
            ),
        ]
        result = run_combined_tests(
            models=models,
            test_name="Gemma-4-26B-A4B (Xeon)",
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k",
                baseline_accuracy=0.95,
                num_threads=64,
                return_latency=True,
            ),
            performance_params=PerformanceTestParams(
                batch_sizes=[16],
                input_lens=(1024,),
                output_lens=(1024,),
                baseline_ftl_s=1.2,
                baseline_itl_ms=45.0,
                include_latency_breakdown=True,
            ),
            share_server=True,
        )
        self.assertTrue(result["all_passed"], f"Test failed: {result}")


if __name__ == "__main__":
    unittest.main()
