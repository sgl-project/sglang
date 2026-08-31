"""Nightly accuracy + performance test for DeepSeek-Coder-V2-Lite-Instruct on CPU.

16B MoE coder model (2.4B active params), TP=6 on Xeon.
GSM8K baseline: 0.80 (86.4% reported - 5% buffer; CPU measured 0.809).

Registry: nightly-xeon-models suite
"""

import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.cpu_test_utils import CPU_BASE_ARGS, CPU_LAUNCH_TIMEOUT
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

register_cpu_ci(est_time=3600, suite="nightly-xeon-models", nightly=True)

MODEL_PATH = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"

BASE_ARGS = CPU_BASE_ARGS


class TestDeepSeekCoderV2LiteXeon(unittest.TestCase):
    """DeepSeek-Coder-V2-Lite-Instruct on Xeon."""

    def test_deepseek_coder_v2_lite(self):
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
            test_name="DeepSeek-Coder-V2-Lite-Instruct (Xeon)",
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k",
                baseline_accuracy=0.80,
                api="completion",
                num_threads=64,
                return_latency=True,
            ),
            performance_params=PerformanceTestParams(
                batch_sizes=[16],
                input_lens=(1024,),
                output_lens=(1024,),
                baseline_ftl_s=2.0,
                baseline_itl_ms=100.0,
                include_latency_breakdown=True,
            ),
            share_server=True,
        )
        self.assertTrue(result["all_passed"], f"Test failed: {result}")


if __name__ == "__main__":
    unittest.main()
