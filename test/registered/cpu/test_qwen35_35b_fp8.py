"""Nightly accuracy + performance test for Qwen3.5-35B-A3B-FP8 on CPU.

35B MoE with 3B active params, FP8 weight quantization, TP=6 on Xeon.
MMLU baseline: 0.83 (~87% for MoE class - 4% buffer).
Using MMLU instead of GSM8K because thinking chains make GSM8K very slow on CPU.

Registry: nightly-xeon-models suite"""

import unittest

from utils import CPU_BASE_ARGS

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

register_cpu_ci(est_time=3600, suite="nightly-xeon-models", nightly=True)

# export SGLANG_DEEPSEEK_FP8A8=1
# build sgl-kernel using SGLANG_CPU_FP8_BRGEMM=1 on DMR
MODEL_PATH = "Qwen/Qwen3.5-35B-A3B-FP8"

BASE_ARGS = CPU_BASE_ARGS


class TestQwen35_35BFPS8Xeon(unittest.TestCase):
    """Qwen3.5-35B-A3B-FP8 on Xeon."""

    def test_qwen35_35b_fp8(self):
        models = [
            ModelLaunchSettings(
                MODEL_PATH,
                extra_args=BASE_ARGS,
                tp_size=6,
            ),
        ]
        result = run_combined_tests(
            models=models,
            test_name="Qwen3.5-35B-A3B-FP8 (Xeon)",
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k",
                baseline_accuracy=0.83,
                api="completion",
                num_threads=64,
            ),
            performance_params=PerformanceTestParams(
                batch_sizes=[16],
                input_lens=(1024,),
                output_lens=(1024,),
                enable_profile=False,
            ),
        )
        self.assertTrue(result["all_passed"], f"Test failed: {result}")


if __name__ == "__main__":
    unittest.main()
