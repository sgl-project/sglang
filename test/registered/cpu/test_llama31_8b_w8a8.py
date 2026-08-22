"""Nightly accuracy + performance test for RedHatAI Llama-3.1-8B-Instruct w8a8 on CPU.

8B dense model with int8 weight+activation quantization (llmcompressor), TP=6 on Xeon.
Quantization config is embedded in the HF repo; SGLang auto-detects it.

Registry: nightly-xeon-models suite"""

import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.cpu_test_utils import CPU_BASE_ARGS, CPU_LAUNCH_TIMEOUT
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

register_cpu_ci(est_time=3600, suite="nightly-xeon-models", nightly=True)

MODEL_PATH = "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8"

BASE_ARGS = CPU_BASE_ARGS + ["--quantization", "w8a8_int8"]


class TestLlama31_8BW8A8Xeon(unittest.TestCase):
    """RedHatAI Llama-3.1-8B-Instruct-quantized.w8a8 on Xeon."""

    def test_llama31_8b_w8a8(self):
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
            test_name="Llama-3.1-8B-Instruct-quantized.w8a8 (Xeon)",
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k",
                baseline_accuracy=0.75,
                api="completion",
                num_threads=64,
                return_latency=True,
            ),
            performance_params=PerformanceTestParams(
                batch_sizes=[16],
                input_lens=(1024,),
                output_lens=(1024,),
                baseline_ftl_s=2.2,
                baseline_itl_ms=50.0,
                include_latency_breakdown=True,
            ),
            share_server=True,
        )
        self.assertTrue(result["all_passed"], f"Test failed: {result}")


if __name__ == "__main__":
    unittest.main()
