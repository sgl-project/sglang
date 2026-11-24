"""Nightly accuracy + performance test for RedHatAI Llama-3.1-8B-Instruct w8a8 on CPU.

8B dense model with int8 weight+activation quantization (llmcompressor), TP=6 on Xeon.
Quantization config is embedded in the HF repo; SGLang auto-detects it.

Registry: nightly-test-cpu-intel suite"""

import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.cpu_test_utils import CPU_BASE_ARGS, CPU_LAUNCH_TIMEOUT
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import CustomTestCase, ModelLaunchSettings
from sglang.test.xeon_nightly_baselines import XEON_NIGHTLY_BASELINES

register_cpu_ci(est_time=1500, suite="nightly-test-cpu-intel")

MODEL_PATH = "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8"
BASELINE = XEON_NIGHTLY_BASELINES[MODEL_PATH]

BASE_ARGS = CPU_BASE_ARGS + ["--quantization", "w8a8_int8"]


class TestLlama31_8BW8A8Xeon(CustomTestCase):
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
                baseline_accuracy=BASELINE.min_accuracy,
                api="completion",
                num_threads=64,
                num_shots=5,
                max_tokens=2048,
                temperature=0,
                top_p=1,
                return_latency=True,
            ),
            performance_params=PerformanceTestParams(
                batch_sizes=[16],
                input_lens=(1024,),
                output_lens=(1024,),
                baseline_ftl_s=BASELINE.max_ftl_s,
                baseline_itl_ms=BASELINE.max_itl_ms,
                include_latency_breakdown=True,
            ),
            share_server=True,
        )
        self.assertTrue(result["all_passed"], f"Test failed: {result}")


if __name__ == "__main__":
    unittest.main()
