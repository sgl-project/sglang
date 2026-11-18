"""Nightly accuracy + performance test for Microsoft Phi-4-reasoning on CPU.

Dense 14B reasoning model with <think> chain-of-thought, TP=1 on Xeon.
GSM8K baseline: 0.90 (~94% reported - 5% buffer).
Uses deepseek-r1 reasoning parser (compatible <think>/</think> format).

Registry: nightly-xeon-models suite"""

import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.cpu_test_utils import CPU_BASE_ARGS, CPU_LAUNCH_TIMEOUT
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

register_cpu_ci(est_time=3600, suite="nightly-xeon-models", nightly=True)

MODEL_PATH = "microsoft/Phi-4-reasoning"

BASE_ARGS = CPU_BASE_ARGS


class TestPhi4ReasoningCPU(unittest.TestCase):
    """Microsoft Phi-4-reasoning on Xeon."""

    def test_phi4_reasoning(self):
        models = [
            ModelLaunchSettings(
                MODEL_PATH,
                extra_args=BASE_ARGS,
                tp_size=2,
                launch_timeout=CPU_LAUNCH_TIMEOUT,
            ),
        ]
        result = run_combined_tests(
            models=models,
            test_name="Phi-4-reasoning (Xeon)",
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k",
                baseline_accuracy=0.90,
                api="completion",
                num_threads=64,
                return_latency=True,
            ),
            performance_params=PerformanceTestParams(
                batch_sizes=[16],
                input_lens=(1024,),
                output_lens=(1024,),
                baseline_ftl_s=23.0,
                baseline_itl_ms=310.0,
                include_latency_breakdown=True,
            ),
            share_server=True,
        )
        self.assertTrue(result["all_passed"], f"Test failed: {result}")


if __name__ == "__main__":
    unittest.main()
