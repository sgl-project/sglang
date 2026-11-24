"""Nightly accuracy + performance test for openai/gpt-oss-20b on CPU.

20B language model, TP=3 on Xeon.

Registry: nightly-test-cpu-intel suite
"""

import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.cpu_test_utils import CPU_BASE_ARGS, CPU_LAUNCH_TIMEOUT
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import CustomTestCase, ModelLaunchSettings
from sglang.test.xeon_nightly_baselines import XEON_NIGHTLY_BASELINES

# Estimated from a complete run including the automatic retry.
register_cpu_ci(est_time=9900, suite="nightly-test-cpu-intel")

MODEL_PATH = "openai/gpt-oss-20b"
BASELINE = XEON_NIGHTLY_BASELINES[MODEL_PATH]

BASE_ARGS = CPU_BASE_ARGS


class TestGPTOSS20BXeon(CustomTestCase):
    """GPT-OSS-20B on Xeon."""

    def test_gpt_oss_20b(self):
        models = [
            ModelLaunchSettings(
                MODEL_PATH,
                extra_args=BASE_ARGS,
                tp_size=3,
                launch_timeout=CPU_LAUNCH_TIMEOUT,
            ),
        ]
        result = run_combined_tests(
            models=models,
            test_name="GPT-OSS-20B (Xeon)",
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k",
                baseline_accuracy=BASELINE.min_accuracy,
                api="chat",
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
