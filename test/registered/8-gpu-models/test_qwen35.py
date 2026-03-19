import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

# Runs on both H200 and B200 via nightly-8-gpu-common suite
register_cuda_ci(est_time=1800, suite="nightly-8-gpu-common", nightly=True)

QWEN35_MODEL_PATH = "Qwen/Qwen3.5-397B-A17B"


class TestQwen35(unittest.TestCase):
    """Unified test class for Qwen3.5-397B-A17B performance and accuracy.

    Qwen3.5 is a 397B MoE VLM with 17B active params.
    Features hybrid reasoning, tool calling, and multimodal capabilities.
    Runs BOTH:
    - Performance test (using NightlyBenchmarkRunner)
    - Accuracy test (using run_eval with gsm8k)
    """

    def test_qwen35(self):
        """Run performance and accuracy for Qwen3.5-397B-A17B."""
        base_args = [
            "--trust-remote-code",
            "--reasoning-parser=qwen3",
            "--tool-call-parser=qwen3_coder",
            "--mem-fraction-static=0.8",
        ]
        mtp_args = [
            "--speculative-algorithm=EAGLE",
            "--speculative-num-steps=3",
            "--speculative-eagle-topk=1",
            "--speculative-num-draft-tokens=4",
            "--mamba-scheduler-strategy=extra_buffer",
        ]

        variants = [
            ModelLaunchSettings(
                QWEN35_MODEL_PATH,
                tp_size=8,
                extra_args=base_args,
                variant="TP8",
            ),
            ModelLaunchSettings(
                QWEN35_MODEL_PATH,
                tp_size=8,
                extra_args=base_args + mtp_args,
                variant="TP8+MTP",
                env={"SGLANG_ENABLE_SPEC_V2": "1"},
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="Qwen3.5-397B-A17B",
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k",
                baseline_accuracy=0.95,
                thinking_mode="qwen3",
                max_tokens=8192,
                num_examples=200,
            ),
            performance_params=PerformanceTestParams(
                profile_dir="performance_profiles_qwen35",
            ),
        )


if __name__ == "__main__":
    unittest.main()
