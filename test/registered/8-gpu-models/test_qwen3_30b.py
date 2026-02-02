import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

# Runs on 8-GPU systems via nightly suite
register_cuda_ci(est_time=1200, suite="nightly-8-gpu-common", nightly=True)

QWEN3_30B_MODEL_PATH = f"/home/scratch.trt_llm_data/llm-models/Qwen3/Qwen3-30B-A3B-FP8/"

BASE_ARGS = [
    "--trust-remote-code",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true, "num_threads": 64}',
]

DP_ARGS = [
    "--tp=4",
    "--moe-cp-size=2",
    "--ep-size=2",
    "--attn-cp-size=2",
    "--enable-prefill-context-parallel",
]

MTP_ARGS = [
    "--cuda-graph-max-bs=32",
    "--max-running-requests=32",
]


class TestQwen330B(unittest.TestCase):
    """Unified test class for Qwen3-30B-A3B performance and accuracy.

    Single variant with simple TP=8 configuration.
    Runs BOTH:
    - Performance test (using NightlyBenchmarkRunner)
    - Accuracy test (using run_eval with gsm8k)
    """

    def test_qwen3_30b(self):
        """Run performance and accuracy for Qwen3-30B-A3B."""
        base_args = [
            "--tp=8",
            "--trust-remote-code",
        ]

        variants = [
            # ModelLaunchSettings(
            #     QWEN3_30B_MODEL_PATH,
            #     tp_size=8,
            #     extra_args=base_args,
            #     variant="TP8",
            # ),
            ModelLaunchSettings(
                QWEN3_30B_MODEL_PATH,
                tp_size=4,
                extra_args=BASE_ARGS + DP_ARGS + MTP_ARGS,
                variant="CP-in-seq-split",
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="Qwen3-30B-A3B Unified",
            accuracy_params=AccuracyTestParams(dataset="gsm8k", baseline_accuracy=0.84),
            performance_params=PerformanceTestParams(
                profile_dir="performance_profiles_qwen3_30b",
            ),
        )


if __name__ == "__main__":
    unittest.main()
