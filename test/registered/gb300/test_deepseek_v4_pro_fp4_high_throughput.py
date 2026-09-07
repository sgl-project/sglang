import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.gb300_utils import GB300_NCCL_PORT
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import CustomTestCase, ModelLaunchSettings

register_cuda_ci(est_time=600, stage="nightly", runner_config="4-gpu-gb300")

MODEL_PATH = "deepseek-ai/DeepSeek-V4-Pro"
SERVER_LAUNCH_TIMEOUT = 3600

HIGH_THROUGHPUT_ARGS = [
    "--trust-remote-code",
    "--dp",
    "4",
    "--enable-dp-attention",
    "--moe-a2a-backend",
    "megamoe",
    "--mem-fraction-static",
    "0.9",
    "--cuda-graph-max-bs-decode",
    "128",
    "--max-running-requests",
    "256",
    "--nccl-port",
    GB300_NCCL_PORT,
]

HIGH_THROUGHPUT_ENV = {
    "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK": "8320",
}


class TestDeepSeekV4ProFp4HighThroughput(CustomTestCase):
    """DeepSeek-V4-Pro FP4 high-throughput config on GB300."""

    def test_deepseek_v4_pro_fp4_high_throughput(self):
        run_combined_tests(
            models=[
                ModelLaunchSettings(
                    MODEL_PATH,
                    tp_size=4,
                    extra_args=HIGH_THROUGHPUT_ARGS,
                    env=HIGH_THROUGHPUT_ENV,
                    variant="high-throughput",
                    launch_timeout=SERVER_LAUNCH_TIMEOUT,
                )
            ],
            test_name="DeepSeek-V4-Pro-FP4 (high-throughput)",
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k",
                baseline_accuracy=0.935,
                temperature=1.0,
                top_p=1.0,
            ),
            performance_params=PerformanceTestParams(
                batch_sizes=[128],
                result_dir="performance_results_gb300",
            ),
        )


if __name__ == "__main__":
    unittest.main()
