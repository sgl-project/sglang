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
DEEPEP_CONFIG = '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'

BALANCED_ARGS = [
    "--trust-remote-code",
    "--dp",
    "4",
    "--enable-dp-attention",
    "--moe-a2a-backend",
    "deepep",
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "1",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "2",
    "--mem-fraction-static",
    "0.85",
    "--cuda-graph-max-bs-decode",
    "128",
    "--max-running-requests",
    "256",
    "--deepep-config",
    DEEPEP_CONFIG,
    "--nccl-port",
    GB300_NCCL_PORT,
]

BALANCED_ENV = {
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
}


class TestDeepSeekV4ProFp4Balanced(CustomTestCase):
    """DeepSeek-V4-Pro FP4 balanced config on GB300."""

    def test_deepseek_v4_pro_fp4_balanced(self):
        run_combined_tests(
            models=[
                ModelLaunchSettings(
                    MODEL_PATH,
                    tp_size=4,
                    extra_args=BALANCED_ARGS,
                    env=BALANCED_ENV,
                    variant="balanced",
                    launch_timeout=SERVER_LAUNCH_TIMEOUT,
                )
            ],
            test_name="DeepSeek-V4-Pro-FP4 (balanced)",
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k",
                baseline_accuracy=0.935,
                temperature=1.0,
                top_p=1.0,
            ),
            performance_params=PerformanceTestParams(
                batch_sizes=[64],
                result_dir="performance_results_gb300",
            ),
        )


if __name__ == "__main__":
    unittest.main()
