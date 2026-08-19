import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.gb300_utils import GB300_NCCL_PORT
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import CustomTestCase, ModelLaunchSettings

register_cuda_ci(est_time=720, stage="nightly", runner_config="4-gpu-gb300")

MODEL_PATH = "deepseek-ai/DeepSeek-V4-Pro"
SERVER_LAUNCH_TIMEOUT = 3600

LOW_LATENCY_ARGS = [
    "--trust-remote-code",
    "--moe-runner-backend",
    "flashinfer_mxfp4",
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--chunked-prefill-size",
    "8192",
    "--disable-flashinfer-autotune",
    "--swa-full-tokens-ratio",
    "0.1",
    "--mem-fraction-static",
    "0.85",
    "--nccl-port",
    GB300_NCCL_PORT,
]


class TestDeepSeekV4ProFp4LowLatency(CustomTestCase):
    """DeepSeek-V4-Pro FP4 low-latency config on GB300."""

    def test_deepseek_v4_pro_fp4_low_latency(self):
        run_combined_tests(
            models=[
                ModelLaunchSettings(
                    MODEL_PATH,
                    tp_size=4,
                    extra_args=LOW_LATENCY_ARGS,
                    variant="low-latency",
                    launch_timeout=SERVER_LAUNCH_TIMEOUT,
                )
            ],
            test_name="DeepSeek-V4-Pro-FP4 (low-latency)",
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k",
                baseline_accuracy=0.935,
                temperature=1.0,
                top_p=1.0,
            ),
            performance_params=PerformanceTestParams(
                batch_sizes=[1, 4, 16],
                result_dir="performance_results_gb300",
            ),
        )


if __name__ == "__main__":
    unittest.main()
