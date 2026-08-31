"""B200 per-commit coverage for the GLM-5.3-Flash serving recipes.

Runs the Low Latency and High Throughput TP4/EP4 recipes on four B200 GPUs.
Both recipes must retain GSM8K accuracy; the Low Latency recipe also checks
EAGLE speculative acceptance and single-request decode performance.
"""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.spec_decoding_kit import SpecDecodingMixin
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    _wait_for_gpu_idle_in_ci,
    popen_launch_server,
    try_cached_model,
)

register_cuda_ci(est_time=1800, stage="base-c", runner_config="4-gpu-b200")

MODEL_PATH = "zai-org/GLM-5.3-Flash"
SERVER_LAUNCH_TIMEOUT = 3600
GPU_IDLE_TIMEOUT = 120

COMMON_SERVER_ARGS = [
    "--tp-size",
    "4",
    "--ep-size",
    "4",
    "--dsa-prefill-backend",
    "trtllm",
    "--dsa-decode-backend",
    "trtllm",
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--moe-runner-backend",
    "deep_gemm",
    "--reasoning-parser",
    "glm45",
    "--tool-call-parser",
    "glm47",
]


def _stop_server(process):
    if process:
        kill_process_tree(process.pid)
        _wait_for_gpu_idle_in_ci(timeout=GPU_IDLE_TIMEOUT)


class _GLM53FlashB200Base(CustomTestCase):
    server_args: list[str]
    gsm8k_score_threshold = 0.93

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL_PATH)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = None
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=cls.server_args,
        )

    @classmethod
    def tearDownClass(cls):
        _stop_server(getattr(cls, "process", None))


class TestGLM53FlashB200LowLatency(
    SpecDecodingMixin,
    GSM8KMixin,
    _GLM53FlashB200Base,
):
    accept_length_thres = 4.0
    bs_1_speed_thres = 300
    server_args = [
        *COMMON_SERVER_ARGS,
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "5",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "6",
        "--speculative-adaptive",
    ]


class TestGLM53FlashB200HighThroughput(
    GSM8KMixin,
    _GLM53FlashB200Base,
):
    server_args = [
        *COMMON_SERVER_ARGS,
        "--enable-dp-attention",
        "--dp-size",
        "4",
        "--moe-a2a-backend",
        "deepep",
    ]


if __name__ == "__main__":
    unittest.main()
