"""H200 per-commit coverage for the GLM-5.3-Flash serving recipes.

Runs the Low Latency and High Throughput TP8/EP8 recipes on eight H200 GPUs.
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

register_cuda_ci(est_time=2400, stage="extra-b", runner_config="8-gpu-h200")

MODEL_PATH = "zai-org/GLM-5.3-Flash"
SERVER_LAUNCH_TIMEOUT = 3600
GPU_IDLE_TIMEOUT = 120

COMMON_SERVER_ARGS = [
    "--tp-size",
    "8",
    "--ep-size",
    "8",
    "--dsa-prefill-backend",
    "tilelang",
    "--dsa-decode-backend",
    "tilelang",
    "--kv-cache-dtype",
    "bf16",
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


class _GLM53FlashH200Base(CustomTestCase):
    server_args: list[str]

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


class TestGLM53FlashH200LowLatency(
    SpecDecodingMixin,
    GSM8KMixin,
    _GLM53FlashH200Base,
):
    gsm8k_score_threshold = 0.93
    # Match the established DSA+MTP accuracy workload. The generic 200-question,
    # 5-shot defaults leave a single question worth 0.5 percentage points and
    # make this tight quality floor unnecessarily sensitive to kernel numerics.
    gsm8k_num_examples = 500
    gsm8k_num_shots = 20
    accept_length_thres = 4.0
    bs_1_speed_thres = 200
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


class TestGLM53FlashH200HighThroughput(
    GSM8KMixin,
    _GLM53FlashH200Base,
):
    gsm8k_score_threshold = 0.93
    gsm8k_num_examples = 500
    gsm8k_num_shots = 20
    server_args = [
        *COMMON_SERVER_ARGS,
        "--enable-dp-attention",
        "--dp-size",
        "8",
        "--moe-a2a-backend",
        "deepep",
    ]


if __name__ == "__main__":
    unittest.main()
