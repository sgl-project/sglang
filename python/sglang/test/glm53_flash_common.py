"""Shared server setup for the GLM-5.3-Flash H200 serving recipes."""

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    _wait_for_gpu_idle_in_ci,
    popen_launch_server,
    try_cached_model,
)

MODEL_PATH = "zai-org/GLM-5.3-Flash"
SERVER_LAUNCH_TIMEOUT = 3600
GPU_IDLE_TIMEOUT = 120

COMMON_SERVER_ARGS = [
    "--tp-size",
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
    "auto",
    "--tool-call-parser",
    "auto",
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
