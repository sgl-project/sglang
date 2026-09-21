import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import CustomTestCase, popen_launch_server

# E2E test for MiniMax-M3 MXFP4/FP8 on PPU with fa3 attention backend
# and deep_gemm MoE runner backend.

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "modelscope.cn/organization/T-HEAD/MiniMax-M3-MXFP4-FP8",
)
BASE_URL = "http://127.0.0.1:9985"
SERVER_LAUNCH_TIMEOUT = 24000

GSM8K_DATA_PATH = os.environ.get("GSM8K_DATA_PATH", None)


class TestMinimaxM3Mxfp4Fp8Fa3DeepGemm(GSM8KMixin, CustomTestCase):
    """E2E: MiniMax-M3 MXFP4/FP8 on PPU with fa3 attention + deep_gemm MoE runner."""

    gsm8k_score_threshold = 0.50
    gsm8k_num_examples = 200
    gsm8k_data_path = GSM8K_DATA_PATH

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--tp-size",
            "8",
            "--trust-remote-code",
            "--disable-radix-cache",
            "--mem-fraction-static",
            "0.9",
            "--reasoning-parser",
            "auto",
            "--tool-call-parser",
            "auto",
            "--attention-backend",
            "fa3",
            "--moe-runner-backend",
            "deep_gemm",
            "--watchdog-timeout",
            "3600",
            "--dist-timeout",
            "3600",
        ]
        cls.process = popen_launch_server(
            MODEL_PATH,
            BASE_URL,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
            env=os.environ,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    @property
    def base_url(self):
        return BASE_URL


if __name__ == "__main__":
    unittest.main()
