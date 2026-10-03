import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import CustomTestCase, popen_launch_server

# E2E test for MiMo-V2.5-Pro MXFP4/FP8 on PPU.
# Verifies the server boots and GSM8K accuracy is correct for the
# MiMo-V2.5-Pro dense-MoE model with DP attention + DeepEP + FA3.

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "modelscope.cn/organization/T-HEAD/MiMo-V2.5-Pro-mxfp4-fp8/",
)
BASE_URL = "http://127.0.0.1:30000"
SERVER_LAUNCH_TIMEOUT = 24000

GSM8K_DATA_PATH = os.environ.get("GSM8K_DATA_PATH", None)


class TestMimoV25ProMxfp4Fp8Ppu(GSM8KMixin, CustomTestCase):
    """E2E: MiMo-V2.5-Pro MXFP4/FP8 on PPU with DP attention + DeepEP + FA3."""

    gsm8k_score_threshold = 0.50
    gsm8k_num_examples = 200
    gsm8k_accept_length_thres = 1.0
    gsm8k_data_path = GSM8K_DATA_PATH

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--trust-remote-code",
            "--tp-size",
            "8",
            "--moe-dense-tp-size",
            "1",
            "--enable-dp-attention",
            "--moe-a2a-backend",
            "deepep",
            "--attention-backend",
            "fa3",
            "--mem-fraction-static",
            "0.8",
            "--max-running-requests",
            "128",
            "--cuda-graph-max-bs",
            "64",
            "--chunked-prefill-size",
            "32768",
            "--context-length",
            "1048576",
            "--tokenizer-worker-num",
            "64",
            "--reasoning-parser",
            "mimo",
            "--tool-call-parser",
            "mimo",
            "--cuda-graph-backend-prefill=disable",
            "--disable-radix-cache",
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
