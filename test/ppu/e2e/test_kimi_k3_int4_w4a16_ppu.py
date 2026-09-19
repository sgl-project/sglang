import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import CustomTestCase, popen_launch_server

# E2E test for Kimi-K3 INT4-W4A16 on PPU.
# Verifies the server boots and GSM8K accuracy is correct for a
# mixed INT4 quantized Kimi K3 model.

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "Kimi-K3-INT4-W4A16",
)
SERVED_MODEL_NAME = "Kimi-K3-INT4-W4A16"
BASE_URL = "http://127.0.0.1:8999"
SERVER_LAUNCH_TIMEOUT = 24000

GSM8K_DATA_PATH = os.environ.get("GSM8K_DATA_PATH", None)


class TestKimiK3INT4W4A16Ppu(GSM8KMixin, CustomTestCase):
    """E2E: Kimi-K3 INT4-W4A16 on PPU."""

    model = SERVED_MODEL_NAME
    gsm8k_score_threshold = 0.50
    gsm8k_num_examples = 200
    gsm8k_accept_length_thres = 1.0
    gsm8k_data_path = GSM8K_DATA_PATH

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--tp-size",
            "16",
            "--trust-remote-code",
            "--watchdog-timeout",
            "600",
            "--served-model-name",
            SERVED_MODEL_NAME,
            "--max-taotal-tokens",
            "32768",
            "--max-running-requests",
            "1",
            "--mem-fraction-static",
            "0.85",
            "--chunked-prefill-size",
            "-1",
            "--disable-radix-cache",
            "--attention-backend",
            "triton",
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
