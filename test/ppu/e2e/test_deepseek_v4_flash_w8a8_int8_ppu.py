import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import CustomTestCase, popen_launch_server

# E2E test for DeepSeek-V4-Flash W8A8-INT8 on PPU.

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "modelscope.cn/organization/T-HEAD/DeepSeek-V4-Flash-W8A8-INT8",
)
SERVED_MODEL_NAME = "DeepSeek-V4"
BASE_URL = "http://127.0.0.1:8999"
SERVER_LAUNCH_TIMEOUT = 24000

GSM8K_DATA_PATH = os.environ.get("GSM8K_DATA_PATH", None)


class TestDeepseekV4FlashW8a8Int8Ppu(GSM8KMixin, CustomTestCase):
    """E2E: DeepSeek-V4-Flash W8A8-INT8 on PPU."""

    model = SERVED_MODEL_NAME
    gsm8k_score_threshold = 0.50
    gsm8k_num_examples = 200
    gsm8k_accept_length_thres = 1.0
    gsm8k_data_path = GSM8K_DATA_PATH

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--tp-size",
            "8",
            "--mem-fraction-static",
            "0.8",
            "--quantization",
            "w8a8_int8",
            "--trust-remote-code",
            "--dist-timeout",
            "60000",
            "--watchdog-timeout",
            "60000",
            "--soft-watchdog-timeout",
            "60000",
            "--served-model-name",
            SERVED_MODEL_NAME,
            "--disable-custom-all-reduce",
            "--disable-shared-experts-fusion",
            "--cuda-graph-max-bs",
            "64",
            "--disable-piecewise-cuda-graph",
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
