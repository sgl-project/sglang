import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import CustomTestCase, popen_launch_server

# E2E test for DeepSeek-R1 AWQ on PPU with flashmla decode + fa3 prefill.
# Verifies the server boots and produces correct GSM8K accuracy for an
# AWQ-quantized DeepSeek model on PPU.

MODEL_PATH = os.environ.get(
    "MODEL_PATH", "modelscope.cn/organization/T-HEAD/DeepSeek-R1-awq"
)
SERVED_MODEL_NAME = "DeepSeek-R1"
BASE_URL = "http://127.0.0.1:8999"
SERVER_LAUNCH_TIMEOUT = 24000

# Local GSM8K dataset path — set GSM8K_DATA_PATH env var to avoid
# re-downloading the dataset on every run.
GSM8K_DATA_PATH = os.environ.get("GSM8K_DATA_PATH", None)


class TestDeepseekR1AwqPpuFlashmlaFa3(GSM8KMixin, CustomTestCase):
    """E2E: DeepSeek-R1 AWQ on PPU with flashmla decode + fa3 prefill.

    Verifies the server boots and GSM8K accuracy is correct for an
    AWQ-quantized DeepSeek model on PPU.
    """

    # --- GSM8KMixin config ---
    model = SERVED_MODEL_NAME
    gsm8k_score_threshold = 0.50
    gsm8k_num_examples = 200
    gsm8k_accept_length_thres = 1.0
    gsm8k_data_path = GSM8K_DATA_PATH

    # --- server launch config ---
    @classmethod
    def setUpClass(cls):
        other_args = [
            "--tp-size",
            "8",
            "--mem-fraction-static",
            "0.8",
            "--trust-remote-code",
            "--served-model-name",
            SERVED_MODEL_NAME,
            "--decode-attention-backend",
            "flashmla",
            "--prefill-attention-backend",
            "fa3",
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
