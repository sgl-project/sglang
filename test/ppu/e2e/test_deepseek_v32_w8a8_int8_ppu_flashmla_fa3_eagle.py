import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import CustomTestCase, popen_launch_server

# E2E test for DeepSeek-V3.2 W8A8-INT8 on PPU with EAGLE speculative decoding,
# flashmla decode + fa3 prefill, and DP attention.

MODEL_PATH = os.environ.get(
    "MODEL_PATH", "modelscope.cn/organization/T-HEAD/DeepSeek-V3.2-W8A8-INT8"
)
BASE_URL = "http://127.0.0.1:8999"
SERVER_LAUNCH_TIMEOUT = 24000

GSM8K_DATA_PATH = os.environ.get("GSM8K_DATA_PATH", None)


class TestDeepseekV32W8a8Int8PpuFlashmlaFa3Eagle(GSM8KMixin, CustomTestCase):
    """E2E: DeepSeek-V3.2 W8A8-INT8 on PPU with EAGLE + flashmla/fa3 + DP."""

    gsm8k_score_threshold = 0.50
    gsm8k_num_examples = 200
    gsm8k_accept_length_thres = 1.0
    gsm8k_data_path = GSM8K_DATA_PATH

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--tp-size",
            "16",
            "--mem-fraction-static",
            "0.8",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "2",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "3",
            "--disable-radix-cache",
            "--enable-dp-attention",
            "--dp-size",
            "16",
            "--quantization",
            "w8a8_int8",
            "--trust-remote-code",
            "--watchdog-timeout",
            "3600",
            "--dist-timeout",
            "3600",
            "--decode-attention-backend",
            "flashmla",
            "--prefill-attention-backend",
            "fa3",
            "--speculative-attention-mode",
            "decode",
            "--cuda-graph-max-bs",
            "64",
            "--disable-shared-experts-fusion",
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
