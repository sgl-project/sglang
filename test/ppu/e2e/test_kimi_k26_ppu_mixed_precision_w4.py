import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import CustomTestCase, popen_launch_server

# E2E test for mixed_precision_w4 (W4AInt8) Kimi-K2.6 on PPU.
# Verifies the server boots and produces correct GSM8K accuracy when:
#   - quantization is mixed_precision_w4 (W4AInt8)
#   - decode attention backend is flashmla and prefill attention backend is fa3
#   - speculative decoding uses EAGLE3

MODEL_PATH = os.environ.get(
    "MODEL_PATH", "modelscope.cn/organization/T-HEAD/Kimi-K2.6-W4A8-INT8"
)
DRAFT_MODEL_PATH = os.environ.get(
    "DRAFT_MODEL_PATH",
    "kimi-k2.6-eagle3-mla",
)
SERVED_MODEL_NAME = "Kimi-K2.6"
BASE_URL = "http://127.0.0.1:8999"
SERVER_LAUNCH_TIMEOUT = 24000

# Local GSM8K dataset path — set GSM8K_DATA_PATH env var to avoid
# re-downloading the dataset on every run.
GSM8K_DATA_PATH = os.environ.get("GSM8K_DATA_PATH", None)


class TestKimiK26PpuMixedPrecisionW4(GSM8KMixin, CustomTestCase):
    """E2E: Kimi-K2.6 W4A8-INT8 mixed_precision_w4 on PPU with EAGLE3.

    Verifies the server boots and speculative decoding produces correct
    GSM8K accuracy for the mixed_precision_w4 quantized model.
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
            "0.9",
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-draft-model-path",
            DRAFT_MODEL_PATH,
            "--speculative-num-steps",
            "2",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "3",
            "--disable-radix-cache",
            "--trust-remote-code",
            "--watchdog-timeout",
            "3600",
            "--dist-timeout",
            "3600",
            "--served-model-name",
            SERVED_MODEL_NAME,
            "--decode-attention-backend",
            "flashmla",
            "--prefill-attention-backend",
            "fa3",
            "--speculative-attention-mode",
            "decode",
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
