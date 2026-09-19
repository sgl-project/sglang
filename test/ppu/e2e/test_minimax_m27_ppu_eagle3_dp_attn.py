import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import CustomTestCase, popen_launch_server

# E2E test for MiniMax-M2.7 W8A8-INT8 on PPU with EAGLE3 + DP attention.
# Verifies the server boots and produces correct GSM8K accuracy when:
#   - quantization is w8a8_int8
#   - DP attention is enabled with dp-size 8 and enable-dp-lm-head
#   - speculative decoding uses EAGLE3 with decode attention mode
#   - MoE A2A backend is deepep with moe-dense-tp-size 1
#   - attention backend is fa3

MODEL_PATH = os.environ.get("MODEL_PATH", "minimax-m2.7-int8")
DRAFT_MODEL_PATH = os.environ.get(
    "DRAFT_MODEL_PATH",
    "MiniMax-M2.5-Eagle3",
)
SERVED_MODEL_NAME = "MiniMax-M2.7"
BASE_URL = "http://127.0.0.1:8999"
SERVER_LAUNCH_TIMEOUT = 24000

# Local GSM8K dataset path — set GSM8K_DATA_PATH env var to avoid
# re-downloading the dataset on every run.
GSM8K_DATA_PATH = os.environ.get("GSM8K_DATA_PATH", None)


class TestMiniMaxM27PpuEagle3DpAttn(GSM8KMixin, CustomTestCase):
    """E2E: MiniMax-M2.7 W8A8-INT8 on PPU with EAGLE3 + DP attention.

    Verifies the server boots and speculative decoding produces correct
    GSM8K accuracy under DP attention with 8 DP ranks, ensuring empty
    hidden_states on idle DP ranks does not crash the draft model.
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
            "--attention-backend",
            "fa3",
            "--mem-fraction-static",
            "0.85",
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
            "--deepep-mode",
            "auto",
            "--enable-dp-attention",
            "--dp-size",
            "8",
            "--quantization",
            "w8a8_int8",
            "--trust-remote-code",
            "--dist-timeout",
            "6000",
            "--enable-dp-lm-head",
            "--watchdog-timeout",
            "6000",
            "--served-model-name",
            SERVED_MODEL_NAME,
            "--speculative-attention-mode",
            "decode",
            "--speculative-draft-model-quantization",
            "unquant",
            "--moe-a2a-backend",
            "deepep",
            "--moe-dense-tp-size",
            "1",
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
