import os
import unittest

from sglang.srt.environ import envs
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

# E2E test for Qwen3.5-397B-A17B W8A8-INT8 on SAIL DeepGEMM.
# Verifies the server boots and produces correct GSM8K accuracy when:
#   - quantization is w8a8_int8
#   - attention backend is fa3
#   - DP attention is enabled with DeepEP all-to-all
#   - DeepGEMM dense and MoE paths are active on SAIL

MODEL_PATH = os.environ.get(
    "MODEL_PATH", "modelscope.cn/organization/T-HEAD/Qwen3.5-397B-A17B-W8A8-INT8"
)
SERVED_MODEL_NAME = "Qwen3.5-397B-A17B"
BASE_URL = "http://127.0.0.1:8999"
SERVER_LAUNCH_TIMEOUT = 24000

# Local GSM8K dataset path — set GSM8K_DATA_PATH env var to avoid
# re-downloading the dataset on every run.
GSM8K_DATA_PATH = os.environ.get("GSM8K_DATA_PATH", None)


class TestQwen35SailDeepgemmInt8(GSM8KMixin, CustomTestCase):
    """Check SAIL DeepGEMM dense/MoE startup and GSM8K accuracy with DP attention."""

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
            "--deepep-mode",
            "auto",
            "--enable-dp-attention",
            "--dp-size",
            "8",
            "--quantization",
            "w8a8_int8",
            "--trust-remote-code",
            "--watchdog-timeout",
            "6000",
            "--dist-timeout",
            "6000",
            "--enable-dp-lm-head",
            "--served-model-name",
            SERVED_MODEL_NAME,
            "--moe-a2a-backend",
            "deepep",
            "--moe-dense-tp-size",
            "1",
            "--mamba-radix-cache-strategy",
            "extra_buffer",
        ]
        with (
            envs.SGLANG_SAIL_DEEPGEMM_DENSE.override(True),
            envs.SGLANG_SAIL_DEEPGEMM_MOE.override(True),
        ):
            cls.process = popen_launch_server(
                MODEL_PATH,
                BASE_URL,
                timeout=SERVER_LAUNCH_TIMEOUT,
                other_args=other_args,
            )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            terminate_and_kill_process_tree(cls.process)

    @property
    def base_url(self):
        return BASE_URL


if __name__ == "__main__":
    unittest.main()
