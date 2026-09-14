import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import CustomTestCase, popen_launch_server

# E2E test for DeepSeek-V4-Pro-0813 mixed MXFP4/FP8 quantization with DSPARK
# speculative decoding and DP attention on PPU (tp8/dp8, fp8_e4m3 KV cache).
# Verifies the server boots and GSM8K accuracy is correct for a
# MoE MXFP4 + dense FP8 (per-channel W / per-token A) quantized DeepSeek V4
# Pro model.

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "modelscope.cn/organization/T-HEAD/DeepSeek-V4-Pro-0813-MoE-Quant-W-MXFP4-A-MXFP4-Dense-Quant-W-FP8-PerChannel-A-FP8-PerToken",
)
SERVED_MODEL_NAME = "DeepSeek-V4-Pro"
BASE_URL = "http://127.0.0.1:9985"
SERVER_LAUNCH_TIMEOUT = 24000

GSM8K_DATA_PATH = os.environ.get("GSM8K_DATA_PATH", None)


class TestDeepseekV4ProMxfp4Fp8DsparkDpAttention(GSM8KMixin, CustomTestCase):
    """E2E: DeepSeek-V4-Pro-0813 MXFP4/FP8 + DSPARK + DP attention on PPU."""

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
            "--trust-remote-code",
            "--speculative-algorithm",
            "DSPARK",
            "--speculative-dspark-block-size",
            "5",
            "--max-running-requests",
            "64",
            "--cuda-graph-max-bs-decode",
            "64",
            "--mem-fraction-static",
            "0.9",
            "--disable-radix-cache",
            "--disable-custom-all-reduce",
            "--disable-overlap-schedule",
            "--disable-shared-experts-fusion",
            "--disable-prefill-cuda-graph",
            "--watchdog-timeout",
            "60000",
            "--soft-watchdog-timeout",
            "60000",
            "--dist-timeout",
            "60000",
            "--enable-dp-lm-head",
            "--enable-dp-attention",
            "--dp-size",
            "8",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--served-model-name",
            SERVED_MODEL_NAME,
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
