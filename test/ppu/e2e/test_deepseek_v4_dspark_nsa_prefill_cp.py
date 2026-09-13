import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import CustomTestCase, popen_launch_server

# E2E test for DeepSeek-V4-Flash DSPARK speculative decoding with NSA
# prefill context parallelism (interleave strategy) on PPU.
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "modelscope.cn/organization/T-HEAD/DeepSeek-V4-Flash-0731",
)
BASE_URL = "http://127.0.0.1:8999"
SERVER_LAUNCH_TIMEOUT = 60000  # 8-card loading on PPU can be slow
GSM8K_DATA_PATH = os.environ.get("GSM8K_DATA_PATH", None)


class TestDSV4FlashDsparkNsaPrefillCp(GSM8KMixin, CustomTestCase):
    """E2E: DeepSeek-V4-Flash DSPARK + NSA prefill CP (interleave) on PPU."""

    gsm8k_score_threshold = 0.90
    gsm8k_num_examples = 200
    gsm8k_accept_length_thres = 1.0
    gsm8k_data_path = GSM8K_DATA_PATH

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--trust-remote-code",
            "--tp",
            "8",
            "--speculative-algorithm",
            "DSPARK",
            "--speculative-dspark-block-size",
            "5",
            "--max-running-requests",
            "64",
            "--cuda-graph-max-bs-decode",
            "64",
            "--mem-fraction-static",
            "0.8",
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
            "--enable-prefill-cp",
            "--cp-strategy",
            "interleave",
            "--enable-dp-lm-head",
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
