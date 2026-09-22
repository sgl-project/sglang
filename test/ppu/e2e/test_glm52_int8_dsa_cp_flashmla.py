import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import CustomTestCase, popen_launch_server

# E2E test for GLM-5.2 INT8 on PPU with DSA attention + context parallelism
# and flashmla_sparse/flashmla_kv DSA backends.

MODEL_PATH = os.environ.get(
    "MODEL_PATH", "modelscope.cn/organization/T-HEAD/GLM-5.2-int8"
)
BASE_URL = "http://127.0.0.1:8999"
SERVER_LAUNCH_TIMEOUT = 24000

GSM8K_DATA_PATH = os.environ.get("GSM8K_DATA_PATH", None)


class TestGlm52Int8DsaCpFlashmla(GSM8KMixin, CustomTestCase):
    """E2E: GLM-5.2 INT8 on PPU with DSA attention + CP + flashmla backends."""

    gsm8k_score_threshold = 0.50
    gsm8k_num_examples = 200
    gsm8k_accept_length_thres = 1.0
    gsm8k_data_path = GSM8K_DATA_PATH

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--tp-size",
            "16",
            "--attention-backend",
            "dsa",
            "--mem-fraction-static",
            "0.9",
            "--disable-radix-cache",
            "--quantization",
            "w8a8_int8",
            "--trust-remote-code",
            "--max-running-requests",
            "16",
            "--num-continuous-decode-steps",
            "1",
            "--dist-timeout",
            "3600",
            "--attn-cp-size",
            "16",
            "--enable-prefill-cp",
            "--cp-strategy",
            "interleave",
            "--watchdog-timeout",
            "3600",
            "--reasoning-parser",
            "glm45",
            "--tool-call-parser",
            "glm47",
            "--enable-metrics",
            "--dsa-prefill-backend",
            "flashmla_sparse",
            "--dsa-decode-backend",
            "flashmla_kv",
            "--enforce-disable-flashinfer-allreduce-fusion",
            "--disable-shared-experts-fusion",
            "--cuda-graph-max-bs-decode",
            "16",
            "--disable-prefill-cuda-graph",
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
