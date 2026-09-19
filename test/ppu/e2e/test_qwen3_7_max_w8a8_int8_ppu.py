import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import CustomTestCase, popen_launch_server

# E2E test for Qwen3.7-Max W8A8-INT8 on PPU with DP attention (2-node).
#
# Multi-node launch:
#   Node 0:  PPU_NODE_RANK=0 PPU_DIST_INIT_ADDR=<node0-host>:23456 ...
#   Node 1:  PPU_NODE_RANK=1 PPU_DIST_INIT_ADDR=<node0-host>:23456 ...
# Both nodes run the same script. The master (rank 0) must start first.

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "modelscope.cn/organization/T-HEAD/qwen3.7-max-int8",
)
SERVED_MODEL_NAME = "Qwen3.7-Max"
BASE_URL = "http://127.0.0.1:8999"
SERVER_LAUNCH_TIMEOUT = 24000

GSM8K_DATA_PATH = os.environ.get("GSM8K_DATA_PATH", None)

NNODES = int(os.environ.get("PPU_NNODES", "2"))


def _multinode_args():
    """Return CLI args for multi-node launch, or skip the test."""
    rank = os.environ.get("NODE_RANK")
    addr = os.environ.get("DIST_INIT_ADDR")
    if rank is None or addr is None:
        raise unittest.SkipTest("multi-node test requires NODE_RANK and DIST_INIT_ADDR")
    return [
        "--nnodes",
        str(NNODES),
        "--node-rank",
        rank,
        "--dist-init-addr",
        addr,
    ]


class TestQwen3_7MaxW8a8Int8Ppu(GSM8KMixin, CustomTestCase):
    """E2E: Qwen3.7-Max W8A8-INT8 on PPU with DP attention (TP=32, DP=2)."""

    model = SERVED_MODEL_NAME
    gsm8k_score_threshold = 0.50
    gsm8k_num_examples = 200
    gsm8k_accept_length_thres = 1.0
    gsm8k_data_path = GSM8K_DATA_PATH

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--tp-size",
            "32",
            "--attention-backend",
            "fa3",
            "--mem-fraction-static",
            "0.75",
            "--disable-radix-cache",
            "--enable-dp-attention",
            "--dp-size",
            "2",
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
        ]
        other_args += _multinode_args()
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
