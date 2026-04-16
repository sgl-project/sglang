import os
import time
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.server_fixtures.disaggregation_fixture import get_rdma_devices_args
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

TEST_MODEL = os.environ.get("NIXL_EP_TEST_MODEL", DEFAULT_MODEL_NAME_FOR_TEST_MLA)
os.environ.setdefault("SGLANG_NIXL_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK", "1024")

ib_devices = get_rdma_devices_args()

NIXL_COMMON = [
    "--trust-remote-code",
    "--moe-a2a-backend",
    "nixl",
    "--deepep-mode",
    "low_latency",
    "--tp",
    "8",
    "--mem-fraction-static",
    "0.78",
]
DP_ATTN = ["--dp", "8", "--enable-dp-attention"]
ELASTIC_NIXL = [
    "--elastic-ep-backend",
    "nixl",
    "--enable-eplb",
    "--ep-num-redundant-experts",
    "24",
]
ELASTIC_MOONCAKE = [
    "--elastic-ep-backend",
    "mooncake",
    "--mooncake-ib-device",
    ib_devices,
    "--enable-eplb",
    "--ep-num-redundant-experts",
    "24",
]


class _EPTestBase(CustomTestCase):
    server_args: list[str] = []

    @classmethod
    def setUpClass(cls):
        cls.model = TEST_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.server_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.process.wait(timeout=15)
        time.sleep(2)

    def _run_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(metrics)
        return metrics

    def test_gsm8k(self):
        metrics = self._run_gsm8k()
        self.assertGreater(metrics["accuracy"], 0.60)


class TestNixlEPTP(_EPTestBase):
    server_args = [*NIXL_COMMON]


class TestNixlEPDPAttn(_EPTestBase):
    server_args = [*NIXL_COMMON, *DP_ATTN]


class TestNixlEPElasticEP(_EPTestBase):
    server_args = [*NIXL_COMMON, *DP_ATTN, *ELASTIC_NIXL]


class TestNixlMoeMooncakeElasticEP(_EPTestBase):
    server_args = [*NIXL_COMMON, *DP_ATTN, *ELASTIC_MOONCAKE]

    pkill_process_1 = "sglang::scheduler_DP1_TP8_EP8"

    def test_gsm8k_fault_1(self):
        os.system(f"pkill -f {self.pkill_process_1}")
        metrics = self._run_gsm8k()
        self.assertGreater(metrics["accuracy"], 0.60)


if __name__ == "__main__":
    unittest.main()
