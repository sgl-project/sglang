import os
import subprocess
import time
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
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
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(metrics)
        return metrics

    def test_gsm8k(self):
        metrics = self._run_gsm8k()
        self.assertGreater(metrics["score"], 0.60)


class TestNixlEPTP(_EPTestBase):
    server_args = [*NIXL_COMMON]


class TestNixlEPDPAttn(_EPTestBase):
    server_args = [*NIXL_COMMON, *DP_ATTN]


class TestNixlEPElasticEP(_EPTestBase):
    server_args = [*NIXL_COMMON, *DP_ATTN, *ELASTIC_NIXL]


class TestNixlMoeMooncakeElasticEP(_EPTestBase):
    server_args = [
        *NIXL_COMMON,
        *DP_ATTN,
        *ELASTIC_MOONCAKE,
        "--moe-dense-tp-size",
        "1",
        "--enable-dp-lm-head",
    ]

    pkill_process_1 = "sglang::scheduler_DP1_TP1_EP1"

    def test_gsm8k_fault_1(self):
        subprocess.run(
            ["pkill", "-f", f"^{self.pkill_process_1}$"],
            check=True,
        )
        # Bootstrap one forward on a survivor so the controller learns the
        # post-fault active-rank mask before dispatching concurrent requests.
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "Hello",
                "sampling_params": {"max_new_tokens": 1},
                "routed_dp_rank": 0,
            },
            timeout=120,
        )
        self.assertEqual(response.status_code, 200, response.text)
        metrics = self._run_gsm8k()
        self.assertGreater(metrics["score"], 0.60)


if __name__ == "__main__":
    unittest.main()
