"""PD with --dsv41-encoder-only-prefill: encoder-only prefill, tail extend on decode."""

import random
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
    assert_process_healthy,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
    try_cached_model,
)

# Opt-in: two copies of V4.1 on one 8-GPU node. Initial estimate.
register_cuda_ci(est_time=1200, stage="extra-b", runner_config="8-gpu-b300")

DSV41_FLASH_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"

# Around the 128-token window: no transfer at all (<= 128), a one-token and an
# odd prefix (pending ratio-2 pair), page-aligned prefixes, and a long prompt.
PROMPT_LENS = [1, 127, 128, 129, 130, 131, 383, 384, 385, 4096]


class TestDisaggregationDSV41EncoderOnlyPrefill(PDDisaggregationServerBase, GSM8KMixin):
    gsm8k_accuracy_thres = 0.93

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = try_cached_model(DSV41_FLASH_MODEL)
        cls.start_server("prefill", gpus=range(4))
        cls.start_server("decode", gpus=range(4, 8))
        cls.wait_server_ready(cls.prefill_url + "/health", process=cls.process_prefill)
        cls.wait_server_ready(cls.decode_url + "/health", process=cls.process_decode)
        cls.launch_lb()

    @classmethod
    def start_server(cls, role, gpus):
        args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            role,
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--nccl-port",
            cls.prefill_nccl_port if role == "prefill" else cls.decode_nccl_port,
            "--tp",
            4,
            "--dsv41-encoder-only-prefill",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--max-running-requests",
            "32",
            "--watchdog-timeout",
            "900",
        ]
        if role == "decode":
            args += ["--base-gpu-id", 4]
        args += cls.transfer_backend + cls.rdma_devices_for(gpus)
        url = cls.prefill_url if role == "prefill" else cls.decode_url
        process = popen_launch_pd_server(
            cls.model,
            url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
        )
        setattr(cls, "process_" + role, process)

    def tearDown(self):
        for role in ("prefill", "decode"):
            assert_process_healthy(
                self,
                role,
                getattr(self, "process_" + role),
                getattr(self, role + "_url"),
            )

    def _generate(self, prompt_len, max_new_tokens=16):
        rng = random.Random(prompt_len)
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "input_ids": [rng.randrange(1000, 20000) for _ in range(prompt_len)],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=600,
        )
        response.raise_for_status()
        meta = response.json()["meta_info"]
        self.assertEqual(meta["completion_tokens"], max_new_tokens)
        self.assertEqual(meta["finish_reason"]["type"], "length")
        return response.json()["output_ids"]

    def test_window_boundaries(self):
        for n in PROMPT_LENS:
            self._generate(n)
        # Batched: the tail extends share one forward and merge into decode
        # while earlier requests are still decoding.
        with ThreadPoolExecutor(len(PROMPT_LENS)) as pool:
            list(pool.map(self._generate, PROMPT_LENS))


if __name__ == "__main__":
    unittest.main()
