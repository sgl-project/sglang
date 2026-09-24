"""PD KV checksum (--disaggregation-enable-kv-checksum) end to end on NPU.

Prefill computes an Adler-32 over the request's KV pages and writes it to the
metadata buffer; decode recomputes it and compares. A mismatch aborts the
request with HTTP 500, so "every request returns 200" is the pass condition.

What this does not catch: decode skips the comparison when the prefill side
never wrote a checksum (expected == 0), so a request would pass vacuously.
test_both_sides_report_checksum_enabled is what rules that out. The kernel's
own arithmetic is covered by test/manual/ascend/test_npu_kv_checksum.py.
"""

import os
import random
import shlex
import unittest
from typing import Dict

import requests

from sglang.bench_serving import get_tokenizer
from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import popen_with_error_check


class TestNpuPdKvChecksum(PDDisaggregationServerBase):
    """Testcase: KV checksum verification stays silent across a PD transfer.

    [Test Category] Functional
    [Test Target] --disaggregation-enable-kv-checksum on NPU
    """

    prefill_tp_size = 2
    decode_tp_size = 2
    # Prefill takes the first two devices, decode the next two.
    decode_base_gpu_id = int(
        os.environ.get("SGLANG_TEST_DECODE_BASE_GPU_ID", str(prefill_tp_size))
    )

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = QWEN3_32B_WEIGHTS_PATH
        cls.tokenizer = get_tokenizer(cls.model)

        # The base class picks mooncake plus RDMA devices from the CI
        # environment; NPU transfers go over the ascend backend instead.
        cls.transfer_backend = ["--disaggregation-transfer-backend", "ascend"]
        cls.rdma_devices = []

        common_args = [
            "--attention-backend",
            "ascend",
            "--mem-fraction-static",
            "0.9",
            "--disable-cuda-graph",
            "--disaggregation-enable-kv-checksum",
        ]
        cls.extra_prefill_args = list(common_args)
        cls.extra_decode_args = list(common_args)

        cls.launch_all()

    @classmethod
    def rdma_devices_for(cls, gpu_indices) -> list:
        return []

    @classmethod
    def launch_lb(cls):
        # The bootstrap-port form that test_npu_pd_disaggregation.py uses, rather
        # than the base class's --mini-lb.
        lb_command = [
            "python3",
            "-m",
            "sglang_router.launch_router",
            "--pd-disaggregation",
            "--prefill",
            cls.prefill_url,
            cls.bootstrap_port,
            "--decode",
            cls.decode_url,
            "--host",
            cls.base_host,
            "--port",
            cls.lb_port,
        ]
        print("Starting load balancer:", shlex.join(lb_command))
        cls.process_lb = popen_with_error_check(lb_command)
        cls.wait_server_ready(cls.lb_url + "/health", process=cls.process_lb)

    def gen_prompt(self, token_num: int) -> str:
        all_available_tokens = list(self.tokenizer.get_vocab().values())
        selected_tokens = random.choices(all_available_tokens, k=token_num)
        return self.tokenizer.decode(selected_tokens)

    def send_request(self, prompt: str, max_tokens: int = 32) -> Dict:
        response = requests.post(
            f"{self.lb_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": max_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=120,
        )
        # A checksum mismatch aborts the request with INTERNAL_SERVER_ERROR.
        self.assertEqual(
            response.status_code,
            200,
            f"Request failed: {response.status_code} - {response.text}",
        )
        return response.json()

    def test_both_sides_report_checksum_enabled(self):
        for name, url in (("prefill", self.prefill_url), ("decode", self.decode_url)):
            with self.subTest(side=name):
                info = requests.get(f"{url}/server_info", timeout=30).json()
                key = "disaggregation_enable_kv_checksum"
                self.assertIn(key, info, f"{name} /server_info has no {key}")
                self.assertTrue(
                    info[key], f"{name} server did not enable the KV checksum"
                )

    def test_requests_across_page_counts(self):
        # Page counts from a single partial page up to a few thousand items,
        # which is what drives the kernel's per-program item loop.
        for token_num in (1, 17, 300, 800, 2000):
            with self.subTest(token_num=token_num):
                response = self.send_request(self.gen_prompt(token_num))
                self.assertTrue(response["text"])

    def test_repeated_prompt_reuses_prefix(self):
        # A second pass over the same prompt takes the cached-prefix path, where
        # prefill checksums a different page set than the first request did.
        prompt = self.gen_prompt(800)
        self.send_request(prompt)
        response = self.send_request(prompt)
        self.assertTrue(response["text"])


if __name__ == "__main__":
    unittest.main()
