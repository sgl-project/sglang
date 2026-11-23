"""Verify configurable top-logprobs metadata survives Ascend P-to-D transfers."""

import json
import math
import secrets
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)

register_npu_ci(est_time=180, suite="base-b-test-2-npu-a3")
register_npu_ci(est_time=180, suite="nightly-2-npu-a3", nightly=True)


class TestNpuPDTopLogprobs(PDDisaggregationServerBase):
    use_rdma = False
    max_top_logprobs_num = 256
    max_new_tokens = 8
    request_timeout = 120
    extra_prefill_args = [
        "--device",
        "npu",
        "--attention-backend",
        "ascend",
        "--sampling-backend",
        "pytorch",
        "--mem-fraction-static",
        "0.2",
        "--context-length",
        "2048",
        "--max-total-tokens",
        "4096",
        "--max-running-requests",
        "8",
        "--disable-cuda-graph",
        "--disaggregation-max-top-logprobs-num",
        "256",
    ]
    extra_decode_args = extra_prefill_args + ["--num-reserved-decode-tokens", "64"]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = QWEN3_0_6B_WEIGHTS_PATH
        cls.transfer_backend = ["--disaggregation-transfer-backend", "ascend"]
        cls.rdma_devices = []
        transfer_env = {
            "ASCEND_MF_STORE_URL": (
                f"tcp://{cls.base_host}:{int(cls.bootstrap_port) + 1}"
            ),
            "ASCEND_MF_TRANSFER_PROTOCOL": "sdma",
        }
        cls.extra_prefill_env = transfer_env
        cls.extra_decode_env = transfer_env
        cls.launch_all()

    def _payload(self, top_k, *, paired=False, stream=False):
        payload = {
            "text": "The capital of France is",
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": self.max_new_tokens,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "top_logprobs_num": top_k,
            "stream": stream,
        }
        if paired:
            payload.update(
                bootstrap_host=self.base_host,
                bootstrap_port=int(self.bootstrap_port),
                bootstrap_room=secrets.randbits(63),
            )
        return payload

    def _generate(self, url, payload):
        response = requests.post(
            url + "/generate", json=payload, timeout=self.request_timeout
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _assert_output_logprobs(self, result, top_k):
        meta = result["meta_info"]
        self.assertEqual(meta["finish_reason"]["type"], "length")
        self.assertEqual(meta["completion_tokens"], self.max_new_tokens)
        self.assertEqual(len(meta["output_token_logprobs"]), self.max_new_tokens)
        self.assertEqual(len(meta["output_top_logprobs"]), self.max_new_tokens)
        for row in meta["output_top_logprobs"]:
            self.assertEqual(len(row), top_k)
            self.assertEqual(len({item[1] for item in row}), top_k)
            values = [item[0] for item in row]
            self.assertTrue(all(math.isfinite(value) for value in values))
            self.assertEqual(values, sorted(values, reverse=True))

    def test_first_token_metadata_transfer(self):
        # A fixed-width or truncated transfer can corrupt only the first token;
        # later tokens' logprobs are computed locally on decode.
        with ThreadPoolExecutor(max_workers=2) as executor:
            for top_k in (128, 129, self.max_top_logprobs_num):
                with self.subTest(top_k=top_k):
                    payload = self._payload(top_k, paired=True)
                    prefill = executor.submit(self._generate, self.prefill_url, payload)
                    decode = executor.submit(self._generate, self.decode_url, payload)
                    prefill_meta = prefill.result()["meta_info"]
                    decode_result = decode.result()
                    self._assert_output_logprobs(decode_result, top_k)
                    self.assertEqual(prefill_meta["completion_tokens"], 1)
                    decode_meta = decode_result["meta_info"]
                    self.assertEqual(
                        prefill_meta["output_top_logprobs"][0],
                        decode_meta["output_top_logprobs"][0],
                    )
                    self.assertEqual(
                        prefill_meta["output_token_logprobs"][0],
                        decode_meta["output_token_logprobs"][0],
                    )

    def test_router_top_logprobs(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                payload = self._payload(self.max_top_logprobs_num, stream=stream)
                if not stream:
                    result = self._generate(self.lb_url, payload)
                else:
                    result = None
                    with requests.post(
                        self.lb_url + "/generate",
                        json=payload,
                        stream=True,
                        timeout=self.request_timeout,
                    ) as response:
                        self.assertEqual(response.status_code, 200, response.reason)
                        for line in response.iter_lines(decode_unicode=True):
                            if line.startswith("data: "):
                                data = line[6:]
                                if data != "[DONE]":
                                    result = json.loads(data)
                    self.assertIsNotNone(result)
                self._assert_output_logprobs(result, self.max_top_logprobs_num)

    def test_capacity_rejection(self):
        for url in (self.prefill_url, self.decode_url):
            with self.subTest(url=url):
                response = requests.post(
                    url + "/generate",
                    json=self._payload(self.max_top_logprobs_num + 1, paired=True),
                    timeout=self.request_timeout,
                )
                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn("257", response.text)
                self.assertIn("256", response.text)
                self.assertIn("capacity", response.text)

        # An invalid request must not poison either role's subsequent transfers.
        result = self._generate(self.lb_url, self._payload(self.max_top_logprobs_num))
        self._assert_output_logprobs(result, self.max_top_logprobs_num)


if __name__ == "__main__":
    unittest.main()
