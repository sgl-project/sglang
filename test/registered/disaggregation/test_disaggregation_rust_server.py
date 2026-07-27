"""PD disaggregation with the embedded Rust server on both sides.

Same 2-GPU layout as test_disaggregation_basic (prefill GPU 0, decode GPU 1,
mini_lb in front), but prefill and decode run with ``SGLANG_RUST_SERVER=1`` —
covering the Rust `/generate` bootstrap-field intake (scalar and per-item list
forms injected by the router), the positional scheduler-wire PD block, the
scheduler-hosted KV bootstrap server, the PD warmup fan-out, and the
fake-bootstrap health probe.

The Rust server has no OpenAI endpoints, so everything (including the gsm8k
eval) goes through ``/generate``.

Usage:
python3 -m unittest test_disaggregation_rust_server.TestDisaggregationRustServer
"""

import json
import unittest
from types import SimpleNamespace

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST, is_rust_server_built

register_cuda_ci(est_time=500, stage="base-b", runner_config="2-gpu-large")


@unittest.skipUnless(
    is_rust_server_built(),
    "embedded rust server extension not built",
)
class TestDisaggregationRustServer(PDDisaggregationServerBase):
    extra_prefill_env = {"SGLANG_RUST_SERVER": "1"}
    extra_decode_env = {"SGLANG_RUST_SERVER": "1"}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        # launch_all already exercises the PD-specific plumbing: the rust PD
        # warmup fan-out and the fake-bootstrap /health probe on both sides.
        cls.launch_all()

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.lb_url,
            eval_name="gsm8k",
            api="generate",  # the Rust server has no /v1/completions
            max_tokens=512,
            num_examples=64,
            num_threads=32,
        )
        metrics = run_eval(args)
        print(f"Evaluation metrics: {metrics}")
        self.assertGreater(metrics["score"], 0.62)

    def test_generate_via_lb(self):
        # Single request: the router injects scalar bootstrap fields.
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
        )
        self.assertEqual(response.status_code, 200)
        j = response.json()
        self.assertTrue(j["text"])
        self.assertIsNotNone(j["meta_info"]["finish_reason"])

    def test_generate_stream_via_lb(self):
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
                "stream": True,
            },
            stream=True,
        )
        self.assertEqual(response.status_code, 200)
        chunks = []
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                break
            chunks.append(json.loads(payload))
        self.assertTrue(chunks)
        self.assertTrue(chunks[-1]["text"])
        self.assertIsNotNone(chunks[-1]["meta_info"]["finish_reason"])

    def test_batch_generate_via_lb(self):
        # A batch makes the router inject per-item bootstrap lists — the list
        # intake + per-item fan-out path on the Rust side.
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": ["The capital of France is", "The capital of Japan is"],
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
        )
        self.assertEqual(response.status_code, 200)
        j = response.json()
        self.assertEqual(len(j), 2)
        for item in j:
            self.assertTrue(item["text"])

    def test_logprob_merge_via_lb(self):
        # With return_logprob the router merges the *prefill* response's
        # input_token_logprobs into the decode response — both sides must
        # produce complete logprob meta_info. (No `return_input_logprob` here:
        # the Rust /generate body does not declare it.)
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
                "return_logprob": True,
                "logprob_start_len": 0,
            },
        )
        self.assertEqual(response.status_code, 200)
        meta = response.json()["meta_info"]
        self.assertEqual(len(meta["output_token_logprobs"]), meta["completion_tokens"])
        self.assertGreater(len(meta["input_token_logprobs"]), 0)

    def test_backend_health(self):
        # /health_generate directly on each side: on a PD node the probe only
        # passes with the fake bootstrap pair injected (room-less requests are
        # 400-aborted by the scheduler).
        for url in (self.prefill_url, self.decode_url):
            self.assertEqual(
                requests.get(url + "/health_generate", timeout=60).status_code,
                200,
            )


if __name__ == "__main__":
    unittest.main()
