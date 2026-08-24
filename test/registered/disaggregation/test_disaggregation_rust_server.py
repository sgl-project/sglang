"""PD disaggregation with the embedded Rust server on both sides.

Same 2-GPU layout as test_disaggregation_basic (prefill GPU 0, decode GPU 1,
mini_lb in front), but prefill and decode run with ``SGLANG_RUST_SERVER=1`` —
covering the Rust `/generate` bootstrap-field intake (scalar form via the gsm8k
eval's single-prompt requests, per-item list form via the batch test), the
positional scheduler-wire PD block, the KV bootstrap registry served on the
rust api listener, the PD warmup fan-out, and the fake-bootstrap health probe.

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

register_cuda_ci(est_time=84, stage="base-b", runner_config="2-gpu-large")


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
        # Rust-server prefill serves the KV bootstrap registry on its api
        # listener (a separate --disaggregation-bootstrap-port is a launch
        # error there), so point both sides' bootstrap port at it: decode's
        # flag is its fallback for requests without a bootstrap_port field,
        # which is what mini_lb sends when --prefill carries no port.
        cls.bootstrap_port = cls.prefill_port
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

    def test_generate_stream_via_lb(self):
        # The scalar-bootstrap non-stream path is already covered 64x with an
        # accuracy gate by test_gsm8k; what is unique here is mini_lb passing
        # the decode node's SSE frames through under PD.
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
        # Frames are cumulative (--incremental-streaming-output defaults off),
        # so the last frame must extend the first. (Frame *count* is not
        # asserted: a slow reader legitimately coalesces a drained backlog.)
        self.assertTrue(chunks[-1]["text"].startswith(chunks[0]["text"]))
        # Exactly one terminal frame, and it is the last one. On a PD stream the
        # prefill node produces its own finish_reason frame; leaking that into
        # the decode stream would truncate the client mid-generation.
        terminal = [
            i
            for i, chunk in enumerate(chunks)
            if chunk["meta_info"]["finish_reason"] is not None
        ]
        self.assertEqual(terminal, [len(chunks) - 1], f"{terminal=} {len(chunks)=}")
        # One request id across the whole stream — not prefill's, then decode's.
        self.assertEqual(len({chunk["meta_info"]["id"] for chunk in chunks}), 1)

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
        # Per-prompt answers, not just non-empty text: a bootstrap room paired
        # with the wrong list index hands one item the other's transferred KV,
        # which a truthiness check cannot see.
        for item, expected in zip(j, ("paris", "tokyo")):
            self.assertIn(expected, item["text"].lower(), item)
            self.assertIsNotNone(item["meta_info"]["finish_reason"])

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
        # The *whole* prompt, since logprob_start_len is 0: a merge that drops
        # prefill's list and leaves only what decode itself saw still yields a
        # non-empty list, so pin the exact length.
        self.assertEqual(len(meta["input_token_logprobs"]), meta["prompt_tokens"])

    def test_missing_bootstrap_is_rejected(self):
        # Negative branch of the fake-bootstrap health probe: a /generate that
        # reaches a PD node *without* the router's bootstrap fields must surface
        # the scheduler's 400 abort through the rust wire — not hang, not 500.
        # Nothing else in this suite reaches the rust egress' abort_status path.
        response = requests.post(
            self.prefill_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
            timeout=60,
        )
        self.assertEqual(response.status_code, 400, response.text)

    def test_backend_health(self):
        # /health_generate directly on each side: on a PD node the probe only
        # passes with the fake bootstrap pair injected (room-less requests are
        # 400-aborted by the scheduler). Not the fixture's assert_process_healthy:
        # its 10s client timeout is shorter than the probe's own deadline
        # (SGLANG_HEALTH_CHECK_TIMEOUT, 20s), which would turn a slow-but-passing
        # side into a connection error.
        for name, process, url in (
            ("prefill", self.process_prefill, self.prefill_url),
            ("decode", self.process_decode, self.decode_url),
        ):
            self.assertIsNone(
                process.poll(), f"{name} exited with code {process.returncode}"
            )
            response = requests.get(url + "/health_generate", timeout=60)
            self.assertEqual(response.status_code, 200, response.text)


if __name__ == "__main__":
    unittest.main()
