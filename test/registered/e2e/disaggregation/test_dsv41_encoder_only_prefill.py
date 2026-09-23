"""Cache-only handoff and recovery through a real V4.1 P/D/router stack.

Uses TP4 Prefill + TP4 Decode on eight B300 GPUs. The existing H20 evidence
uses two TP8 nodes; this smaller topology needs its own serving validation.
"""

import json
import random
import threading
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
    assert_process_healthy,
)
from sglang.test.test_utils import try_cached_model

# Opt-in: loading two copies of V4.1 needs a separate, high-memory PD fixture.
# Initial estimate; replace with measured runtime after the first B300 run.
register_cuda_ci(est_time=1800, stage="extra-b", runner_config="8-gpu-b300")


class TestDSV41EncoderOnlyPrefill(PDDisaggregationServerBase):
    model = "deepseek-ai/DeepSeek-V4.1-Flash"
    prefill_tp_size = decode_tp_size = 4
    decode_base_gpu_id = 4
    extra_prefill_args = [
        "--language-model-only",
        "--dsv41-encoder-only-prefill",
        "--attention-backend",
        "dsv4",
        "--moe-runner-backend",
        "flashinfer_mxfp4",
        "--ep-size",
        "4",
        "--page-size",
        "256",
        "--chunked-prefill-size",
        "8192",
        "--enable-decoder-swa-bounded-replay",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--cuda-graph-max-bs-decode",
        "8",
        "--max-running-requests",
        "8",
        "--mem-fraction-static",
        "0.8",
        "--context-length",
        "16384",
        "--random-seed",
        "0",
        "--enable-metrics",
    ]
    extra_decode_args = list(extra_prefill_args)
    extra_prefill_env = {
        "SGLANG_DSV4_FP4_EXPERTS": "1",
        "SGLANG_RAGGED_VERIFY_MODE": "static",
    }
    extra_decode_env = dict(extra_prefill_env)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = try_cached_model(cls.model)
        cls.transfer_backend = ["--disaggregation-transfer-backend", "mooncake"]
        cls.launch_all()
        for url, role in ((cls.prefill_url, "prefill"), (cls.decode_url, "decode")):
            response = requests.get(url + "/server_info", timeout=30)
            response.raise_for_status()
            info = response.json()
            assert info["dsv41_encoder_only_prefill"]
            assert info["disaggregation_mode"] == role
            assert info["speculative_algorithm"] is None
            assert not info["disable_overlap_schedule"]

    def tearDown(self):
        for name in ("prefill", "decode", "lb"):
            assert_process_healthy(
                self,
                name,
                getattr(self, "process_" + name),
                getattr(self, name + "_url"),
            )

    def _post_decode(self, path, payload):
        response = requests.post(self.decode_url + path, json=payload, timeout=30)
        response.raise_for_status()

    def _payload(self, prompt_len, max_tokens):
        # In-vocabulary token IDs, independent prefixes and exact lengths.
        rng = random.Random(uuid.uuid4().int)
        return {
            "rid": uuid.uuid4().hex,
            "input_ids": [rng.randrange(1000, 20000) for _ in range(prompt_len)],
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_tokens,
                "ignore_eos": True,
            },
        }

    def _assert_complete(self, result, max_tokens):
        self.assertNotIn("error", result)
        meta = result["meta_info"]
        self.assertEqual(meta["completion_tokens"], max_tokens, result)
        self.assertEqual(len(result["output_ids"]), max_tokens, result)
        self.assertEqual(meta["finish_reason"]["type"], "length", result)

    def _generate(self, prompt_len, max_tokens=32):
        response = requests.post(
            self.lb_url + "/generate",
            json=self._payload(prompt_len, max_tokens),
            timeout=180,
        )
        response.raise_for_status()
        result = response.json()
        self._assert_complete(result, max_tokens)
        return result

    def _stream(self, payload, on_progress):
        emitted = []
        final = None
        with requests.post(
            self.lb_url + "/generate",
            json={**payload, "stream": True},
            stream=True,
            timeout=180,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines(chunk_size=1):
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                final = json.loads(line[6:])
                self.assertNotIn("error", final)
                output = final["output_ids"]
                self.assertEqual(len(output), final["meta_info"]["completion_tokens"])
                self.assertEqual(output[: len(emitted)], emitted)
                emitted = list(output)
                on_progress(final)
        self.assertIsNotNone(final)
        self._assert_complete(final, payload["sampling_params"]["max_new_tokens"])
        return final

    def test_first_handoff_boundaries_and_cold_cache_accounting(self):
        for prompt_len in (127, 128, 129, 255, 256, 257, 8192, 8193):
            with self.subTest(prompt_len=prompt_len):
                for url in (self.prefill_url, self.decode_url):
                    requests.post(url + "/flush_cache", timeout=30).raise_for_status()
                result = self._generate(prompt_len)
                self.assertEqual(result["meta_info"]["prompt_tokens"], prompt_len)
                self.assertEqual(result["meta_info"]["cached_tokens"], 0)

    def test_cache_only_replay_while_decode_is_running(self):
        ready = threading.Event()
        payload = self._payload(256, 512)
        self._post_decode("/slow_down", {"forward_sleep_time": 0.05})
        try:
            with ThreadPoolExecutor(max_workers=5) as pool:
                running = pool.submit(
                    self._stream,
                    payload,
                    lambda event: (
                        ready.set()
                        if event["meta_info"]["completion_tokens"] >= 8
                        else None
                    ),
                )
                self.assertTrue(ready.wait(120), "Decode produced no streaming tokens")
                self.assertFalse(running.done(), "Decode finished before replay burst")
                bursts = [pool.submit(self._generate, 8192) for _ in range(4)]
                for burst in bursts:
                    burst.result(timeout=180)
                self.assertFalse(running.done(), "Replay burst did not overlap Decode")
                self._post_decode("/slow_down", {"forward_sleep_time": 0.0})
                running.result(timeout=180)
        finally:
            self._post_decode("/slow_down", {"forward_sleep_time": 0.0})
            self._post_decode("/abort_request", {"rid": payload["rid"]})

    def test_retract_resume_rebootstraps_emitted_prefix(self):
        payload = self._payload(256, 256)
        paused = False
        retracted = False

        def retract_after_output(event):
            nonlocal paused, retracted
            meta = event["meta_info"]
            if not retracted and 8 <= meta["completion_tokens"] < 256:
                # Decode retract-pause holds a true P-side rebootstrap. Freeing
                # its cache before resume rules out reusing the old local KV.
                self._post_decode("/pause_generation", {"mode": "retract"})
                paused = True
                self._post_decode("/flush_cache", {})
                self._post_decode("/continue_generation", {"torch_empty_cache": False})
                paused = False
                retracted = True

        self._post_decode("/slow_down", {"forward_sleep_time": 0.02})
        try:
            result = self._stream(payload, retract_after_output)
            self.assertTrue(retracted, "No retraction was triggered during generation")
            self.assertGreaterEqual(result["meta_info"]["num_retractions"], 1)
        finally:
            if paused:
                self._post_decode("/continue_generation", {"torch_empty_cache": False})
            self._post_decode("/abort_request", {"rid": payload["rid"]})
            self._post_decode("/slow_down", {"forward_sleep_time": 0.0})
        self._generate(129)


if __name__ == "__main__":
    unittest.main()
