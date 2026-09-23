"""Real, multi-chunk PD inference with deferred destination KV allocation."""

import json
import math
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)

register_cuda_ci(est_time=480, stage="base-b", runner_config="2-gpu-large")


class _PrefillCompleteServer(PDDisaggregationServerBase):
    model = "Qwen/Qwen3-0.6B"

    model_revision = "c1899de289a04d12100db370d81485cdf75e47ca"

    policy = "prefill_complete"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        common = [
            "--revision",
            cls.model_revision,
            "--disaggregation-decode-allocation-policy",
            cls.policy,
            "--attention-backend",
            "flashinfer",
            "--mem-fraction-static",
            "0.65",
            "--max-total-tokens",
            "32768",
            "--context-length",
            "8192",
            "--max-running-requests",
            "16",
            "--cuda-graph-max-bs-decode",
            "16",
            "--enable-metrics",
        ]
        cls.extra_prefill_args = common + ["--chunked-prefill-size", "512"]
        cls.extra_decode_args = list(common)
        cls.launch_all()

    def generate(self, text, *, input_logprob=False):
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": text,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "return_input_logprob": input_logprob,
                "logprob_start_len": 0 if input_logprob else -1,
            },
            timeout=120,
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        meta = result["meta_info"]
        self.assertEqual(meta["completion_tokens"], 32, result)
        self.assertEqual(len(meta["output_token_logprobs"]), 32)
        self.assertTrue(
            all(math.isfinite(item[0]) for item in meta["output_token_logprobs"])
        )
        return result

    def _run_concurrent_requests(self):
        prompts = [
            f"Example {i}. " + "Explain how a compiler works. " * 120 for i in range(8)
        ]
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(self.generate, prompts))
        self.assertEqual(len(results), len(prompts))
        for process in (self.process_prefill, self.process_decode, self.process_lb):
            self.assertIsNone(process.poll())

    def _decode_load(self):
        response = requests.get(
            self.decode_url + "/v1/loads?include=core,disagg,queues", timeout=10
        )
        response.raise_for_status()
        loads = response.json()["loads"]
        self.assertEqual(len(loads), 1)
        return loads[0]

    def _flush_when_idle(self):
        for url in (self.prefill_url, self.decode_url):
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                response = requests.post(url + "/flush_cache", timeout=10)
                if response.status_code == 200:
                    break
                time.sleep(0.1)
            else:
                self.fail(f"Worker did not release the aborted request: {url}")


class TestPrefillCompleteAllocation(_PrefillCompleteServer):
    def test_multi_chunk_input_logprobs(self):
        prompt = "The capital of France is Paris. " * 300
        first = self.generate(prompt, input_logprob=True)
        second = self.generate(prompt, input_logprob=True)
        self.assertGreater(first["meta_info"]["prompt_tokens"], 512)
        for result in (first, second):
            self.assertGreaterEqual(
                len(result["meta_info"]["input_token_logprobs"]),
                result["meta_info"]["prompt_tokens"] - 1,
            )
        self.assertEqual(first["text"], second["text"])
        self.assertEqual(
            [x[1] for x in first["meta_info"]["output_token_logprobs"]],
            [x[1] for x in second["meta_info"]["output_token_logprobs"]],
        )

    def test_full_prefix_reuse(self):
        prompt = (
            "A compiler translates source code into executable instructions. " * 250
        )
        first = self.generate(prompt)
        second = self.generate(prompt)
        self.assertGreater(second["meta_info"]["prompt_tokens"], 512)
        self.assertGreaterEqual(
            second["meta_info"]["cached_tokens"],
            second["meta_info"]["prompt_tokens"] - 1,
        )
        self.assertEqual(first["text"], second["text"])
        self.assertEqual(
            [item[1] for item in first["meta_info"]["output_token_logprobs"]],
            [item[1] for item in second["meta_info"]["output_token_logprobs"]],
        )

    def test_concurrent_requests_complete(self):
        self._run_concurrent_requests()

    def test_decode_abort_before_allocation_releases_prefill(self):
        self._flush_when_idle()
        rid = "prefill-complete-abort-" + uuid.uuid4().hex
        response = requests.post(
            self.prefill_url + "/slow_down",
            json={"forward_sleep_time": 1.0},
            timeout=10,
        )
        response.raise_for_status()
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    requests.post,
                    self.lb_url + "/generate",
                    json={
                        "rid": rid,
                        "text": rid + " Explain how a compiler works." * 700,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 32,
                            "ignore_eos": True,
                        },
                    },
                    timeout=60,
                )
                try:
                    deadline = time.monotonic() + 15
                    while time.monotonic() < deadline:
                        load = self._decode_load()
                        if load["disaggregation"]["decode_prealloc_queue_reqs"] > 0:
                            break
                        self.assertFalse(
                            future.done(),
                            "Request finished before the gate was observed",
                        )
                        time.sleep(0.1)
                    else:
                        self.fail("Did not observe the decode readiness wait")
                    self.assertEqual(load["num_used_tokens"], 0, load)
                    self.assertEqual(load["queues"]["prealloc_ready"], 0, load)
                    # Only abort decode: the readiness CANCEL must release the
                    # source request even though no KV metadata was published.
                    abort = requests.post(
                        self.decode_url + "/abort_request",
                        json={"rid": rid},
                        timeout=10,
                    )
                    abort.raise_for_status()
                    result = future.result(timeout=30)
                    self.assertIn("abort", result.text.lower(), result.text)
                finally:
                    if not future.done():
                        for url in (self.prefill_url, self.decode_url):
                            requests.post(
                                url + "/abort_request", json={"rid": rid}, timeout=10
                            )
        finally:
            requests.post(
                self.prefill_url + "/slow_down",
                json={"forward_sleep_time": 0.0},
                timeout=10,
            ).raise_for_status()
        self._flush_when_idle()
        self.generate("The capital of France is")

    def test_decode_abort_after_transfer_remains_usable(self):
        rid = "prefill-complete-stream-abort-" + uuid.uuid4().hex
        with requests.post(
            self.lb_url + "/generate",
            json={
                "rid": rid,
                "text": "Explain how a compiler works in detail.",
                "stream": True,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 4096,
                    "ignore_eos": True,
                },
            },
            stream=True,
            timeout=60,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith(b"data: ") and line != b"data: [DONE]":
                    event = json.loads(line[6:])
                    if event["meta_info"]["completion_tokens"] > 0:
                        break
            else:
                self.fail("No token arrived before streaming abort")
            requests.post(
                self.decode_url + "/abort_request", json={"rid": rid}, timeout=10
            ).raise_for_status()
        self._flush_when_idle()
        self.generate("The capital of France is")

    def test_retract_resume_recomputes_before_reallocation(self):
        payload = {
            "text": "Explain how a compiler works in detail.",
            "return_logprob": True,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 256,
                "ignore_eos": True,
            },
        }
        reference = requests.post(self.lb_url + "/generate", json=payload, timeout=60)
        reference.raise_for_status()
        reference = reference.json()
        rid = "prefill-complete-rebootstrap-" + uuid.uuid4().hex
        requests.post(
            self.decode_url + "/slow_down",
            json={"forward_sleep_time": 0.01},
            timeout=10,
        ).raise_for_status()
        paused = False
        try:
            with requests.post(
                self.lb_url + "/generate",
                json={**payload, "rid": rid, "stream": True},
                stream=True,
                timeout=90,
            ) as response:
                response.raise_for_status()
                retracted = False
                final = None
                for line in response.iter_lines():
                    if not line.startswith(b"data: ") or line == b"data: [DONE]":
                        continue
                    final = json.loads(line[6:])
                    if not retracted and final["meta_info"]["completion_tokens"] > 0:
                        requests.post(
                            self.decode_url + "/pause_generation",
                            json={"mode": "retract"},
                            timeout=15,
                        ).raise_for_status()
                        paused = True
                        # Retract pause frees destination KV and stages a true
                        # rebootstrap; flush must succeed before resuming it.
                        requests.post(
                            self.decode_url + "/flush_cache", timeout=10
                        ).raise_for_status()
                        requests.post(
                            self.decode_url + "/continue_generation",
                            json={"torch_empty_cache": False},
                            timeout=15,
                        ).raise_for_status()
                        paused = False
                        retracted = True
                self.assertIsNotNone(final)
                self.assertGreater(final["meta_info"]["num_retractions"], 0, final)
                self.assertEqual(final["meta_info"]["completion_tokens"], 256, final)
                self.assertEqual(final["text"], reference["text"])
                self.assertEqual(
                    [item[1] for item in final["meta_info"]["output_token_logprobs"]],
                    [
                        item[1]
                        for item in reference["meta_info"]["output_token_logprobs"]
                    ],
                )
        finally:
            if paused:
                requests.post(
                    self.decode_url + "/continue_generation", json={}, timeout=15
                )
            for url in (self.prefill_url, self.decode_url):
                requests.post(url + "/abort_request", json={"rid": rid}, timeout=10)
            requests.post(
                self.decode_url + "/slow_down",
                json={"forward_sleep_time": 0.0},
                timeout=10,
            ).raise_for_status()
        self._flush_when_idle()


class TestPrefillCompleteRetry(_PrefillCompleteServer):
    extra_prefill_env = {"SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB": "1"}

    def _retry_count(self):
        response = requests.get(self.prefill_url + "/metrics", timeout=10)
        response.raise_for_status()
        return sum(
            float(line.split()[-1])
            for line in response.text.splitlines()
            if line.startswith("sglang:num_prefill_retries_total{")
        )

    def test_retry_after_attempt_budget_still_completes(self):
        # The policy resolves to one optimistic attempt. Force that first
        # attempt to yield, then verify real retries and eventual completion.
        before = self._retry_count()
        self._run_concurrent_requests()
        self.assertGreaterEqual(self._retry_count() - before, 8)
        self._flush_when_idle()


class TestPrefillCompleteTimeout(_PrefillCompleteServer):
    extra_decode_env = {"SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "3"}

    def test_readiness_timeout_releases_both_sides(self):
        self._flush_when_idle()
        requests.post(
            self.prefill_url + "/slow_down",
            json={"forward_sleep_time": 1.0},
            timeout=10,
        ).raise_for_status()
        start = time.monotonic()
        try:
            response = requests.post(
                self.lb_url + "/generate",
                json={
                    "rid": "prefill-complete-timeout-" + uuid.uuid4().hex,
                    "text": "Explain how a compiler works. " * 700,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 32,
                        "ignore_eos": True,
                    },
                },
                timeout=30,
            )
            self.assertIn("Timed out waiting for prefill completion", response.text)
            self.assertLess(time.monotonic() - start, 15)
        finally:
            requests.post(
                self.prefill_url + "/slow_down",
                json={"forward_sleep_time": 0.0},
                timeout=10,
            ).raise_for_status()
        # Do not explicitly abort either side: timeout handling must release
        # the source as well as the destination before cache flush can succeed.
        self._flush_when_idle()
        self.generate("The capital of France is")


if __name__ == "__main__":
    unittest.main()
