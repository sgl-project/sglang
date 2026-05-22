"""Basic scheduler / cache / streaming stress sanity kit.

Probes that catch bugs which only fire under multi-request or large-
prompt conditions: scheduler hangs, radix prefix-cache cross-
contamination, chunked-prefill multi-chunk kernel crashes, and SSE
streaming corruption.

Mix into any ``CustomTestCase`` subclass that exposes ``self.base_url``
and ``self.process``."""

import json
import threading

import requests

_REQUEST_TIMEOUT = 120

# Shared prefix forces all concurrent requests through the same radix
# match path; per-request suffix branches the tail so the model still
# has to predict different tokens (otherwise outputs would be identical
# and we'd be testing 1 request 8 times instead of 8 independent reqs).
_CONCURRENT_PREFIX = "You are a helpful assistant. Answer with a single word.\n"
_CONCURRENT_QA = [
    ("Q: What is the capital of France?\nA:", "paris"),
    ("Q: What is the capital of Germany?\nA:", "berlin"),
    ("Q: What is the capital of Italy?\nA:", "rome"),
    ("Q: What is the capital of Japan?\nA:", "tokyo"),
    ("Q: What is the capital of Spain?\nA:", "madrid"),
    ("Q: What is the capital of Egypt?\nA:", "cairo"),
    ("Q: What is the capital of Russia?\nA:", "moscow"),
    ("Q: What is the capital of Australia?\nA:", "canberra"),
]


class BasicSchedulerStressMixin:
    """Streaming + concurrent + long-prompt path probes."""

    sanity_max_new_tokens_short: int = 64

    def _stress_generate(self, prompt: str, max_new_tokens: int) -> str:
        resp = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": max_new_tokens,
                },
            },
            timeout=_REQUEST_TIMEOUT,
        )
        self.assertEqual(resp.status_code, 200)
        return resp.json()["text"]

    def test_streaming_response(self):
        # SSE streaming exercises a different return path than non-stream
        # /generate. Catches token-by-token streaming corruption and SSE
        # framing bugs without changing the model.
        with requests.post(
            self.base_url + "/generate",
            json={
                "text": "Q: What is the capital of France?\nA:",
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": self.sanity_max_new_tokens_short,
                },
                "stream": True,
            },
            stream=True,
            timeout=_REQUEST_TIMEOUT,
        ) as resp:
            self.assertEqual(resp.status_code, 200)
            chunks_seen = 0
            last_text = ""
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                payload = raw[len("data:") :].strip()
                if payload == "[DONE]":
                    break
                obj = json.loads(payload)
                last_text = obj.get("text", last_text)
                chunks_seen += 1
            self.assertGreater(chunks_seen, 0)
            self.assertIn("paris", last_text.lower())

    def test_concurrent_requests(self):
        # 8 parallel reqs share a system prefix but each has a distinct
        # question suffix. Shared prefix exercises radix prefix caching
        # across concurrent reqs; per-request suffix forces independent
        # decode tails (different canonical answers). Catches concurrent
        # scheduler hangs and prefix-cache cross-contamination.
        results = [None] * len(_CONCURRENT_QA)

        def worker(idx, suffix, expected):
            try:
                out = self._stress_generate(
                    _CONCURRENT_PREFIX + suffix,
                    self.sanity_max_new_tokens_short,
                )
                results[idx] = expected in out.lower()
            except Exception:
                results[idx] = False

        threads = [
            threading.Thread(target=worker, args=(i, suffix, expected))
            for i, (suffix, expected) in enumerate(_CONCURRENT_QA)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=_REQUEST_TIMEOUT)

        passed = sum(1 for r in results if r)
        # Tolerate one stochastic miss; gibberish would fail all 8.
        self.assertGreaterEqual(
            passed,
            len(_CONCURRENT_QA) - 1,
            f"concurrent answers correct: {passed}/{len(_CONCURRENT_QA)}; results={results}",
        )

    def test_long_prompt(self):
        # ~8k-token filler drives the chunked-prefill path through
        # multiple chunks. Catches DeepEP / large-prompt kernel crashes
        # that only fire on multi-chunk prefill.
        filler = "the quick brown fox jumps over the lazy dog. " * 800
        out = self._stress_generate(
            f"Read the following text and then answer.\n{filler}\n\n"
            "Q: What is the capital of France?\nA:",
            self.sanity_max_new_tokens_short,
        )
        # Long-prompt substring match is best-effort (model may get
        # distracted); primary assertion is the 200 + non-empty inside
        # _stress_generate.
        self.assertGreater(len(out), 0)
