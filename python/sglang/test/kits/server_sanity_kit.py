"""Black-box server sanity prompts: cheap checks that catch silent
correctness regressions (gibberish / repetition collapse / encoding),
streaming/concurrent path bugs, and endpoint health.

Mix into any ``CustomTestCase`` subclass that exposes ``self.base_url``
and ``self.process``. Each test is independent and fast (≤ 5 s after
warmup); the whole kit completes in < 1 min."""

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


class ServerSanityMixin:
    """12 cheap black-box probes for silent-correctness / hang / endpoint
    regressions."""

    sanity_max_new_tokens_short: int = 64
    sanity_max_new_tokens_long: int = 128

    def _sanity_generate(self, prompt: str, max_new_tokens: int, stop=None) -> str:
        sampling_params = {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
        }
        if stop is not None:
            sampling_params["stop"] = stop
        resp = requests.post(
            self.base_url + "/generate",
            json={"text": prompt, "sampling_params": sampling_params},
            timeout=_REQUEST_TIMEOUT,
        )
        self.assertEqual(resp.status_code, 200)
        return resp.json()["text"]

    def test_health(self):
        # Cheapest possible alive check; FastAPI route alone.
        resp = requests.get(self.base_url + "/health", timeout=10)
        self.assertEqual(resp.status_code, 200)

    def test_health_generate(self):
        # sglang's built-in minimal-forward sanity. 200 only if the
        # scheduler can complete one prefill+decode end to end.
        resp = requests.get(self.base_url + "/health_generate", timeout=60)
        self.assertEqual(resp.status_code, 200)

    def test_capital_france(self):
        out = self._sanity_generate(
            "Q: What is the capital of France?\nA:",
            self.sanity_max_new_tokens_short,
        )
        self.assertIn("paris", out.lower())

    def test_basic_math(self):
        out = self._sanity_generate(
            "Q: What is 17 multiplied by 23? Reply with just the number.\nA:",
            self.sanity_max_new_tokens_short,
        )
        self.assertIn("391", out)

    def test_color_completion(self):
        out = self._sanity_generate(
            "Q: The three primary colors are red, blue, and ___. "
            "Fill in the blank.\nA:",
            self.sanity_max_new_tokens_short,
        )
        self.assertIn("yellow", out.lower())

    def test_ascii_ratio(self):
        # Language-agnostic gibberish detector. Healthy English output is
        # >90% printable ASCII; multilingual token salad / Unicode noise
        # from broken weight load drops well below 50%.
        out = self._sanity_generate(
            "Write a single sentence about a sunny day in the park.",
            self.sanity_max_new_tokens_long,
        )
        printable = sum(1 for c in out if 32 <= ord(c) < 127 or c in "\n\t")
        ratio = printable / max(len(out), 1)
        self.assertGreater(
            ratio,
            0.85,
            f"output looks like gibberish (printable ASCII ratio={ratio:.2f}): {out!r}",
        )

    def test_no_repetition_blowup(self):
        # KV-cache / attn corruption often manifests as the model getting
        # stuck looping the same n-gram.
        out = self._sanity_generate(
            "Briefly explain what gravity is.",
            self.sanity_max_new_tokens_long,
        )
        if len(out) >= 50:
            windows = [out[i : i + 5] for i in range(len(out) - 5)]
            most_common_count = max((windows.count(w) for w in set(windows)), default=0)
            ratio = most_common_count / len(windows)
            self.assertLess(
                ratio,
                0.25,
                f"output appears to repeat heavily (top 5-gram ratio={ratio:.2f}): {out!r}",
            )

    def test_max_token_one(self):
        # Degenerate spec step. cuda-graph capture path bugs that only
        # fire on minimal-output requests.
        out = self._sanity_generate(
            "Q: What is the capital of France? Just one word.\nA:",
            max_new_tokens=1,
        )
        self.assertGreater(len(out), 0)

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
                out = self._sanity_generate(
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
        out = self._sanity_generate(
            f"Read the following text and then answer.\n{filler}\n\n"
            "Q: What is the capital of France?\nA:",
            self.sanity_max_new_tokens_short,
        )
        # Long-prompt substring match is best-effort (model may get
        # distracted); primary assertion is the 200 + non-empty inside
        # _sanity_generate.
        self.assertGreater(len(out), 0)

    def test_determinism_temp_zero(self):
        # temp=0 must be byte-identical across runs. Stop on "\n" so we
        # only compare the answer word; long continuations drift on
        # near-tie tokens (EP MoE / EAGLE spec) and aren't the point.
        prompt = "Q: What is the capital of France? Reply in one word.\nA:"
        out1 = self._sanity_generate(
            prompt, self.sanity_max_new_tokens_short, stop=["\n"]
        )
        # Second call exercises cache-hit path.
        out2 = self._sanity_generate(
            prompt, self.sanity_max_new_tokens_short, stop=["\n"]
        )
        self.assertEqual(
            out1.strip(),
            out2.strip(),
            f"temp=0 outputs diverged:\n  out1={out1!r}\n  out2={out2!r}",
        )
