"""Basic decode correctness sanity kit.

Probes that catch the model producing wrong output: weight load
failure, sampling path bugs, KV / attention corruption, and cuda graph
edge cases. Single-prompt smoke only -- dataset-driven accuracy gates
belong to the consuming test class, not this kit.

Mix into any ``CustomTestCase`` subclass that exposes ``self.base_url``
and ``self.process``. Probes complete in well under a minute after
warmup."""

import requests

_REQUEST_TIMEOUT = 120


class BasicDecodeCorrectnessMixin:
    """Cheap output-quality probes."""

    sanity_max_new_tokens_short: int = 64
    sanity_max_new_tokens_long: int = 128

    def _decode_generate(self, prompt: str, max_new_tokens: int, stop=None) -> str:
        sampling_params = {"temperature": 0.0, "max_new_tokens": max_new_tokens}
        if stop is not None:
            sampling_params["stop"] = stop
        resp = requests.post(
            self.base_url + "/generate",
            json={"text": prompt, "sampling_params": sampling_params},
            timeout=_REQUEST_TIMEOUT,
        )
        self.assertEqual(resp.status_code, 200)
        return resp.json()["text"]

    def test_capital_france(self):
        out = self._decode_generate(
            "Q: What is the capital of France?\nA:",
            self.sanity_max_new_tokens_short,
        )
        self.assertIn("paris", out.lower())

    def test_basic_math(self):
        out = self._decode_generate(
            "Q: What is 17 multiplied by 23? Reply with just the number.\nA:",
            self.sanity_max_new_tokens_short,
        )
        self.assertIn("391", out)

    def test_color_completion(self):
        out = self._decode_generate(
            "Q: The three primary colors are red, blue, and ___. "
            "Fill in the blank.\nA:",
            self.sanity_max_new_tokens_short,
        )
        self.assertIn("yellow", out.lower())

    def test_ascii_ratio(self):
        # Language-agnostic gibberish detector. Healthy English output is
        # >90% printable ASCII; multilingual token salad / Unicode noise
        # from broken weight load drops well below 50%.
        out = self._decode_generate(
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
        out = self._decode_generate(
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

    def test_determinism_temp_zero(self):
        # temp=0 must be byte-identical across runs. Stop on "\n" so we
        # only compare the answer word; long continuations drift on
        # near-tie tokens (EP MoE / EAGLE spec) and aren't the point.
        prompt = "Q: What is the capital of France? Reply in one word.\nA:"
        out1 = self._decode_generate(
            prompt, self.sanity_max_new_tokens_short, stop=["\n"]
        )
        out2 = self._decode_generate(
            prompt, self.sanity_max_new_tokens_short, stop=["\n"]
        )
        self.assertEqual(
            out1.strip(),
            out2.strip(),
            f"temp=0 outputs diverged:\n  out1={out1!r}\n  out2={out2!r}",
        )

    def test_max_token_one(self):
        # Degenerate spec step. cuda-graph capture path bugs that only
        # fire on minimal-output requests.
        out = self._decode_generate(
            "Q: What is the capital of France? Just one word.\nA:",
            max_new_tokens=1,
        )
        self.assertGreater(len(out), 0)
