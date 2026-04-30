"""DSv4-Flash coherence sanity matrix — multiple topology fixtures share a
set of cheap prompts to catch silent-correctness (gibberish / repetition
collapse / encoding) regressions."""

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

DSV4_FLASH_MODEL_PATH = "sgl-project/DeepSeek-V4-Flash-FP8"

DSV4_FLASH_ENV = {
    "SGLANG_DSV4_FP4_EXPERTS": "0",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "1024",
}

DEEPEP_CONFIG = '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'


class _CoherenceSanityMixin:

    def _generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        resp = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": max_new_tokens,
                },
            },
            timeout=120,
        )
        self.assertEqual(resp.status_code, 200)
        return resp.json()["text"]

    def test_capital_france(self):
        out = self._generate("Q: What is the capital of France?\nA:")
        self.assertIn("paris", out.lower())

    def test_basic_math(self):
        out = self._generate(
            "Q: What is 17 multiplied by 23? Reply with just the number.\nA:"
        )
        self.assertIn("391", out)

    def test_color_completion(self):
        out = self._generate(
            "Q: The three primary colors are red, blue, and ___. "
            "Fill in the blank.\nA:"
        )
        self.assertIn("yellow", out.lower())

    def test_ascii_ratio(self):
        # Language-agnostic gibberish detector. Healthy English output is
        # >90% printable ASCII; multilingual token salad / Unicode noise
        # from broken weight load drops well below 50%.
        out = self._generate(
            "Write a single sentence about a sunny day in the park.",
            max_new_tokens=128,
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
        out = self._generate(
            "Briefly explain what gravity is.",
            max_new_tokens=128,
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
        out = self._generate(
            "Q: What is the capital of France? Just one word.\nA:",
            max_new_tokens=1,
        )
        self.assertGreater(len(out), 0)


def _launch(other_args, env_extra=None, timeout_mult=1):
    env = dict(DSV4_FLASH_ENV)
    if env_extra:
        env.update(env_extra)
    return popen_launch_server(
        DSV4_FLASH_MODEL_PATH,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * timeout_mult,
        other_args=other_args,
        env=env,
    )


_EAGLE_SPEC_ARGS = [
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
]


class TestDSv4FlashTP4DP4(_CoherenceSanityMixin, CustomTestCase):
    """TP4 + DP4 + deepep + EAGLE MTP."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = _launch(
            [
                "--trust-remote-code",
                "--tp",
                "4",
                "--dp",
                "4",
                "--enable-dp-attention",
                "--moe-a2a-backend",
                "deepep",
                "--cuda-graph-max-bs",
                "128",
                "--max-running-requests",
                "256",
                "--deepep-config",
                DEEPEP_CONFIG,
                "--mem-fraction-static",
                "0.7",
                *_EAGLE_SPEC_ARGS,
            ]
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestDSv4FlashTP4EP(_CoherenceSanityMixin, CustomTestCase):
    """TP attn + EP MoE (no DP attn) — exercises the DeepEP + TP-attn path."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = _launch(
            [
                "--trust-remote-code",
                "--tp",
                "4",
                "--ep",
                "4",
                # No --enable-dp-attention by design: covers TP-attn path.
                "--moe-a2a-backend",
                "deepep",
                "--cuda-graph-max-bs",
                "128",
                "--max-running-requests",
                "64",
                "--deepep-config",
                DEEPEP_CONFIG,
                "--mem-fraction-static",
                "0.7",
                *_EAGLE_SPEC_ARGS,
            ]
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestDSv4FlashTP4DP4ChunkedPrefillLarge(_CoherenceSanityMixin, CustomTestCase):
    """TP4 + DP4 with --chunked-prefill-size 16384 — large chunked prefill."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = _launch(
            [
                "--trust-remote-code",
                "--tp",
                "4",
                "--dp",
                "4",
                "--enable-dp-attention",
                "--moe-a2a-backend",
                "deepep",
                "--chunked-prefill-size",
                "16384",
                "--cuda-graph-max-bs",
                "128",
                "--max-running-requests",
                "256",
                "--deepep-config",
                DEEPEP_CONFIG,
                "--mem-fraction-static",
                "0.7",
                *_EAGLE_SPEC_ARGS,
            ]
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestDSv4FlashTP8NoSpec(_CoherenceSanityMixin, CustomTestCase):
    """TP8, no spec decoding."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = _launch(
            [
                "--trust-remote-code",
                "--tp",
                "8",
                "--max-running-requests",
                "8",
                "--mem-fraction-static",
                "0.85",
            ]
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
