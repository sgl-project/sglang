"""E2E test: partial BF16 with INT4-only experts.

Launches Qwen3-30B-A3B with heter_precision_config where the first 24
layers have all experts marked INT4-only (no BF16 weights loaded).
The last 24 layers run full dual-precision (BF16 + INT4).

Sends a few general prompts to verify inference produces coherent output.
"""

import os
import unittest
from pathlib import Path

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

BF16_MODEL = "Qwen/Qwen3-30B-A3B"
HETER_CONFIG = str(Path(__file__).parent / "heter_config.json")


class TestPartialBF16(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            BF16_MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--heter-precision-config",
                HETER_CONFIG,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _generate(self, prompt, max_tokens=128):
        """Send a completion request and return the generated text."""
        resp = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": BF16_MODEL,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["text"]

    def test_simple_math(self):
        """Model can do basic arithmetic."""
        text = self._generate("What is 2 + 3? Answer with just the number:")
        print(f"\n[test_simple_math] prompt='What is 2 + 3? Answer with just the number:'\n  output={text!r}")
        self.assertIn("5", text)

    def test_general_knowledge(self):
        """Model produces coherent general knowledge response."""
        text = self._generate("The capital of France is")
        print(f"\n[test_general_knowledge] prompt='The capital of France is'\n  output={text!r}")
        self.assertIn("Paris", text)

    def test_longer_generation(self):
        """Model can generate longer coherent text without crashing."""
        text = self._generate(
            "Write a short paragraph about the benefits of exercise:",
            max_tokens=256,
        )
        print(f"\n[test_longer_generation] prompt='Write a short paragraph about the benefits of exercise:'\n  output={text!r}")
        # Should produce substantial output, not empty/garbage
        self.assertGreater(len(text.strip()), 50)
        # Basic coherence: should contain common English words
        text_lower = text.lower()
        self.assertTrue(
            any(w in text_lower for w in ["health", "exercise", "body", "physical", "fit"]),
            f"Generated text doesn't seem coherent: {text[:200]}",
        )


if __name__ == "__main__":
    unittest.main()
