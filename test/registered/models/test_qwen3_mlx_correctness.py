import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

MODEL_PATH = "Qwen/Qwen3-0.6B"


class TestQwen3MlxCorrectness(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL_PATH)
        cls.base_url = DEFAULT_URL_FOR_TEST

        env = os.environ.copy()
        env["SGLANG_USE_MLX"] = "1"

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "1",
                "--disable-radix-cache",
                "--disable-cuda-graph",
            ],
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process is not None:
            kill_process_tree(cls.process.pid)

    def _chat(self, messages, max_tokens=32, temperature=0):
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": MODEL_PATH,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {
                    "enable_thinking": False
                },
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def test_basic_generation_nonempty(self):
        text = self._chat(
            [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "Say hello briefly."},
            ],
            max_tokens=16,
            temperature=0,
        )
        self.assertTrue(len(text) > 0)
        self.assertIsInstance(text, str)

    def test_simple_fact_sanity(self):
        text = self._chat(
            [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "What is 2+2? Reply briefly."},
            ],
            max_tokens=8,
            temperature=0,
        )
        self.assertTrue(len(text) > 0)
        self.assertIn("4", text)


if __name__ == "__main__":
    unittest.main()
