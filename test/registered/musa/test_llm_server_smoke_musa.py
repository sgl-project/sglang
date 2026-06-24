import os
import unittest

import requests
import torch

from sglang.test.ci.ci_register import register_musa_ci
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_musa_ci(est_time=1200, suite="nightly-musa-1-gpu", nightly=True)


_REQUEST_TIMEOUT = 60


@unittest.skipIf(
    not (hasattr(torch, "musa") and torch.musa.is_available()),
    "MUSA device not available",
)
class TestMusaDeepSeekV2LiteChatServerSmoke(DefaultServerBase):
    """MUSA LLM server smoke test: launch, health check, and non-empty generation."""

    model = os.getenv("SGLANG_MUSA_LLM_MODEL", "deepseek-ai/DeepSeek-V2-Lite-Chat")
    served_model_name = "deepseek-v2-lite-chat"
    other_args = [
        "--trust-remote-code",
        "--served-model-name",
        served_model_name,
        "--attention-backend",
        "fa3",
        "--cuda-graph-max-bs",
        "32",
        "--tp-size",
        "1",
        "--chunked-prefill-size",
        "-1",
        "--disable-piecewise-cuda-graph",
        "--context-length",
        "4096",
        "--max-total-tokens",
        "8192",
        "--max-running-requests",
        "4",
    ]

    def test_health(self):
        resp = requests.get(self.base_url + "/health", timeout=10)
        self.assertEqual(resp.status_code, 200, resp.text)

    def test_health_generate(self):
        resp = requests.get(
            self.base_url + "/health_generate", timeout=_REQUEST_TIMEOUT
        )
        self.assertEqual(resp.status_code, 200, resp.text)

    def test_send_receive_chat_message_contains_beijing(self):
        resp = requests.post(
            self.base_url + "/v1/chat/completions",
            json={
                "model": self.served_model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "What is the capital of China? Answer in one word."
                        ),
                    },
                ],
                "temperature": 0.0,
                "max_tokens": 16,
            },
            timeout=_REQUEST_TIMEOUT,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertIn("choices", body)
        self.assertGreater(len(body["choices"]), 0)
        content = body["choices"][0]["message"]["content"]
        print(
            f"[MUSA Chat Completion] prompt='What is the capital of China? Answer in one word.' response={content!r}",
            flush=True,
        )
        self.assertIsInstance(content, str)
        self.assertGreater(len(content.strip()), 0)
        self.assertIn("beijing", content.strip().lower())

    def test_generate(self):
        resp = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 16,
                },
                "stream": False,
            },
            timeout=_REQUEST_TIMEOUT,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        if isinstance(body, list):
            self.assertGreater(len(body), 0)
            body = body[0]
        self.assertIn("text", body)
        self.assertIsInstance(body["text"], str)
        self.assertGreater(len(body["text"].strip()), 0)


if __name__ == "__main__":
    unittest.main()
