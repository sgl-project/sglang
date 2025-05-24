"""
Usage:
python3 -m unittest test_globally_disable_thinking.TestGloballyDisableThinking.test_chat_completion_with_global_disable_thinking
python3 -m unittest test_globally_disable_thinking.TestGloballyDisableThinking.test_stream_chat_completion_with_global_disable_thinking
"""

import asyncio
import json
import os
import sys
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_ENABLE_THINKING_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestGloballyDisableThinking(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_ENABLE_THINKING_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-1234"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--reasoning-parser",
                "qwen3",
                "--disable-thinking",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_chat_completion_with_global_disable_thinking(self):
        # Test non-streaming with global disable_thinking=True, reasoning_content should be empty
        client = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0,
                "separate_reasoning": True,
            },
        )

        self.assertEqual(client.status_code, 200, f"Failed with: {client.text}")
        data = client.json()

        self.assertIn("choices", data)
        self.assertTrue(len(data["choices"]) > 0)
        self.assertIn("message", data["choices"][0])

        if "reasoning_content" in data["choices"][0]["message"]:
            self.assertIsNone(data["choices"][0]["message"]["reasoning_content"])

    def test_stream_chat_completion_with_global_disable_thinking(self):
        # Test streaming with global disable_thinking=True, reasoning_content should be empty
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0,
                "separate_reasoning": True,
                "stream": True,
            },
            stream=True,
        )

        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")

        has_reasoning = False
        has_content = False

        print("\n=== Stream With Global Disable Thinking ===")
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data:") and not line.startswith("data: [DONE]"):
                    data = json.loads(line[6:])
                    if "choices" in data and len(data["choices"]) > 0:
                        delta = data["choices"][0].get("delta", {})

                        if "reasoning_content" in delta and delta["reasoning_content"]:
                            has_reasoning = True

                        if "content" in delta and delta["content"]:
                            has_content = True

        self.assertFalse(
            has_reasoning,
            "The reasoning content should not be included in the stream response with global disable_thinking",
        )
        self.assertTrue(
            has_content, "The stream response does not contain normal content"
        )


if __name__ == "__main__":
    unittest.main()
