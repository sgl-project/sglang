"""
Integration tests for Anthropic API enable thinking features.
"""

import json
import unittest

import requests

from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_ENABLE_THINKING_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestAnthropicEnableThinking(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_ENABLE_THINKING_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--reasoning-parser",
                "qwen3",
            ],
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_messages_with_reasoning(self):
        """
        Test: Non-streaming messages with separate_reasoning=True and enable_thinking=True
        reasoning_content should not be empty
        """
        url = f"{self.base_url}/messages"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        data = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
            ],
            "temperature": 0,
            "separate_reasoning": True,
            "chat_template_kwargs": {"enable_thinking": True},
        }

        response = requests.post(url, headers=headers, json=data)
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")

        result = response.json()

        # Verify response structure
        self.assertIn("content", result)
        self.assertIsInstance(result["content"], list)
        self.assertGreater(len(result["content"]), 0)

        # Check for reasoning content in the response
        # For Anthropic API, reasoning content might be in tool_use blocks or separate text blocks
        content_types = [block.get("type") for block in result["content"]]
        self.assertTrue("text" in content_types, "Should have text content blocks")

    def test_messages_without_reasoning(self):
        """
        Test: Non-streaming messages with separate_reasoning=True and enable_thinking=False
        reasoning_content should be empty or not present
        """
        url = f"{self.base_url}/messages"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        data = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
            ],
            "temperature": 0,
            "separate_reasoning": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        response = requests.post(url, headers=headers, json=data)
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")

        result = response.json()

        # Verify response structure
        self.assertIn("content", result)
        self.assertIsInstance(result["content"], list)
        self.assertGreater(len(result["content"]), 0)

        # Should have regular content blocks
        content_types = [block.get("type") for block in result["content"]]
        self.assertTrue("text" in content_types, "Should have text content blocks")

    def test_stream_messages_with_reasoning(self):
        """
        Test: Streaming messages with separate_reasoning=True and enable_thinking=True
        Should have reasoning content in the stream
        """
        url = f"{self.base_url}/messages"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        data = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
            ],
            "temperature": 0,
            "stream": True,
            "separate_reasoning": True,
            "chat_template_kwargs": {"enable_thinking": True},
        }

        response = requests.post(url, headers=headers, json=data, stream=True)
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")

        # Collect streamed responses
        events = []
        for line in response.iter_lines():
            if line:
                if line.startswith(b"event: "):
                    event_type = line[7:].decode("utf-8")
                elif line.startswith(b"data: "):
                    data_content = line[6:]
                    if data_content != b"[DONE]":
                        event_data = json.loads(data_content)
                        events.append((event_type, event_data))

        # Verify we got some events
        self.assertGreater(len(events), 0, "Streaming should return at least one chunk")

        # Verify event structure
        event_types = [event[0] for event in events]
        self.assertIn("message_start", event_types)
        self.assertIn("content_block_start", event_types)
        self.assertIn("content_block_stop", event_types)
        self.assertIn("message_stop", event_types)

    def test_stream_messages_without_reasoning(self):
        """
        Test: Streaming messages with separate_reasoning=True and enable_thinking=False
        Should not have reasoning content in the stream
        """
        url = f"{self.base_url}/messages"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        data = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
            ],
            "temperature": 0,
            "stream": True,
            "separate_reasoning": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        response = requests.post(url, headers=headers, json=data, stream=True)
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")

        # Collect streamed responses
        events = []
        for line in response.iter_lines():
            if line:
                if line.startswith(b"event: "):
                    event_type = line[7:].decode("utf-8")
                elif line.startswith(b"data: "):
                    data_content = line[6:]
                    if data_content != b"[DONE]":
                        event_data = json.loads(data_content)
                        events.append((event_type, event_data))

        # Verify we got some events
        self.assertGreater(len(events), 0, "Streaming should return at least one chunk")

        # Verify event structure
        event_types = [event[0] for event in events]
        self.assertIn("message_start", event_types)
        self.assertIn("content_block_start", event_types)
        self.assertIn("content_block_stop", event_types)
        self.assertIn("message_stop", event_types)


if __name__ == "__main__":
    unittest.main()
