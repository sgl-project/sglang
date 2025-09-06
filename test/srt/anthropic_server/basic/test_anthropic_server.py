"""
Integration tests for Anthropic API endpoints.
Run with:
    python3 -m unittest anthropic_server.basic.test_anthropic_server.TestAnthropicServer.test_completion
    python3 -m unittest anthropic_server.basic.test_anthropic_server.TestAnthropicServer.test_completion_stream
    python3 -m unittest anthropic_server.basic.test_anthropic_server.TestAnthropicServer.test_chat_completion
    python3 -m unittest anthropic_server.basic.test_anthropic_server.TestAnthropicServer.test_chat_completion_stream
"""

import json
import re
import unittest

import requests

from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestAnthropicServer(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
        )
        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer(DEFAULT_SMALL_MODEL_NAME_FOR_TEST)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def run_completion(self, use_list_input=False):
        """Test basic completion functionality using messages endpoint."""
        # Anthropic API only has messages endpoint, not completions
        url = f"{self.base_url}/messages"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        data = {
            "model": self.model,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "The capital of France is"}],
            "temperature": 0,
        }

        response = requests.post(url, headers=headers, json=data)
        self.assertEqual(response.status_code, 200)

        result = response.json()

        # Verify response structure
        self.assertIn("id", result)
        self.assertIn("type", result)
        self.assertEqual(result["type"], "message")
        self.assertIn("content", result)
        self.assertIsInstance(result["content"], list)
        self.assertGreater(len(result["content"]), 0)
        self.assertIn("text", result["content"][0])

    def run_completion_stream(self, use_list_input=False):
        """Test streaming completion functionality using messages endpoint."""
        # Anthropic API only has messages endpoint, not completions
        url = f"{self.base_url}/messages"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        data = {
            "model": self.model,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "The capital of France is"}],
            "temperature": 0,
            "stream": True,
        }

        response = requests.post(url, headers=headers, json=data, stream=True)
        self.assertEqual(response.status_code, 200)

        # Collect streamed responses
        events = []
        for line in response.iter_lines():
            if line:
                if line.startswith(b"event: "):
                    event_type = line[7:].decode("utf-8")
                elif line.startswith(b"data: "):
                    data = line[6:]
                    if data != b"[DONE]":
                        event_data = json.loads(data)
                        events.append((event_type, event_data))

        # Verify we got some events
        self.assertGreater(len(events), 0)

        # Verify event structure
        event_types = [event[0] for event in events]
        self.assertIn("message_start", event_types)
        self.assertIn("content_block_start", event_types)
        self.assertIn("content_block_delta", event_types)
        self.assertIn("content_block_stop", event_types)
        self.assertIn("message_stop", event_types)

    def run_chat_completion(self):
        """Test basic chat completion functionality."""
        # Test Anthropic messages endpoint
        url = f"{self.base_url}/messages"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        data = {
            "model": self.model,
            "max_tokens": 64,
            "messages": [
                {
                    "role": "user",
                    "content": "What is the capital of France? Answer in a few words.",
                },
            ],
            "temperature": 0,
        }

        response = requests.post(url, headers=headers, json=data)
        self.assertEqual(response.status_code, 200)

        result = response.json()

        # Verify response structure
        self.assertIn("id", result)
        self.assertIn("type", result)
        self.assertEqual(result["type"], "message")
        self.assertIn("role", result)
        self.assertEqual(result["role"], "assistant")
        self.assertIn("content", result)
        self.assertIsInstance(result["content"], list)
        self.assertGreater(len(result["content"]), 0)
        self.assertIn("text", result["content"][0])
        self.assertIn("model", result)
        self.assertIn("stop_reason", result)
        self.assertIn("usage", result)
        self.assertIn("input_tokens", result["usage"])
        self.assertIn("output_tokens", result["usage"])

    def run_chat_completion_stream(self):
        """Test streaming chat completion functionality."""
        # Test Anthropic streaming messages endpoint
        url = f"{self.base_url}/messages"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        data = {
            "model": self.model,
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "What is the capital of France?"},
            ],
            "temperature": 0,
            "stream": True,
        }

        response = requests.post(url, headers=headers, json=data, stream=True)
        self.assertEqual(response.status_code, 200)

        # Collect streamed responses
        events = []
        for line in response.iter_lines():
            if line:
                if line.startswith(b"event: "):
                    event_type = line[7:].decode("utf-8")
                elif line.startswith(b"data: "):
                    data = line[6:]
                    if data != b"[DONE]":
                        event_data = json.loads(data)
                        events.append((event_type, event_data))

        # Verify we got some events
        self.assertGreater(len(events), 0)

        # Verify event structure
        event_types = [event[0] for event in events]
        self.assertIn("message_start", event_types)
        self.assertIn("content_block_start", event_types)
        self.assertIn("content_block_delta", event_types)
        self.assertIn("content_block_stop", event_types)
        self.assertIn("message_stop", event_types)

    def test_completion(self):
        """Test completion functionality."""
        for use_list_input in [False, True]:
            self.run_completion(use_list_input=use_list_input)

    def test_completion_stream(self):
        """Test streaming completion functionality."""
        for use_list_input in [False, True]:
            self.run_completion_stream(use_list_input=use_list_input)

    def test_chat_completion(self):
        """Test chat completion functionality."""
        self.run_chat_completion()

    def test_chat_completion_stream(self):
        """Test streaming chat completion functionality."""
        self.run_chat_completion_stream()

    def test_system_prompt(self):
        """Test system prompt functionality."""
        url = f"{self.base_url}/messages"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        data = {
            "model": self.model,
            "max_tokens": 32,
            "system": "You are a helpful assistant specialized in geography.",
            "messages": [
                {"role": "user", "content": "What is the capital of France?"},
            ],
            "temperature": 0,
        }

        response = requests.post(url, headers=headers, json=data)
        self.assertEqual(response.status_code, 200)

        result = response.json()
        self.assertEqual(result["type"], "message")
        self.assertEqual(result["role"], "assistant")
        self.assertIn("content", result)
        self.assertIsInstance(result["content"], list)
        self.assertGreater(len(result["content"]), 0)

    def test_stop_sequences(self):
        """Test stop sequences functionality."""
        url = f"{self.base_url}/messages"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        data = {
            "model": self.model,
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "Count from 1 to 10."},
            ],
            "stop_sequences": ["5"],
            "temperature": 0,
        }

        response = requests.post(url, headers=headers, json=data)
        self.assertEqual(response.status_code, 200)

        result = response.json()
        self.assertIn("stop_reason", result)
        # The response should stop at or before "5" due to stop sequence


if __name__ == "__main__":
    unittest.main(verbosity=2)
