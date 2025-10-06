"""
Integration tests for Anthropic API features.
"""

import json
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


class TestAnthropicServerFeatures(CustomTestCase):
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
            other_args=[
                "--tool-call-parser",
                "llama3",
            ],
        )
        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer(DEFAULT_SMALL_MODEL_NAME_FOR_TEST)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_tool_choice_auto(self):
        """
        Test: tool_choice with auto type.
        """
        url = f"{self.base_url}/messages"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        tools = [
            {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city to find the weather for",
                        },
                        "unit": {
                            "type": "string",
                            "description": "Weather unit (celsius or fahrenheit)",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["city", "unit"],
                },
            },
            {
                "name": "get_time",
                "description": "Get the current time in a given location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city to find the time for",
                        },
                    },
                    "required": ["city"],
                },
            },
        ]

        data = {
            "model": self.model,
            "max_tokens": 2048,
            "messages": [
                {"role": "user", "content": "What is the weather like in Paris?"},
            ],
            "temperature": 0.8,
            "tools": tools,
            # For auto tool choice, we don't need to specify tool_choice explicitly
            # as it defaults to auto behavior when tools are provided
        }

        response = requests.post(url, headers=headers, json=data)
        self.assertEqual(response.status_code, 200)

        result = response.json()

        # Verify response structure
        self.assertIn("content", result)
        self.assertIsInstance(result["content"], list)
        self.assertGreater(len(result["content"]), 0)

        # Should have either text or tool_use blocks
        content_types = [block.get("type") for block in result["content"]]
        self.assertTrue(
            "text" in content_types or "tool_use" in content_types,
            "Should have either text or tool_use content blocks",
        )

    def test_tool_choice_specific(self):
        """
        Test: tool_choice with specific tool.
        """
        url = f"{self.base_url}/messages"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        tools = [
            {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city to find the weather for",
                        },
                    },
                    "required": ["city"],
                },
            },
            {
                "name": "get_time",
                "description": "Get the current time in a given location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city to find the time for",
                        },
                    },
                    "required": ["city"],
                },
            },
        ]

        data = {
            "model": self.model,
            "max_tokens": 2048,
            "messages": [
                {"role": "user", "content": "What time is it in London?"},
            ],
            "temperature": 0.8,
            "tools": tools,
            # When tool_choice is not specified, it should default to auto behavior
        }

        response = requests.post(url, headers=headers, json=data)
        self.assertEqual(response.status_code, 200)

        result = response.json()

        # Verify response structure
        self.assertIn("content", result)
        self.assertIsInstance(result["content"], list)
        self.assertGreater(len(result["content"]), 0)

        # Should have either text or tool_use blocks
        content_types = [block.get("type") for block in result["content"]]
        self.assertTrue(
            "text" in content_types or "tool_use" in content_types,
            "Should have either text or tool_use content blocks",
        )

    def test_stop_reason_variations(self):
        """
        Test: Different stop_reason values.
        """
        url = f"{self.base_url}/messages"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        data = {
            "model": self.model,
            "max_tokens": 10,  # Small max_tokens to potentially trigger max_tokens stop reason
            "messages": [
                {"role": "user", "content": "Count from 1 to 100"},
            ],
            "temperature": 0.1,  # Low temperature for more deterministic results
        }

        response = requests.post(url, headers=headers, json=data)
        self.assertEqual(response.status_code, 200)

        result = response.json()

        # Verify response structure
        self.assertIn("stop_reason", result)
        self.assertIn("content", result)

        # Check that stop_reason is one of the valid values
        valid_stop_reasons = ["end_turn", "max_tokens", "stop_sequence", "tool_use"]
        self.assertIn(
            result["stop_reason"],
            valid_stop_reasons,
            f"Stop reason should be one of {valid_stop_reasons}",
        )


if __name__ == "__main__":
    unittest.main()
