"""
Usage:
python3 -m unittest openai_server.features.test_reasoning_content.TestReasoningContentAPI.test_streaming_separate_reasoning_false
python3 -m unittest openai_server.features.test_reasoning_content.TestReasoningContentAPI.test_streaming_separate_reasoning_true
python3 -m unittest openai_server.features.test_reasoning_content.TestReasoningContentAPI.test_streaming_separate_reasoning_true_stream_reasoning_false
python3 -m unittest openai_server.features.test_reasoning_content.TestReasoningContentAPI.test_nonstreaming_separate_reasoning_false
python3 -m unittest openai_server.features.test_reasoning_content.TestReasoningContentAPI.test_nonstreaming_separate_reasoning_true
python3 -m unittest openai_server.features.test_reasoning_content.TestReasoningContentAPI.test_reasoning_tokens_in_usage
python3 -m unittest openai_server.features.test_reasoning_content.TestReasoningContentStartup.test_nonstreaming
python3 -m unittest openai_server.features.test_reasoning_content.TestReasoningContentStartup.test_streaming
"""

import json
import unittest

import openai
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_REASONING_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestReasoningContentAPI(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_REASONING_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-1234"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--reasoning-parser",
                "deepseek-r1",
            ],
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_streaming_separate_reasoning_false(self):
        # Test streaming with separate_reasoning=False, reasoning_content should be empty
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 1+3?",
                }
            ],
            "max_tokens": 100,
            "stream": True,
            "extra_body": {"separate_reasoning": False},
        }
        response = client.chat.completions.create(**payload)

        reasoning_content = ""
        content = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
            elif chunk.choices[0].delta.reasoning_content:
                reasoning_content += chunk.choices[0].delta.reasoning_content

        assert len(reasoning_content) == 0
        assert len(content) > 0

    def test_streaming_separate_reasoning_true(self):
        # Test streaming with separate_reasoning=True, reasoning_content should not be empty
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 1+3?",
                }
            ],
            "max_tokens": 100,
            "stream": True,
            "stream_options": {"include_usage": True},
            "extra_body": {"separate_reasoning": True},
        }
        response = client.chat.completions.create(**payload)

        reasoning_content = ""
        content = ""
        usage_found = False
        for chunk in response:
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
            elif chunk.choices[0].delta.reasoning_content:
                reasoning_content += chunk.choices[0].delta.reasoning_content

            # Check usage chunk for reasoning_tokens
            if chunk.usage:
                usage_found = True
                assert chunk.usage.completion_tokens > 0
                # Check if completion_tokens_details exists and has reasoning_tokens
                if (
                    hasattr(chunk.usage, "completion_tokens_details")
                    and chunk.usage.completion_tokens_details
                ):
                    if hasattr(
                        chunk.usage.completion_tokens_details, "reasoning_tokens"
                    ):
                        reasoning_tokens = (
                            chunk.usage.completion_tokens_details.reasoning_tokens
                        )
                        assert (
                            reasoning_tokens is not None and reasoning_tokens > 0
                        ), f"Expected reasoning_tokens > 0, got {reasoning_tokens}"

        assert len(reasoning_content) > 0
        assert len(content) > 0
        assert usage_found, "Usage chunk should be present"

    def test_streaming_separate_reasoning_true_stream_reasoning_false(self):
        # Test streaming with separate_reasoning=True, reasoning_content should not be empty
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 1+3?",
                }
            ],
            "max_tokens": 100,
            "stream": True,
            "extra_body": {"separate_reasoning": True, "stream_reasoning": False},
        }
        response = client.chat.completions.create(**payload)

        reasoning_content = ""
        content = ""
        first_chunk = False
        for chunk in response:
            if chunk.choices[0].delta.reasoning_content:
                reasoning_content = chunk.choices[0].delta.reasoning_content
                first_chunk = True
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
                if not first_chunk:
                    reasoning_content = chunk.choices[0].delta.reasoning_content
                first_chunk = True
            if not first_chunk:
                assert (
                    not chunk.choices[0].delta.reasoning_content
                    or len(chunk.choices[0].delta.reasoning_content) == 0
                )
        assert len(reasoning_content) > 0
        assert len(content) > 0

    def test_nonstreaming_separate_reasoning_false(self):
        # Test non-streaming with separate_reasoning=False, reasoning_content should be empty
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 1+3?",
                }
            ],
            "max_tokens": 100,
            "extra_body": {"separate_reasoning": False},
        }
        response = client.chat.completions.create(**payload)

        assert (
            not response.choices[0].message.reasoning_content
            or len(response.choices[0].message.reasoning_content) == 0
        )
        assert len(response.choices[0].message.content) > 0

    def test_nonstreaming_separate_reasoning_true(self):
        # Test non-streaming with separate_reasoning=True, reasoning_content should not be empty
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 1+3?",
                }
            ],
            "max_tokens": 100,
            "extra_body": {"separate_reasoning": True},
        }
        response = client.chat.completions.create(**payload)

        assert len(response.choices[0].message.reasoning_content) > 0
        assert len(response.choices[0].message.content) > 0

        # Check usage has reasoning_tokens
        assert response.usage is not None
        assert response.usage.completion_tokens > 0
        if (
            hasattr(response.usage, "completion_tokens_details")
            and response.usage.completion_tokens_details
        ):
            if hasattr(response.usage.completion_tokens_details, "reasoning_tokens"):
                reasoning_tokens = (
                    response.usage.completion_tokens_details.reasoning_tokens
                )
                assert (
                    reasoning_tokens is not None and reasoning_tokens > 0
                ), f"Expected reasoning_tokens > 0, got {reasoning_tokens}"

    def test_reasoning_tokens_in_usage(self):
        # Test that reasoning_tokens field exists in usage for reasoning models
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 2+2? Think step by step.",
                }
            ],
            "max_tokens": 300,
        }
        response = client.chat.completions.create(**payload)

        # Verify usage field exists
        assert response.usage is not None, "Usage field should be present"
        assert response.usage.prompt_tokens > 0, "Prompt tokens should be > 0"
        assert response.usage.completion_tokens > 0, "Completion tokens should be > 0"
        assert response.usage.total_tokens > 0, "Total tokens should be > 0"

        # Verify completion_tokens_details and reasoning_tokens exist
        assert hasattr(
            response.usage, "completion_tokens_details"
        ), "completion_tokens_details should exist"

        assert (
            response.usage.completion_tokens_details is not None
        ), "completion_tokens_details should not be None"

        assert hasattr(
            response.usage.completion_tokens_details, "reasoning_tokens"
        ), "reasoning_tokens field should exist in completion_tokens_details"

        reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens
        assert (
            reasoning_tokens is not None and reasoning_tokens > 0
        ), f"Expected reasoning_tokens > 0 with reasoning parser enabled, got {reasoning_tokens}"


class TestReasoningContentWithoutParser(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_REASONING_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-1234"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[],  # No reasoning parser
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_streaming_separate_reasoning_false(self):
        # Test streaming with separate_reasoning=False, reasoning_content should be empty
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 1+3?",
                }
            ],
            "max_tokens": 100,
            "stream": True,
            "extra_body": {"separate_reasoning": False},
        }
        response = client.chat.completions.create(**payload)

        reasoning_content = ""
        content = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
            elif chunk.choices[0].delta.reasoning_content:
                reasoning_content += chunk.choices[0].delta.reasoning_content

        assert len(reasoning_content) == 0
        assert len(content) > 0

    def test_streaming_separate_reasoning_true(self):
        # Test streaming with separate_reasoning=True, reasoning_content should not be empty
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 1+3?",
                }
            ],
            "max_tokens": 100,
            "stream": True,
            "extra_body": {"separate_reasoning": True},
        }
        response = client.chat.completions.create(**payload)

        reasoning_content = ""
        content = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
            elif chunk.choices[0].delta.reasoning_content:
                reasoning_content += chunk.choices[0].delta.reasoning_content

        assert len(reasoning_content) == 0
        assert len(content) > 0

    def test_streaming_separate_reasoning_true_stream_reasoning_false(self):
        # Test streaming with separate_reasoning=True, reasoning_content should not be empty
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 1+3?",
                }
            ],
            "max_tokens": 100,
            "stream": True,
            "extra_body": {"separate_reasoning": True, "stream_reasoning": False},
        }
        response = client.chat.completions.create(**payload)

        reasoning_content = ""
        content = ""
        first_chunk = False
        for chunk in response:
            if chunk.choices[0].delta.reasoning_content:
                reasoning_content = chunk.choices[0].delta.reasoning_content
                first_chunk = True
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
                if not first_chunk:
                    reasoning_content = chunk.choices[0].delta.reasoning_content
                first_chunk = True
            if not first_chunk:
                assert (
                    not chunk.choices[0].delta.reasoning_content
                    or len(chunk.choices[0].delta.reasoning_content) == 0
                )
        assert not reasoning_content or len(reasoning_content) == 0
        assert len(content) > 0

    def test_nonstreaming_separate_reasoning_false(self):
        # Test non-streaming with separate_reasoning=False, reasoning_content should be empty
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 1+3?",
                }
            ],
            "max_tokens": 100,
            "extra_body": {"separate_reasoning": False},
        }
        response = client.chat.completions.create(**payload)

        assert (
            not response.choices[0].message.reasoning_content
            or len(response.choices[0].message.reasoning_content) == 0
        )
        assert len(response.choices[0].message.content) > 0

    def test_nonstreaming_separate_reasoning_true(self):
        # Test non-streaming with separate_reasoning=True, reasoning_content should not be empty
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 1+3?",
                }
            ],
            "max_tokens": 100,
            "extra_body": {"separate_reasoning": True},
        }
        response = client.chat.completions.create(**payload)

        assert (
            not response.choices[0].message.reasoning_content
            or len(response.choices[0].message.reasoning_content) == 0
        )
        assert len(response.choices[0].message.content) > 0


if __name__ == "__main__":
    unittest.main()
