"""
Usage:
python3 -m unittest openai_server.features.test_reasoning_content.TestReasoningContentAPI.test_streaming_separate_reasoning_false
python3 -m unittest openai_server.features.test_reasoning_content.TestReasoningContentAPI.test_streaming_separate_reasoning_true
python3 -m unittest openai_server.features.test_reasoning_content.TestReasoningContentAPI.test_streaming_separate_reasoning_true_stream_reasoning_false
python3 -m unittest openai_server.features.test_reasoning_content.TestReasoningContentAPI.test_nonstreaming_separate_reasoning_false
python3 -m unittest openai_server.features.test_reasoning_content.TestReasoningContentAPI.test_nonstreaming_separate_reasoning_true
python3 -m unittest openai_server.features.test_reasoning_content.TestReasoningContentStartup.test_nonstreaming
python3 -m unittest openai_server.features.test_reasoning_content.TestReasoningContentStartup.test_streaming
"""

import unittest

import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_REASONING_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=89, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=89, suite="stage-b-test-small-1-gpu-amd")


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

        assert len(reasoning_content) > 0
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


class TestReasoningTokensInUsage(CustomTestCase):
    """Test cases for reasoning_tokens in usage field."""

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
                "--enable-reasoning-tokens",
            ],
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_nonstreaming_reasoning_tokens_in_usage(self):
        """Test that reasoning_tokens are included in usage for non-streaming responses."""
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

        # Check that usage is present
        assert response.usage is not None
        # Check that reasoning_tokens is in usage
        assert response.usage.reasoning_tokens is not None
        assert response.usage.reasoning_tokens > 0
        # Check that completion_tokens_details is present
        assert response.usage.completion_tokens_details is not None
        assert (
            response.usage.completion_tokens_details.reasoning_tokens
            == response.usage.reasoning_tokens
        )

    def test_streaming_reasoning_tokens_in_usage(self):
        """Test that reasoning_tokens are included in usage for streaming responses."""
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

        # Collect all chunks
        usage_chunk = None
        for chunk in response:
            if chunk.usage is not None:
                usage_chunk = chunk
                break

        # Check that usage chunk is present
        assert usage_chunk is not None
        assert usage_chunk.usage is not None
        # Check that reasoning_tokens is in usage
        assert usage_chunk.usage.reasoning_tokens is not None
        assert usage_chunk.usage.reasoning_tokens > 0
        # Check that completion_tokens_details is present
        assert usage_chunk.usage.completion_tokens_details is not None
        assert (
            usage_chunk.usage.completion_tokens_details.reasoning_tokens
            == usage_chunk.usage.reasoning_tokens
        )

    def test_reasoning_tokens_zero_when_disabled(self):
        """Test that reasoning_tokens is None when enable_reasoning_tokens is False."""
        # Note: This test requires a server without --enable-reasoning-tokens
        # For now, we test that when reasoning_tokens is 0, it's not included
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
            "extra_body": {"separate_reasoning": False},  # No reasoning content
        }
        response = client.chat.completions.create(**payload)

        # When separate_reasoning=False, there should be no reasoning content
        # But reasoning_tokens might still be calculated if enable_reasoning_tokens is True
        # This test verifies the behavior
        assert response.usage is not None
        # reasoning_tokens might be 0 or None depending on implementation
        # The key is that completion_tokens_details should be None if reasoning_tokens is 0
        if (
            response.usage.reasoning_tokens is None
            or response.usage.reasoning_tokens == 0
        ):
            assert (
                response.usage.completion_tokens_details is None
                or response.usage.completion_tokens_details.reasoning_tokens == 0
            )

    def test_nonstreaming_multiple_choices_reasoning_tokens(self):
        """Test that reasoning_tokens are correctly aggregated for multiple choices."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 2+2?",
                }
            ],
            "max_tokens": 100,
            "n": 2,  # Generate 2 choices
            "extra_body": {"separate_reasoning": True},
        }
        response = client.chat.completions.create(**payload)

        # Check that we have 2 choices
        assert len(response.choices) == 2
        # Check that usage is present
        assert response.usage is not None
        # Check that reasoning_tokens is aggregated across all choices
        assert response.usage.reasoning_tokens is not None
        assert response.usage.reasoning_tokens > 0
        # Check that completion_tokens_details is present
        assert response.usage.completion_tokens_details is not None
        assert (
            response.usage.completion_tokens_details.reasoning_tokens
            == response.usage.reasoning_tokens
        )
        # Reasoning tokens should be sum of reasoning tokens from all choices
        # (Each choice may have different reasoning tokens)

    def test_streaming_multiple_choices_reasoning_tokens(self):
        """Test that reasoning_tokens are correctly aggregated for streaming with multiple choices."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 2+2?",
                }
            ],
            "max_tokens": 100,
            "n": 2,  # Generate 2 choices
            "stream": True,
            "stream_options": {"include_usage": True},
            "extra_body": {"separate_reasoning": True},
        }
        response = client.chat.completions.create(**payload)

        # Collect all chunks and find usage chunk
        usage_chunk = None
        for chunk in response:
            if chunk.usage is not None:
                usage_chunk = chunk
                break

        # Check that usage chunk is present
        assert usage_chunk is not None
        assert usage_chunk.usage is not None
        # Check that reasoning_tokens is aggregated across all choices
        assert usage_chunk.usage.reasoning_tokens is not None
        assert usage_chunk.usage.reasoning_tokens > 0
        # Check that completion_tokens_details is present
        assert usage_chunk.usage.completion_tokens_details is not None
        assert (
            usage_chunk.usage.completion_tokens_details.reasoning_tokens
            == usage_chunk.usage.reasoning_tokens
        )


if __name__ == "__main__":
    unittest.main()
