"""
Usage:
python3 -m unittest openai_server.features.test_usage_tokens.TestReasoningTokenUsage.test_nonstreaming_usage_reasoning_tokens_present
python3 -m unittest openai_server.features.test_usage_tokens.TestReasoningTokenUsage.test_streaming_usage_reasoning_tokens_present
"""

import unittest

import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_REASONING_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestReasoningTokenUsage(CustomTestCase):
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

    def test_nonstreaming_usage_reasoning_tokens_present(self):
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

        assert response.usage is not None
        assert response.usage.reasoning_tokens is not None
        assert response.usage.reasoning_tokens > 0

    def test_streaming_usage_reasoning_tokens_present(self):
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

        final_usage = None
        for chunk in response:
            if chunk.usage is not None:
                final_usage = chunk.usage

        assert final_usage is not None
        assert final_usage.reasoning_tokens is not None
        assert final_usage.reasoning_tokens > 0


if __name__ == "__main__":
    unittest.main()
