"""
Usage:
python3 -m unittest test_reasoning_content.TestReasoningContentAPI.test_streaming_separate_reasoning_false
python3 -m unittest test_reasoning_content.TestReasoningContentAPI.test_streaming_separate_reasoning_true
python3 -m unittest test_reasoning_content.TestReasoningContentAPI.test_nonstreaming_separate_reasoning_false
python3 -m unittest test_reasoning_content.TestReasoningContentAPI.test_nonstreaming_separate_reasoning_true
python3 -m unittest test_reasoning_content.TestReasoningContentStartup.test_nonstreaming
python3 -m unittest test_reasoning_content.TestReasoningContentStartup.test_streaming
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
    popen_launch_server,
)


class TestReasoningContentAPI(unittest.TestCase):
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
            "separate_reasoning": False,
        }
        response = client.chat_completions.create(**payload)

        assert response.status_code == 200
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
            "separate_reasoning": True,
        }
        response = client.chat_completions.create(**payload)

        assert response.status_code == 200
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
            "separate_reasoning": True,
            "stream_reasoning": False,
        }
        response = client.chat_completions.create(**payload)

        assert response.status_code == 200
        reasoning_content = ""
        content = ""
        first_chunk = False
        for chunk in response:
            if chunk.choices[0].delta.content:
                first_chunk = True
                content += chunk.choices[0].delta.content
                reasoning_content = chunk.choices[0].delta.reasoning_content
            if not first_chunk:
                assert len(chunk.choices[0].delta.reasoning_content) == 0
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
            "separate_reasoning": False,
        }
        response = client.chat_completions.create(**payload)

        assert response.status_code == 200
        assert len(response.choices[0].message.reasoning_content) == 0
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
            "separate_reasoning": True,
        }
        response = client.chat_completions.create(**payload)

        assert response.status_code == 200
        assert len(response.choices[0].message.reasoning_content) > 0
        assert len(response.choices[0].message.content) > 0


if __name__ == "__main__":
    unittest.main()
