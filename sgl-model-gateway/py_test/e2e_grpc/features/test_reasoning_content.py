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

import sys
import unittest
from pathlib import Path

import openai

_TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(_TEST_DIR.parent))
from fixtures import popen_launch_workers_and_router
from util import (
    DEFAULT_REASONING_MODEL_PATH,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    kill_process_tree,
)


class TestReasoningContentAPI(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # CHANGE: Launch gRPC router with integrated workers (single command)
        cls.model = DEFAULT_REASONING_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-1234"
        cls.cluster = popen_launch_workers_and_router(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            router_args=[
                "--reasoning-parser",
                "deepseek_r1",
            ],
            num_workers=1,
            tp_size=2,
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        # Cleanup router and workers
        kill_process_tree(cls.cluster["router"].pid)
        for worker in cls.cluster.get("workers", []):
            kill_process_tree(worker.pid)

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


if __name__ == "__main__":
    unittest.main()
