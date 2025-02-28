"""
python3 -m unittest test_reasoning_content.TestReasoningContentAPI.test_streaming_separate_reasoning_false
python3 -m unittest test_reasoning_content.TestReasoningContentAPI.test_streaming_separate_reasoning_true
python3 -m unittest test_reasoning_content.TestReasoningContentAPI.test_nonstreaming_separate_reasoning_false
python3 -m unittest test_reasoning_content.TestReasoningContentAPI.test_nonstreaming_separate_reasoning_true
python3 -m unittest test_reasoning_content.TestReasoningContentStartup.test_nonstreaming
python3 -m unittest test_reasoning_content.TestReasoningContentStartup.test_streaming
"""

import json
import unittest

import requests

from sglang.test.test_utils import (
    DEFAULT_REASONING_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    kill_process_tree,
    popen_launch_server,
)


class TestReasoningContentAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_REASONING_MODEL_NAME_FOR_TEST
        cls.base_url = "http://0.0.0.0:5000"  # DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )

    @classmethod
    def tearDownClass(cls):
        # kill_process_tree(cls.process.pid)
        pass

    def test_streaming_separate_reasoning_false(self):
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 1+1?",
                }
            ],
            "max_tokens": 100,
            "stream": True,
            "separate_reasoning": False,
        }
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            stream=True,
        )
        assert response.status_code == 200
        reasoning_content = ""
        content = ""
        for line in response.iter_lines():
            print(f"[test_streaming_separate_reasoning_false] {line}")
            if line and not line.startswith(b"data: [DONE]"):
                parsed = json.loads(line[6:])
                if (
                    "reasoning_content" in parsed["choices"][0]["delta"]
                    and parsed["choices"][0]["delta"]["reasoning_content"]
                ):
                    reasoning_content += parsed["choices"][0]["delta"][
                        "reasoning_content"
                    ]
                if (
                    "content" in parsed["choices"][0]["delta"]
                    and parsed["choices"][0]["delta"]["content"]
                ):
                    content += parsed["choices"][0]["delta"]["content"]

        assert len(reasoning_content) == 0
        assert len(content) > 0

    def test_streaming_separate_reasoning_true(self):
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 1+1?",
                }
            ],
            "max_tokens": 100,
            "stream": True,
            "separate_reasoning": True,
        }
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            stream=True,
        )
        assert response.status_code == 200
        reasoning_content = ""
        content = ""
        for line in response.iter_lines():
            print(f"[test_streaming_separate_reasoning_true] {line}")
            if line and not line.startswith(b"data: [DONE]"):
                parsed = json.loads(line[6:])
                if (
                    "reasoning_content" in parsed["choices"][0]["delta"]
                    and parsed["choices"][0]["delta"]["reasoning_content"]
                ):
                    reasoning_content += parsed["choices"][0]["delta"][
                        "reasoning_content"
                    ]
                if (
                    "content" in parsed["choices"][0]["delta"]
                    and parsed["choices"][0]["delta"]["content"]
                ):
                    content += parsed["choices"][0]["delta"]["content"]

        assert len(reasoning_content) > 0

    def test_nonstreaming_separate_reasoning_false(self):
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 1+1?",
                }
            ],
            "max_tokens": 100,
            "separate_reasoning": False,
        }
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
        )
        assert response.status_code == 200
        resp = response.json()
        assert resp["choices"][0]["message"]["reasoning_content"] == None
        assert len(resp["choices"][0]["message"]["content"]) > 0

    def test_nonstreaming_separate_reasoning_true(self):
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 1+1?",
                }
            ],
            "max_tokens": 100,
            "separate_reasoning": True,
        }
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
        )
        assert response.status_code == 200
        resp = response.json()
        assert len(resp["choices"][0]["message"]["reasoning_content"]) > 0


class TestReasoningContentStartup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_REASONING_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--reasoning-parser",
                "deepseek-r1",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_nonstreaming(self):
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 1+1?",
                }
            ],
            "max_tokens": 100,
        }
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
        )
        assert response.status_code == 200
        resp = response.json()
        assert len(resp["choices"][0]["message"]["reasoning_content"]) > 0

    def test_streaming(self):
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 1+1?",
                }
            ],
            "max_tokens": 100,
            "stream": True,
        }
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            stream=True,
        )
        assert response.status_code == 200
        reasoning_content = ""
        content = ""
        for line in response.iter_lines():
            print(f"[test_streaming_separate_reasoning_true] {line}")
            if line and not line.startswith(b"data: [DONE]"):
                parsed = json.loads(line[6:])
                if (
                    "reasoning_content" in parsed["choices"][0]["delta"]
                    and parsed["choices"][0]["delta"]["reasoning_content"]
                ):
                    reasoning_content += parsed["choices"][0]["delta"][
                        "reasoning_content"
                    ]
                if (
                    "content" in parsed["choices"][0]["delta"]
                    and parsed["choices"][0]["delta"]["content"]
                ):
                    content += parsed["choices"][0]["delta"]["content"]

        assert len(reasoning_content) > 0
