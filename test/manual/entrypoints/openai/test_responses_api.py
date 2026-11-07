"""
Manual integration tests for the OpenAI Responses API on a gpt-oss server.

Covers the streaming token-usage regression where the final
``response.completed`` event reported zero usage for harmony (gpt-oss)
requests: usage must reflect the tokens actually streamed.

Run with:
    python -m pytest \
        test/manual/entrypoints/openai/test_responses_api.py -v
"""

import json
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

GPT_OSS_MODEL = "openai/gpt-oss-20b"


class TestResponsesApi(CustomTestCase):
    """Integration tests for /v1/responses against a gpt-oss server."""

    model = GPT_OSS_MODEL
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )
        cls.responses_url = f"{cls.base_url}/v1/responses"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _stream_events(self, payload):
        events = []
        with requests.post(self.responses_url, json=payload, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8")
                if not decoded.startswith("data: "):
                    continue
                data_str = decoded[len("data: ") :]
                if data_str.strip() == "[DONE]":
                    break
                events.append(json.loads(data_str))
        return events

    def test_streaming_responses_reports_token_usage(self):
        events = self._stream_events(
            {
                "model": self.model,
                "input": "What is 1 + 1?",
                "stream": True,
                "temperature": 0,
            }
        )
        completed = [e for e in events if e.get("type") == "response.completed"]
        self.assertEqual(len(completed), 1)
        usage = completed[0]["response"]["usage"]
        self.assertGreater(usage["input_tokens"], 0)
        self.assertGreater(usage["output_tokens"], 0)
        self.assertGreaterEqual(usage["total_tokens"], usage["input_tokens"])
        self.assertGreaterEqual(usage["total_tokens"], usage["output_tokens"])

    def test_non_streaming_responses_reports_token_usage(self):
        response = requests.post(
            self.responses_url,
            json={
                "model": self.model,
                "input": "What is 1 + 1?",
                "temperature": 0,
            },
        )
        response.raise_for_status()
        usage = response.json()["usage"]
        self.assertGreater(usage["input_tokens"], 0)
        self.assertGreater(usage["output_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
