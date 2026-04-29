import json
import unittest

import openai
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_amd_ci,
    popen_launch_server,
)

register_cuda_ci(est_time=168, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=270, suite="stage-b-test-1-gpu-small-amd")


class JSONModeMixin:
    """Mixin class containing JSON mode test methods"""

    def test_json_mode_response(self):
        """Test that response_format json_object (also known as "json mode") produces valid JSON, even without a system prompt that mentions JSON."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                # We are deliberately omitting "That produces JSON" or similar phrases from the assistant prompt so that we don't have misleading test results
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that gives a short answer.",
                },
                {"role": "user", "content": "What is the capital of Bulgaria?"},
            ],
            temperature=0,
            max_tokens=128,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content

        print(f"Response ({len(text)} characters): {text}")

        # Verify the response is valid JSON
        try:
            js_obj = json.loads(text)
        except json.JSONDecodeError as e:
            self.fail(f"Response is not valid JSON. Error: {e}. Response: {text}")

        # Verify it's actually an object (dict)
        self.assertIsInstance(js_obj, dict, f"Response is not a JSON object: {text}")

    def test_json_mode_with_streaming(self):
        """Test that streaming with json_object response (also known as "json mode") format works correctly, even without a system prompt that mentions JSON."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                # We are deliberately omitting "That produces JSON" or similar phrases from the assistant prompt so that we don't have misleading test results
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that gives a short answer.",
                },
                {"role": "user", "content": "What is the capital of Bulgaria?"},
            ],
            temperature=0,
            max_tokens=128,
            response_format={"type": "json_object"},
            stream=True,
        )

        # Collect all chunks
        chunks = []
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                chunks.append(chunk.choices[0].delta.content)
        full_response = "".join(chunks)

        print(
            f"Concatenated Response ({len(full_response)} characters): {full_response}"
        )

        # Verify the combined response is valid JSON
        try:
            js_obj = json.loads(full_response)
        except json.JSONDecodeError as e:
            self.fail(
                f"Streamed response is not valid JSON. Error: {e}. Response: {full_response}"
            )

        self.assertIsInstance(js_obj, dict)


class ServerWithGrammarBackend(CustomTestCase):
    """Base class for tests requiring a grammar backend server"""

    backend = "xgrammar"

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST

        other_args = [
            "--max-running-requests",
            "10",
            "--grammar-backend",
            cls.backend,
        ]

        if is_in_amd_ci():
            other_args.append("--constrained-json-disable-any-whitespace")

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        cls.client = openai.Client(api_key="EMPTY", base_url=f"{cls.base_url}/v1")

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestJSONModeXGrammar(ServerWithGrammarBackend, JSONModeMixin):
    backend = "xgrammar"


class TestJSONModeOutlines(ServerWithGrammarBackend, JSONModeMixin):
    backend = "outlines"


class TestJSONModeLLGuidance(ServerWithGrammarBackend, JSONModeMixin):
    backend = "llguidance"


class TestReasoningJsonSchemaParallelSampling(CustomTestCase):
    """Regression test for n>1 + json_schema with a reasoning parser.

    Verifies that batched sub-requests preserve `require_reasoning`, so the
    structured JSON ends up in `message.content` instead of leaking into
    `message.reasoning_content`.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen3-0.6B"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--reasoning-parser",
                "qwen3",
            ],
        )
        cls.base_url += "/v1"
        cls.json_schema = {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "judge_result": {"type": "boolean"},
            },
            "required": ["reason", "judge_result"],
            "additionalProperties": False,
        }

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_json_schema_parallel_sampling_keeps_content(self):
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": "Bearer EMPTY"},
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Judge if this is positive: 'Great product!'",
                    }
                ],
                "temperature": 0.6,
                "max_tokens": 256,
                "n": 2,
                "chat_template_kwargs": {"enable_thinking": True},
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "judge_result",
                        "schema": self.json_schema,
                    },
                },
            },
        )

        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        data = response.json()

        self.assertEqual(len(data["choices"]), 2)
        for choice in data["choices"]:
            message = choice["message"]
            self.assertIsNotNone(message["content"])
            self.assertIsInstance(message["content"], str)
            payload = json.loads(message["content"])
            self.assertIsInstance(payload["reason"], str)
            self.assertIsInstance(payload["judge_result"], bool)


if __name__ == "__main__":
    unittest.main()
