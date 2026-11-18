import json
import os
import re
import unittest

import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestJSONModeMixin:
    """Mixin class containing JSON mode test methods"""

    def test_json_mode_response(self):
        """Test that response_format json_object (also known as "JSON mode") produces valid JSON even when JSON is not mentioned in the system prompt."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": "Output a user information JSON."},
            ],
            temperature=0,
            max_tokens=128,
            response_format={
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name", "age"],
                },
            },
        )
        text = response.choices[0].message.content

        print(f"Response ({len(text)} characters): {text}")

        # Verify the response is valid JSON
        try:
            js_obj = json.loads(text)
        except json.JSONDecodeError as e:
            self.fail(
                f"Response is not valid JSON. Error: {e}. Response content: {text}"
            )

        # Verify it is a JSON object (dict)
        self.assertIsInstance(js_obj, dict, f"Response is not a JSON object: {text}")
        self._verify_whitespace_pattern_constraint(text)

    def test_json_mode_with_streaming(self):
        """Test that streaming JSON mode (json_object) works correctly even when JSON is not mentioned in the system prompt."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": "Output a user information JSON."},
            ],
            temperature=0,
            max_tokens=1024,
            response_format={
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name", "age"],
                },
            },
            stream=True,
        )

        # Collect all chunks
        chunks = []
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                chunks.append(chunk.choices[0].delta.content)
        full_response = "".join(chunks)

        print(
            f"Concatenated response ({len(full_response)} characters): {full_response}"
        )

        # Verify the concatenated result is valid JSON
        try:
            js_obj = json.loads(full_response)
        except json.JSONDecodeError as e:
            self.fail(
                f"Streamed response is not valid JSON. Error: {e}. Response content: {full_response}"
            )

        self.assertIsInstance(js_obj, dict)
        self._verify_whitespace_pattern_constraint(full_response)

    def _verify_whitespace_pattern_constraint(self, json_str):
        """
        Verify that the --constrained-json-whitespace-pattern parameter only takes effect (JSON output contains newline whitespace)
        when the grammar backend is outlines/llguidance (pattern: [\n]?); for other backends, the parameter has no effect (no whitespace).
        """
        # Detect newline whitespace (\n) in JSON string (matching pattern [\n]?)
        has_newline_whitespace = bool(re.search(r"\n", json_str))

        # Expect newline whitespace (parameter takes effect)
        self.assertTrue(
            has_newline_whitespace,
            f"[{self.backend}] Missing expected newline whitespace after enabling --constrained-json-whitespace-pattern! JSON: {json_str}",
        )


class ServerWithGrammarBackend(CustomTestCase):
    """Testcase: Verify that when the grammar backend is outlines/llguidance, --constrained-json-whitespace-pattern=[\n]? takes effect (JSON output contains newline whitespace)

    [Test Category] Parameter
    [Test Target] --constrained-json-whitespace-pattern
    """

    backend = "outlines"

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        os.environ["TORCH_COMPILE_DISABLE"] = "1"

        # Server startup arguments: use constrained-json-whitespace-pattern with value [\n]?
        other_args = [
            "--max-running-requests",
            "10",
            "--grammar-backend",
            cls.backend,
            "--constrained-json-whitespace-pattern",
            "[\\n]?",
            "--attention-backend",
            "ascend",
        ]

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


class TestJSONModeLOutlines(ServerWithGrammarBackend, TestJSONModeMixin):
    backend = "outlines"


class TestJSONModeLLGuidance(ServerWithGrammarBackend, TestJSONModeMixin):
    backend = "llguidance"


if __name__ == "__main__":
    unittest.main()
