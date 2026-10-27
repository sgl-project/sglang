"""NPU end-to-end tests for the `--constrained-json-max-whitespace-cnt` feature.

These tests launch a real SGLang server on Ascend NPU with the xgrammar grammar
backend and verify that the constrained JSON output only contains at most
`MAX_WHITESPACE_CNT` consecutive whitespace characters, while remaining valid
JSON with the expected field types.
"""

import json
import unittest

import openai
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH as MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Max consecutive whitespace characters allowed in constrained JSON output.
MAX_WHITESPACE_CNT = 1

# Whitespace characters that may be emitted between JSON tokens.
WHITESPACE_CHARS = {" ", "\t", "\n", "\r"}

JSON_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "pattern": "^[\\w]+$"},
            "population": {"type": "integer"},
        },
        "required": ["name", "population"],
        "additionalProperties": False,
    }
)

register_npu_ci(est_time=360, suite="nightly-npu-a3-merged", nightly=True)


class TestNpuConstrainedJsonWhitespace(CustomTestCase):
    """Testcase: Verify that --constrained-json-max-whitespace-cnt limits the number
    of consecutive whitespace characters in xgrammar constrained JSON output on NPU.

    [Test Category] Interface
    [Test Target] /generate; /v1/chat/completions
    """

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--grammar-backend",
            "xgrammar",
            "--constrained-json-max-whitespace-cnt",
            str(MAX_WHITESPACE_CNT),
            "--attention-backend",
            "ascend",
            "--mem-fraction-static",
            "0.8",
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

    @staticmethod
    def _max_consecutive_whitespace(text):
        """Return the length of the longest whitespace run in text."""
        longest = 0
        current = 0
        for ch in text:
            if ch in WHITESPACE_CHARS:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return longest

    @staticmethod
    def _assert_valid_json(text):
        js_obj = json.loads(text)
        assert isinstance(js_obj["name"], str), "name must be a string"
        assert isinstance(js_obj["population"], int), "population must be an int"
        return js_obj

    def run_generate_json(self):
        # Use the raw /generate endpoint with json_schema in sampling_params.
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": (
                    "Introduce the capital of France. Return in a JSON format. "
                    "The JSON Schema is: " + JSON_SCHEMA
                ),
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 128,
                    "json_schema": JSON_SCHEMA,
                },
                "stream": False,
            },
        )
        ret = response.json()
        print(json.dumps(ret))
        print("=" * 100)
        text = ret["text"]

        assert self._max_consecutive_whitespace(text) <= MAX_WHITESPACE_CNT, (
            f"constrained JSON output has >{MAX_WHITESPACE_CNT} consecutive whitespace: {text!r}"
        )
        self._assert_valid_json(text)

    def run_chat_completions_json(self):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {
                    "role": "user",
                    "content": (
                        "Introduce the capital of France. Return in a JSON format. "
                        "The JSON Schema is: " + JSON_SCHEMA
                    ),
                },
            ],
            temperature=0,
            max_tokens=128,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "capital", "schema": json.loads(JSON_SCHEMA)},
            },
        )
        text = response.choices[0].message.content
        print(json.dumps(text))
        print("=" * 100)

        assert self._max_consecutive_whitespace(text) <= MAX_WHITESPACE_CNT, (
            f"constrained JSON output has >{MAX_WHITESPACE_CNT} consecutive whitespace: {text!r}"
        )
        self._assert_valid_json(text)

    def test_json_generate_with_whitespace_limit(self):
        self.run_generate_json()

    def test_json_chat_completions_with_whitespace_limit(self):
        self.run_chat_completions_json()


if __name__ == "__main__":
    unittest.main()
