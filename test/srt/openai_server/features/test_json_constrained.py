"""
python3 -m unittest openai_server.features.test_json_constrained.TestJSONConstrainedOutlinesBackend.test_json_generate
python3 -m unittest openai_server.features.test_json_constrained.TestJSONConstrainedXGrammarBackend.test_json_generate
python3 -m unittest openai_server.features.test_json_constrained.TestJSONConstrainedLLGuidanceBackend.test_json_generate
"""

import json
import unittest
from concurrent.futures import ThreadPoolExecutor

import openai
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


def setup_class(cls, backend: str):
    cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    cls.base_url = DEFAULT_URL_FOR_TEST
    cls.json_schema = json.dumps(
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

    other_args = [
        "--max-running-requests",
        "10",
        "--grammar-backend",
        backend,
    ]

    cls.process = popen_launch_server(
        cls.model,
        cls.base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
    )


class TestJSONConstrained(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        setup_class(cls, backend="xgrammar")

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def run_decode(self, json_schema, return_logprob=False, top_logprobs_num=0, n=1):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0 if n == 1 else 0.5,
                    "max_new_tokens": 128,
                    "n": n,
                    "stop_token_ids": [119690],
                    "json_schema": json_schema,
                },
                "stream": False,
                "return_logprob": return_logprob,
                "top_logprobs_num": top_logprobs_num,
                "logprob_start_len": 0,
            },
        )
        ret = response.json()
        print(json.dumps(ret))
        print("=" * 100)

        if not json_schema or json_schema == "INVALID":
            return

        # Make sure the json output is valid
        try:
            js_obj = json.loads(ret["text"])
        except (TypeError, json.decoder.JSONDecodeError):
            raise

        self.assertIsInstance(js_obj["name"], str)
        self.assertIsInstance(js_obj["population"], int)

    def test_json_generate(self):
        self.run_decode(json_schema=self.json_schema)

    def test_json_invalid(self):
        self.run_decode(json_schema="INVALID")

    def test_json_openai(self):
        client = openai.Client(api_key="EMPTY", base_url=f"{self.base_url}/v1")

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {
                    "role": "user",
                    "content": "Introduce the capital of France. Return in a JSON format.",
                },
            ],
            temperature=0,
            max_tokens=128,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "foo", "schema": json.loads(self.json_schema)},
            },
        )
        text = response.choices[0].message.content

        try:
            js_obj = json.loads(text)
        except (TypeError, json.decoder.JSONDecodeError):
            print("JSONDecodeError", text)
            raise

        self.assertIsInstance(js_obj["name"], str)
        self.assertIsInstance(js_obj["population"], int)

    def test_mix_json_and_other(self):
        json_schemas = [None, None, self.json_schema, self.json_schema] * 10

        with ThreadPoolExecutor(len(json_schemas)) as executor:
            list(executor.map(self.run_decode, json_schemas))


class TestJSONConstrainedOutlinesBackend(TestJSONConstrained):
    @classmethod
    def setUpClass(cls):
        setup_class(cls, backend="outlines")


class TestJSONConstrainedLLGuidanceBackend(TestJSONConstrained):
    @classmethod
    def setUpClass(cls):
        setup_class(cls, backend="llguidance")


if __name__ == "__main__":
    unittest.main()
