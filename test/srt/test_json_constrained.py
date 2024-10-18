"""
python3 -m unittest test_json_constrained.TestJSONConstrained.test_json_generate
"""

import json
import unittest
from concurrent.futures import ThreadPoolExecutor

import openai
import requests

from sglang.srt.utils import kill_child_process
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestJSONConstrained(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.json_schema = json.dumps(
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "pattern": "^[\\w]+$"},
                    "population": {"type": "integer"},
                },
                "required": ["name", "population"],
            }
        )
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=300,
            other_args=["--max-running-requests", "10"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_child_process(cls.process.pid)

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
        print(json.dumps(response.json()))
        print("=" * 100)

        if not json_schema:
            return

        try:
            js_obj = json.loads(response.json()["text"])
        except (TypeError, json.decoder.JSONDecodeError):
            raise
        assert isinstance(js_obj["name"], str)
        assert isinstance(js_obj["population"], int)

    def test_json_generate(self):
        self.run_decode(json_schema=self.json_schema)

    def test_json_openai(self):
        client = openai.Client(api_key="EMPTY", base_url=f"{self.base_url}/v1")

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {"role": "user", "content": "Introduce the capital of France."},
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
        assert isinstance(js_obj["name"], str)
        assert isinstance(js_obj["population"], int)

    def test_mix_json_and_other(self):
        json_schemas = [None, None, self.json_schema, self.json_schema] * 10

        with ThreadPoolExecutor(len(json_schemas)) as executor:
            list(executor.map(self.run_decode, json_schemas))


if __name__ == "__main__":
    unittest.main()
