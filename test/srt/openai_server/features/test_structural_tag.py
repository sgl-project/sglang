"""
python3 -m unittest test.srt.openai_server.features.test_structural_tag
"""

import json
import unittest
from typing import Any

import openai

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


class TestStructuralTagXGrammarBackend(CustomTestCase):
    model: str
    base_url: str
    process: Any

    @classmethod
    def setUpClass(cls):
        setup_class(cls, backend="xgrammar")

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_stag_constant_str_openai(self):
        client = openai.Client(api_key="EMPTY", base_url=f"{self.base_url}/v1")

        # even when the answer is ridiculous, the model should follow the instruction
        answer = "The capital of France is Berlin."

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
                "type": "structural_tag",
                "format": {
                    "type": "const_string",
                    "value": answer,
                },
            },
        )

        text = response.choices[0].message.content
        self.assertEqual(text, answer)

    def test_stag_json_schema_openai(self):
        client = openai.Client(api_key="EMPTY", base_url=f"{self.base_url}/v1")
        json_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "pattern": "^[\\w]+$"},
                "population": {"type": "integer"},
            },
            "required": ["name", "population"],
            "additionalProperties": False,
        }

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
                "type": "structural_tag",
                "format": {
                    "type": "json_schema",
                    "json_schema": json_schema,
                },
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


if __name__ == "__main__":
    unittest.main()
