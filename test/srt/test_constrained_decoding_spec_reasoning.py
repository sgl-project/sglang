import json
import unittest

import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class ServerWithGrammar(CustomTestCase):
    json_schema = json.dumps(
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "pattern": "^[\\w]+$"},
                "population": {"type": "integer"},
                "languages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "has_held_olympics": {"type": "boolean"},
            },
            "required": ["name", "population", "languages", "has_held_olympics"],
            "additionalProperties": False,
        }
    )

    @classmethod
    def setUpClass(cls):
        cls.model = "openai/gpt-oss-120b"
        cls.base_url = DEFAULT_URL_FOR_TEST
        launch_args = [
            "--trust-remote-code",
            "--tp=2",
            "--reasoning-parser=gpt-oss",
            "--speculative-algorithm=EAGLE3",
            "--speculative-draft-model-path=lmsys/EAGLE3-gpt-oss-120b-bf16",
            "--speculative-num-steps=5",
            "--speculative-eagle-topk=4",
            "--speculative-num-draft-tokens=8",
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=launch_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_json_openai(self):
        client = openai.Client(api_key="EMPTY", base_url=f"{self.base_url}/v1")

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {
                    "role": "user",
                    "content": "Introduce the capital of France. Return in a JSON format. "
                    "The JSON Schema is: " + json.dumps(self.json_schema),
                },
            ],
            temperature=0,
            max_tokens=1024,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "foo", "schema": json.loads(self.json_schema)},
            },
        )
        text = response.choices[0].message.content

        print("\n=== Reasoning Content ===")
        reasoning_content = response.choices[0].message.reasoning_content
        assert reasoning_content is not None and len(reasoning_content) > 0
        print(reasoning_content)

        try:
            js_obj = json.loads(text)
            print("\n=== Parsed JSON Content ===")
            print(json.dumps(js_obj))
        except (TypeError, json.decoder.JSONDecodeError):
            print("JSONDecodeError", text)
            raise

        self.assertIsInstance(js_obj["name"], str)
        self.assertIsInstance(js_obj["population"], int)


if __name__ == "__main__":
    unittest.main()
