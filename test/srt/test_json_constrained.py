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
        cls.process = popen_launch_server(cls.model, cls.base_url, timeout=300)

    @classmethod
    def tearDownClass(cls):
        kill_child_process(cls.process.pid)

    def run_decode(
        self, prompt, json_schema, return_logprob=False, top_logprobs_num=0, n=1
    ):
        return requests.post(
            self.base_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0 if n == 1 else 0.5,
                    "max_new_tokens": 128,
                    "n": n,
                    "stop_token_ids": [119690],
                    "json_schema": json_schema,
                    "json_schema": json_schema,
                },
                "stream": False,
                "return_logprob": return_logprob,
                "top_logprobs_num": top_logprobs_num,
                "logprob_start_len": 0,
            },
        )

    def test_json_generate_simple(self):
        prompt = "The capital of France is"
        json_schema = json.dumps(
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "population": {"type": "integer"},
                },
                "required": ["name", "population"],
            }
        )
        response = self.run_decode(prompt, json_schema)
        print(json.dumps(response.json()))
        print("=" * 100)
        js_obj = json.loads(response.json()["text"])
        assert isinstance(js_obj["name"], str)
        assert isinstance(js_obj["population"], int)

    def test_json_generate_complex(self):
        prompt = "Please create a character named Komeiji Satori:"
        json_schema = """{
    "title": "Character",
    "type": "object",
    "properties": {
        "name": {
            "title": "Name",
            "type": "string"
        },
        "age": {
            "title": "Age",
            "type": "integer"
        },
        "armor": {"$ref": "#/$defs/Armor"},
        "weapon": {"$ref": "#/$defs/Weapon"},
        "strength": {
            "title": "Strength",
            "type": "integer"
        }
    },
    "required": ["name", "age", "armor", "weapon", "strength"],
    "$defs": {
        "Armor": {
            "title": "Armor",
            "description": "An enumeration.",
            "enum": ["leather", "chainmail", "plate"],
            "type": "string"
        },
        "Weapon": {
            "title": "Weapon",
            "description": "An enumeration.",
            "enum": ["third eye", "sword", "axe", "mace", "spear", "bow", "crossbow"],
            "type": "string"
        }
    }
}"""
        response = self.run_decode(prompt, json_schema)
        print(json.dumps(response.json()))
        print("=" * 100)
        js_obj = json.loads(response.json()["text"])
        assert isinstance(js_obj["name"], str)
        assert isinstance(js_obj["age"], int)
        assert js_obj["armor"] in ["leather", "chainmail", "plate"]
        assert js_obj["weapon"] in ["third eye", "sword", "axe", "mace", "spear", "bow", "crossbow"]
        assert isinstance(js_obj["strength"], int)

if __name__ == "__main__":
    unittest.main()
