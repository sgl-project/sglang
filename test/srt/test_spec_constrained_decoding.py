"""
Test speculative decoding (EAGLE v1 and v2) with constrained decoding.

Run:
    python3 test_spec_constrained_decoding.py
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

# Use a small model for quick testing
MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DRAFT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


class SpecConstrainedBase(CustomTestCase):
    """Base class for spec + constrained decoding tests"""

    spec_version = "v1"  # Override in subclasses
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        launch_args = [
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-draft-model-path",
            DRAFT_MODEL,
            "--speculative-num-steps",
            "2",
            "--speculative-eagle-topk",
            "2",
            "--speculative-num-draft-tokens",
            "4",
            "--grammar-backend",
            "xgrammar",
            "--mem-fraction-static",
            "0.7",
            "--max-running-requests",
            "8",
        ]

        # Disable overlap for v1 to test traditional spec decoding
        if cls.spec_version == "v1":
            launch_args += ["--disable-overlap-schedule"]

        cls.process = popen_launch_server(
            MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=launch_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _generate(self, prompt, json_schema=None, regex=None):
        """Helper method to generate text with optional constraints"""
        sampling_params = {
            "temperature": 0.0,
            "max_new_tokens": 64,
        }

        if json_schema:
            sampling_params["json_schema"] = json.dumps(json_schema)
        if regex:
            sampling_params["regex"] = regex

        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": sampling_params,
            },
        )
        return response.json()

    def test_json_schema_constrained(self):
        """Test JSON schema constraint with spec decoding"""
        json_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }

        result = self._generate(
            "Generate a person's info:", json_schema=json_schema
        )

        # Validate output is valid JSON matching schema
        output_json = json.loads(result["text"])
        self.assertIn("name", output_json)
        self.assertIn("age", output_json)
        self.assertIsInstance(output_json["name"], str)
        self.assertIsInstance(output_json["age"], int)

    def test_regex_constrained(self):
        """Test regex constraint with spec decoding"""
        # Simple digit pattern
        regex = r"\d+"

        result = self._generate("The answer is ", regex=regex)

        # Validate output matches regex
        import re

        self.assertTrue(re.fullmatch(regex, result["text"].strip()))

    def test_nested_json_schema(self):
        """Test nested JSON schema with spec decoding"""
        json_schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "name": {"type": "string"},
                        },
                        "required": ["id", "name"],
                    },
                },
            },
            "required": ["items"],
        }

        result = self._generate("Generate a list:", json_schema=json_schema)

        # Validate output
        output_json = json.loads(result["text"])
        self.assertIn("items", output_json)
        self.assertIsInstance(output_json["items"], list)
        if len(output_json["items"]) > 0:
            self.assertIn("id", output_json["items"][0])
            self.assertIn("name", output_json["items"][0])


class TestSpecV1Constrained(SpecConstrainedBase):
    """Test EAGLE v1 (without overlap) + constrained decoding"""

    spec_version = "v1"


class TestSpecV2Constrained(SpecConstrainedBase):
    """Test EAGLE v2 (with overlap) + constrained decoding"""

    spec_version = "v2"


if __name__ == "__main__":
    unittest.main()
