"""
Test speculative decoding (EAGLE v1 and v2) with constrained decoding.

Run:
    python3 test_spec_constrained_decoding.py
"""

import json
import os
import re
import unittest

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Use a small model for quick testing
MODEL = "meta-llama/Llama-2-7b-chat-hf"
DRAFT_MODEL = "lmzheng/sglang-EAGLE-llama2-chat-7B"


class TestSpecV2Constrained(CustomTestCase):
    """Test EAGLE v2 + constrained decoding"""

    spec_version = "v2"
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        # Set environment variable for v2 to enable overlap scheduling
        if cls.spec_version == "v2":
            os.environ["SGLANG_ENABLE_SPEC_V2"] = "1"
            os.environ["SGLANG_ENABLE_OVERLAP_PLAN_STREAM"] = "0"

        launch_args = [
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-draft-model-path",
            DRAFT_MODEL,
            "--speculative-num-steps",
            "5",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "6",
            "--grammar-backend",
            "xgrammar",
            "--mem-fraction-static",
            "0.5",
            "--max-running-requests",
            "1",
            "--context-length",
            2048,
            "--attention-backend",
            "triton",
        ]

        cls.process = popen_launch_server(
            MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=launch_args,
        )

    @classmethod
    def tearDownClass(cls):
        # kill_process_tree(cls.process.pid)
        # Clean up environment variable
        if "SGLANG_ENABLE_SPEC_V2" in os.environ:
            del os.environ["SGLANG_ENABLE_SPEC_V2"]

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

    def test_simple(self):
        with torch.cuda.nvtx.range("Generate simple"):
            result = self._generate("Generate some text: ")
        print("result simple", result)

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

        with torch.cuda.nvtx.range("Generate json schema constrained"):
            result = self._generate(
                "Generate a person's info. Respond in JSON format. The JSON schema is: "
                + json.dumps(json_schema),
                json_schema=json_schema,
            )

        print("result json schema constrained", result)
        # Validate output is valid JSON matching schema
        output_json = json.loads(result["text"])
        self.assertIn("name", output_json)
        self.assertIn("age", output_json)
        self.assertIsInstance(output_json["name"], str)
        self.assertIsInstance(output_json["age"], int)

    def test_regex_constrained(self):
        """Test regex constraint with spec decoding"""
        # Simple digit pattern
        regex = rf"^user@example\.com$"

        with torch.cuda.nvtx.range("Generate regex constrained"):
            result = self._generate("Generate an email address: ", regex=regex)

        print("result regex constrained", result)
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

        with torch.cuda.nvtx.range("Generate nested json schema"):
            result = self._generate(
                "Generate a list of items. The list should be in JSON format. The JSON schema is: "
                + json.dumps(json_schema),
                json_schema=json_schema,
            )

        # Validate output
        print("result nested json schema", result)
        output_json = json.loads(result["text"])
        self.assertIn("items", output_json)
        self.assertIsInstance(output_json["items"], list)
        if len(output_json["items"]) > 0:
            self.assertIn("id", output_json["items"][0])
            self.assertIn("name", output_json["items"][0])


# class TestSpecV2Constrained(TestSpecV1Constrained):
#     """Test EAGLE v2 (with overlap) + constrained decoding"""

#     spec_version = "v2"


if __name__ == "__main__":
    unittest.main()
