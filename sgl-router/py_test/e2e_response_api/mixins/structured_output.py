"""
Structured output tests for Response API.

Tests for text.format field with json_object and json_schema formats.
"""

import json
import sys
from pathlib import Path

# Add current directory for local imports
_TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(_TEST_DIR))

from util import CustomTestCase


class StructuredOutputBaseTest(CustomTestCase):
    """Base class for structured output tests with common utilities."""

    # To be set by subclasses
    base_url: str = None
    api_key: str = None
    model: str = None

    def make_request(self, endpoint, method="GET", data=None):
        """Make HTTP request to the API."""
        url = f"{self.base_url}{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        if method == "GET":
            response = self.session.get(url, headers=headers)
        elif method == "POST":
            response = self.session.post(url, headers=headers, json=data)
        elif method == "DELETE":
            response = self.session.delete(url, headers=headers)
        else:
            raise ValueError(f"Unsupported method: {method}")

        return response

    def test_structured_output_json_schema(self):
        """Test structured output with json_schema format."""

        # Create response with structured output
        data = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": "You are a helpful math tutor. Guide the user through the solution step by step.",
                },
                {"role": "user", "content": "how can I solve 8x + 7 = -23"},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "math_reasoning",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "steps": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "explanation": {"type": "string"},
                                        "output": {"type": "string"},
                                    },
                                    "required": ["explanation", "output"],
                                    "additionalProperties": False,
                                },
                            },
                            "final_answer": {"type": "string"},
                        },
                        "required": ["steps", "final_answer"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            },
        }

        create_resp = self.make_request("/v1/responses", "POST", data)
        self.assertEqual(create_resp.status_code, 200)

        create_data = create_resp.json()
        self.assertIn("id", create_data)
        self.assertIn("output", create_data)
        self.assertIn("text", create_data)

        # Verify text format was echoed back correctly
        self.assertIn("format", create_data["text"])
        self.assertEqual(create_data["text"]["format"]["type"], "json_schema")
        self.assertEqual(create_data["text"]["format"]["name"], "math_reasoning")
        self.assertIn("schema", create_data["text"]["format"])
        self.assertEqual(create_data["text"]["format"]["strict"], True)

        # Find the message output (output[0] may be reasoning, output[1] is message)
        output_text = next(
            (
                content.get("text", "")
                for item in create_data.get("output", [])
                if item.get("type") == "message"
                for content in item.get("content", [])
                if content.get("type") == "output_text"
            ),
            None,
        )

        self.assertIsNotNone(output_text, "No output_text found in response")
        self.assertTrue(output_text.strip(), "output_text is empty")

        # Parse JSON output
        output_json = json.loads(output_text)

        # Verify schema structure
        self.assertIn("steps", output_json)
        self.assertIn("final_answer", output_json)
        self.assertIsInstance(output_json["steps"], list)
        self.assertGreater(len(output_json["steps"]), 0)

        # Verify each step has required fields
        for step in output_json["steps"]:
            self.assertIn("explanation", step)
            self.assertIn("output", step)
