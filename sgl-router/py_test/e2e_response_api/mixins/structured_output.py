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

from basic_crud import ResponseAPIBaseTest


class StructuredOutputBaseTest(ResponseAPIBaseTest):

    def test_structured_output_json_schema(self):
        """Test structured output with json_schema format."""

        # Create response with structured output
        params = {
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

        create_resp = self.create_response(**params)
        self.assertIsNone(create_resp.error)
        self.assertIsNotNone(create_resp.id)
        self.assertIsNotNone(create_resp.output)
        self.assertIsNotNone(create_resp.text)

        # Verify text format was echoed back correctly
        self.assertIsNotNone(create_resp.text.format)
        self.assertEqual(create_resp.text.format.type, "json_schema")
        self.assertEqual(create_resp.text.format.name, "math_reasoning")
        self.assertIsNotNone(create_resp.text.format.schema_)
        self.assertEqual(create_resp.text.format.strict, True)

        # Find the message output (output[0] may be reasoning, output[1] is message)
        output_text = next(
            (
                content.text
                for item in create_resp.output
                if item.type == "message"
                for content in item.content
                if content.type == "output_text"
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
