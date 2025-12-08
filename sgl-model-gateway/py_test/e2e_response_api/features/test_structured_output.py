"""
Structured output tests for Response API.

Tests for text.format field with json_object and json_schema formats.
"""

import json
import sys
from pathlib import Path

import pytest

# Add current directory for local imports
_TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(_TEST_DIR))


@pytest.mark.parametrize("setup_backend", ["openai", "grpc_harmony"], indirect=True)
class TestStructuredOutput:

    def test_structured_output_json_schema(self, setup_backend):
        """Test structured output with json_schema format."""
        _, model, client = setup_backend

        # Create response with structured output
        params = {
            "model": model,
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

        create_resp = client.responses.create(**params)
        assert create_resp.error is None
        assert create_resp.id is not None
        assert create_resp.output is not None
        assert create_resp.text is not None

        # Verify text format was echoed back correctly
        assert create_resp.text.format is not None
        assert create_resp.text.format.type == "json_schema"
        assert create_resp.text.format.name == "math_reasoning"
        assert create_resp.text.format.schema_ is not None
        assert create_resp.text.format.strict

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

        assert output_text is not None, "No output_text found in response"
        assert output_text.strip(), "output_text is empty"

        # Parse JSON output
        output_json = json.loads(output_text)

        # Verify schema structure
        assert "steps" in output_json
        assert "final_answer" in output_json
        assert isinstance(output_json["steps"], list)
        assert len(output_json["steps"]) > 0

        # Verify each step has required fields
        for step in output_json["steps"]:
            assert "explanation" in step
            assert "output" in step


@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestSimpleSchemaStructuredOutput:

    def test_structured_output_json_schema(self, setup_backend):
        """Override with simpler schema for Llama model (complex schemas not well supported)."""
        _, model, client = setup_backend

        params = {
            "model": model,
            "input": [
                {
                    "role": "system",
                    "content": "You are a math solver. Return ONLY a JSON object that matches the schema—no extra text.",
                },
                {
                    "role": "user",
                    "content": "What is 1 + 1?",
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "math_answer",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                }
            },
        }

        create_resp = client.responses.create(**params)
        assert create_resp.error is None
        assert create_resp.id is not None
        assert create_resp.output is not None
        assert create_resp.text is not None

        # Verify text format was echoed back correctly
        assert create_resp.text.format is not None
        assert create_resp.text.format.type == "json_schema"
        assert create_resp.text.format.name == "math_answer"
        assert create_resp.text.format.schema_ is not None

        # Find the message output
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

        assert output_text is not None, "No output_text found in response"
        assert output_text.strip(), "output_text is empty"

        # Parse JSON output
        output_json = json.loads(output_text)

        # Verify simple schema structure (just answer field)
        assert "answer" in output_json
        assert isinstance(output_json["answer"], str)
        assert output_json["answer"], "Answer is empty"
