"""Schema-driven argument conversion for the Spark2_5 tool parser."""

import json
import unittest

from jsonschema import Draft202012Validator
from pydantic import BaseModel

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.spark25_detector import Spark25Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSpark25SchemaTypes(CustomTestCase):
    def _parse(self, schema, values, chunk_size):
        tools = [
            Tool(
                type="function",
                function=Function(name="set_state", parameters=schema),
            )
        ]
        arguments = "".join(
            f"<arg_key>{key}</arg_key><arg_value>{value}</arg_value>"
            for key, value in values.items()
        )
        text = f"<tool_call>set_state{arguments}</tool_call>"
        detector = Spark25Detector()
        if chunk_size is None:
            calls = detector.detect_and_parse(text, tools).calls
        else:
            calls = []
            for start in range(0, len(text), chunk_size):
                result = detector.parse_streaming_increment(
                    text[start : start + chunk_size], tools
                )
                self.assertEqual(result.normal_text, "")
                calls.extend(result.calls)
            calls.extend(detector.finish(tools).calls)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "set_state")
        parsed = json.loads(calls[0].parameters)
        Draft202012Validator(schema).validate(parsed)
        return parsed

    def test_nullable_parameters(self):
        class Arguments(BaseModel):
            count: int | None
            ratio: float | None
            active: bool | None
            items: list[int] | None
            metadata: dict[str, int] | None
            name: str | None

        expected = {
            "count": 42,
            "ratio": 2.5,
            "active": False,
            "items": [1, 2],
            "metadata": {"value": 3},
            "name": "42",
        }
        schemas = [Arguments.model_json_schema()]
        schemas.append(
            {
                "type": "object",
                "properties": {
                    key: {"type": ["null", value["anyOf"][0]["type"]]}
                    for key, value in schemas[0]["properties"].items()
                },
            }
        )
        for schema in schemas:
            for use_null in (False, True):
                values = {
                    key: "null" if use_null else json.dumps(value)
                    for key, value in expected.items()
                }
                if not use_null:
                    values["name"] = "42"
                wanted = dict.fromkeys(expected) if use_null else expected
                for chunk_size in (None, 1, 7):
                    with self.subTest(
                        schema=schema, use_null=use_null, chunk_size=chunk_size
                    ):
                        self.assertEqual(
                            self._parse(schema, values, chunk_size), wanted
                        )

    def test_mixed_string_union_preserves_text(self):
        definitions = [
            {"anyOf": [{"type": "integer"}, {"type": "string"}]},
            {"oneOf": [{"type": "integer"}, {"type": "string"}]},
            {"type": ["integer", "string"]},
            {"type": ["string", "integer"]},
            {"type": ["null", "integer", "string"]},
            {"type": ["integer", "null", "string"]},
        ]
        for definition in definitions:
            schema = {"type": "object", "properties": {"value": definition}}
            for value in ("42", "2.5", "true", "[1,2]", '{"nested":1}'):
                for chunk_size in (None, 1, 7):
                    with self.subTest(
                        definition=definition, value=value, chunk_size=chunk_size
                    ):
                        self.assertEqual(
                            self._parse(schema, {"value": value}, chunk_size),
                            {"value": value},
                        )


if __name__ == "__main__":
    unittest.main()
