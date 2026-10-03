import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMinimaxM2NullableParameters(unittest.TestCase):
    def check_value(self, schema, raw, expected):
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="lookup",
                    parameters={"type": "object", "properties": {"value": schema}},
                ),
            )
        ]
        pieces = [
            "<minimax:tool_call>",
            '<invoke name="lookup">',
            f'<parameter name="value">{raw}</parameter>',
            "</invoke>",
            "</minimax:tool_call>",
        ]
        for chunks in (
            None,
            ["".join(pieces)],
            pieces,
            [pieces[0], *"".join(pieces[1:])],
        ):
            with self.subTest(schema=schema, raw=raw, streaming=chunks is not None):
                parser = FunctionCallParser(tools, "minimax-m2")
                if chunks is None:
                    normal, calls = parser.parse_non_stream("".join(pieces))
                else:
                    normal, calls = "", []
                    for chunk in chunks:
                        content, delta = parser.parse_stream_chunk(chunk)
                        normal += content
                        calls.extend(delta)
                self.assertEqual(normal, "")
                self.assertEqual([call.name for call in calls if call.name], ["lookup"])
                self.assertTrue(all(call.tool_index == 0 for call in calls))
                arguments = json.loads("".join(call.parameters for call in calls))
                self.assertEqual(arguments, {"value": expected})
                self.assertIs(type(arguments["value"]), type(expected))

    def test_nullable_type_arrays_preserve_non_null_values(self):
        for kind, raw, expected in (
            ("integer", "42", 42),
            ("number", "2.5", 2.5),
            ("string", "hello", "hello"),
            ("boolean", "false", False),
            ("array", "[1,2]", [1, 2]),
            ("object", '{"a":1}', {"a": 1}),
        ):
            self.check_value({"type": [kind, "null"]}, raw, expected)

    def test_nullable_compositions_preserve_non_null_values(self):
        for keyword in ("anyOf", "oneOf"):
            self.check_value(
                {keyword: [{"type": "integer"}, {"type": "null"}]}, "42", 42
            )

    def test_nullable_enum_preserves_non_null_values(self):
        self.check_value({"enum": ["low", "high", None]}, "high", "high")

    def test_explicit_null_and_legacy_aliases_are_unchanged(self):
        for raw in ("null", "none", "nil"):
            self.check_value({"type": ["integer", "null"]}, raw, None)

    def test_non_nullable_values_are_unchanged(self):
        self.check_value({"type": "integer"}, "42", 42)
        self.check_value({"type": "string"}, "hello", "hello")


if __name__ == "__main__":
    unittest.main()
