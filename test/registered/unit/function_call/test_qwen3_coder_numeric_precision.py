import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestQwen3CoderNumericPrecision(unittest.TestCase):
    def check_value(self, schema, raw, expected):
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="lookup",
                    parameters={
                        "type": "object",
                        "properties": {"value": schema},
                    },
                ),
            )
        ]
        text = (
            "<tool_call><function=lookup><parameter=value>"
            f"{raw}</parameter></function></tool_call>"
        )
        for parser_name in ("qwen3_coder", "step3p5", "nanbeige"):
            for chunk_size in (0, 1, 11, len(text)):
                with self.subTest(parser=parser_name, chunk_size=chunk_size, raw=raw):
                    parser = FunctionCallParser(tools, parser_name)
                    if chunk_size == 0:
                        normal, calls = parser.parse_non_stream(text)
                    else:
                        normal, calls = "", []
                        for offset in range(0, len(text), chunk_size):
                            content, delta = parser.parse_stream_chunk(
                                text[offset : offset + chunk_size]
                            )
                            normal += content
                            calls.extend(delta)
                    self.assertEqual(normal, "")
                    self.assertEqual(
                        [call.name for call in calls if call.name], ["lookup"]
                    )
                    self.assertTrue(all(call.tool_index == 0 for call in calls))
                    arguments = json.loads("".join(call.parameters for call in calls))
                    self.assertEqual(arguments, {"value": expected})
                    self.assertIs(type(arguments["value"]), type(expected))

    def test_number_schema_preserves_large_integer_literals(self):
        for value in (
            9007199254740993,
            -9007199254740993,
            18446744073709551615,
            1790892345000000001,
        ):
            self.check_value({"type": "number"}, str(value), value)
        self.check_value({"type": "number"}, " 9007199254740993 ", 9007199254740993)

    def test_nullable_number_schema_preserves_large_integer_literals(self):
        self.check_value(
            {"anyOf": [{"type": "number"}, {"type": "null"}]},
            "9007199254740993",
            9007199254740993,
        )

    def test_other_numeric_forms_are_unchanged(self):
        for raw, expected in (("42", 42), ("-12.5", -12.5), ("1.25e3", 1250.0)):
            self.check_value({"type": "number"}, raw, expected)
        self.check_value({"type": "number"}, "not-a-number", "not-a-number")
        self.check_value({"type": "number"}, "null", None)

    def test_integer_and_string_schemas_are_unchanged(self):
        self.check_value({"type": "integer"}, "9007199254740993", 9007199254740993)
        self.check_value({"type": "string"}, "9007199254740993", "9007199254740993")


if __name__ == "__main__":
    unittest.main()
