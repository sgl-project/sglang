import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.minimax_m3 import MINIMAX_NS_TOKEN
from sglang.srt.function_call.utils import get_json_schema_properties
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestRootCombinatorToolParsers(unittest.TestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="acme",
                    parameters={
                        "type": "object",
                        "oneOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "kind": {"const": "acme"},
                                    "payload": {
                                        "type": "object",
                                        "properties": {"value": {"type": "string"}},
                                        "required": ["value"],
                                    },
                                },
                                "required": ["kind", "payload"],
                            },
                            {
                                "type": "object",
                                "properties": {"kind": {"const": "other"}},
                                "required": ["kind"],
                            },
                        ],
                    },
                ),
            )
        ]
        self.expected = {"kind": "acme", "payload": {"value": "hello"}}

    def test_property_lookup_supports_root_combinators(self):
        payload_schema = {
            "type": "object",
            "properties": {"value": {"type": "string"}},
        }
        for keyword in ("anyOf", "oneOf", "allOf"):
            with self.subTest(keyword=keyword):
                schema = {
                    keyword: [
                        {
                            "type": "object",
                            "properties": {"payload": payload_schema},
                        }
                    ]
                }
                self.assertEqual(
                    get_json_schema_properties(schema), {"payload": payload_schema}
                )

        self.assertEqual(
            get_json_schema_properties(self.tools[0].function.parameters)["kind"],
            {"oneOf": [{"const": "acme"}, {"const": "other"}]},
        )

        parser = FunctionCallParser(self.tools, "qwen3_coder")
        self.assertNotIn("properties", self.tools[0].function.parameters)
        self.assertIn(
            "payload", parser.detector_tools[0].function.parameters["properties"]
        )

    def test_parsers(self):
        ns = MINIMAX_NS_TOKEN
        cases = [
            (
                "qwen3_coder",
                [
                    "<tool_call>",
                    "<function=acme>",
                    "<parameter=kind>acme</parameter>",
                    '<parameter=payload>{"value":"hello"}</parameter>',
                    "</function>",
                    "</tool_call>",
                ],
            ),
            (
                "glm",
                [
                    "<tool_call>acme\n",
                    "<arg_key>kind</arg_key>\n<arg_value>acme</arg_value>\n",
                    (
                        '<arg_key>payload</arg_key>\n<arg_value>{"value":"hello"}'
                        "</arg_value>\n"
                    ),
                    "</tool_call>",
                ],
            ),
            (
                "glm47",
                [
                    "<tool_call>acme",
                    "<arg_key>kind</arg_key><arg_value>acme</arg_value>",
                    (
                        '<arg_key>payload</arg_key><arg_value>{"value":"hello"}'
                        "</arg_value>"
                    ),
                    "</tool_call>",
                ],
            ),
            (
                "dots",
                [
                    '<dots_function_call><invoke name="acme">',
                    '<parameter name="kind">acme</parameter>',
                    '<parameter name="payload">{"value":"hello"}</parameter>',
                    "</invoke></dots_function_call>",
                ],
            ),
            (
                "hunyuan",
                [
                    "<tool_calls><tool_call>acme<tool_sep>",
                    "<arg_key>kind</arg_key><arg_value>acme</arg_value>",
                    (
                        '<arg_key>payload</arg_key><arg_value>{"value":"hello"}'
                        "</arg_value>"
                    ),
                    "</tool_call></tool_calls>",
                ],
            ),
            (
                "mimo",
                [
                    "<tool_call><function=acme>",
                    "<parameter=kind>acme</parameter>",
                    '<parameter=payload>{"value":"hello"}</parameter>',
                    "</function></tool_call>",
                ],
            ),
            (
                "minicpm5",
                [
                    '<function name="acme">',
                    '<param name="kind">acme</param>',
                    '<param name="payload">{"value":"hello"}</param>',
                    "</function>",
                ],
            ),
            (
                "minimax-m2",
                [
                    '<minimax:tool_call><invoke name="acme">',
                    '<parameter name="kind">acme</parameter>',
                    '<parameter name="payload">{"value":"hello"}</parameter>',
                    "</invoke></minimax:tool_call>",
                ],
            ),
            (
                "minimax-m3",
                [
                    ns + segment
                    for segment in (
                        "<tool_call>",
                        '<invoke name="acme">',
                        "<kind>acme",
                        "</kind>",
                        "<payload>",
                        "<value>hello",
                        "</value>",
                        "</payload>",
                        "</invoke>",
                        "</tool_call>",
                    )
                ],
            ),
            (
                "poolside_v1",
                [
                    "<tool_call>acme\n",
                    "<arg_key>kind</arg_key>\n<arg_value>acme</arg_value>\n",
                    (
                        '<arg_key>payload</arg_key>\n<arg_value>{"value":"hello"}'
                        "</arg_value>\n"
                    ),
                    "</tool_call>",
                ],
            ),
            (
                "step3",
                [
                    "<｜tool_calls_begin｜><｜tool_call_begin｜>function<｜tool_sep｜>",
                    '<steptml:invoke name="acme">',
                    '<steptml:parameter name="kind">acme</steptml:parameter>',
                    (
                        '<steptml:parameter name="payload">{"value":"hello"}'
                        "</steptml:parameter>"
                    ),
                    "</steptml:invoke><｜tool_call_end｜><｜tool_calls_end｜>",
                ],
            ),
        ]

        for parser_name, chunks in cases:
            with self.subTest(parser=parser_name):
                parser = FunctionCallParser(self.tools, parser_name)
                _, calls = parser.parse_non_stream("".join(chunks))
                self.assertEqual(json.loads(calls[0].parameters), self.expected)

                parser = FunctionCallParser(self.tools, parser_name)
                parameters = ""
                for chunk in chunks:
                    _, calls = parser.parse_stream_chunk(chunk)
                    parameters += "".join(call.parameters for call in calls)
                self.assertEqual(json.loads(parameters), self.expected)

    def test_other_schema_consumers(self):
        other_tool = Tool(
            type="function",
            function=Function(
                name="search",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            ),
        )
        parser = FunctionCallParser([self.tools[0], other_tool], "kimi_k2")
        _, calls = parser.parse_non_stream(
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>0"
            f"<|tool_call_argument_begin|>{json.dumps(self.expected)}"
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        self.assertEqual(calls[0].name, "acme")


if __name__ == "__main__":
    unittest.main()
