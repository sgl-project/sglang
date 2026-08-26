import copy
import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.minimax_m3 import MINIMAX_NS_TOKEN
from sglang.srt.function_call.utils import get_tool_parser_property_hints
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


PROPERTY_SCHEMAS = {
    "number": {"type": "number"},
    "stringList": {
        "type": "array",
        "items": {"type": "string"},
    },
    "numberList": {
        "type": "array",
        "items": {"type": "number"},
    },
}
ARGUMENT_CASES = (
    ("number", "1.5", 1.5),
    ("stringList", '["alpha","beta"]', ["alpha", "beta"]),
    ("numberList", "[1,2.5]", [1, 2.5]),
)


def disjoint_schema(keyword: str = "oneOf") -> dict:
    return {
        "type": "object",
        keyword: [
            {
                "type": "object",
                "properties": {name: property_schema},
                "required": [name],
                "additionalProperties": False,
            }
            for name, property_schema in PROPERTY_SCHEMAS.items()
        ],
    }


class TestToolParserPropertyHints(unittest.TestCase):
    def test_disjoint_combinator_properties_are_projected(self):
        for keyword in ("allOf", "anyOf", "oneOf"):
            with self.subTest(keyword=keyword):
                schema = disjoint_schema(keyword)
                if keyword == "allOf":
                    for branch in schema[keyword]:
                        branch.pop("additionalProperties")

                self.assertEqual(
                    get_tool_parser_property_hints(schema),
                    PROPERTY_SCHEMAS,
                )

    def test_direct_property_is_authoritative(self):
        direct_schema = {"type": "object"}
        schema = {
            "properties": {"payload": direct_schema},
            "oneOf": [
                {"properties": {"payload": {"type": "string"}}},
                {"properties": {"payload": {"type": "array"}}},
            ],
        }

        self.assertEqual(
            get_tool_parser_property_hints(schema),
            {"payload": direct_schema},
        )

    def test_identical_branch_property_is_retained(self):
        property_schema = {"type": "object"}
        schema = {
            "anyOf": [
                {"properties": {"payload": property_schema}},
                {"properties": {"payload": copy.deepcopy(property_schema)}},
            ]
        }

        self.assertEqual(
            get_tool_parser_property_hints(schema),
            {"payload": property_schema},
        )

    def test_conflicting_branch_property_is_unconstrained(self):
        schema = {
            "oneOf": [
                {"properties": {"payload": {"type": "object"}}},
                {"properties": {"payload": {"type": "string"}}},
            ]
        }

        self.assertEqual(
            get_tool_parser_property_hints(schema),
            {"payload": {}},
        )

    def test_local_refs_are_resolved(self):
        schema = {
            "$defs": {
                "Arguments": {
                    "oneOf": [{"properties": {"payload": {"$ref": "#/$defs/Payload"}}}]
                },
                "Payload": {"type": "object"},
            },
            "$ref": "#/$defs/Arguments",
        }

        self.assertEqual(
            get_tool_parser_property_hints(schema),
            {"payload": {"type": "object"}},
        )

    def test_original_schema_is_not_modified(self):
        schema = disjoint_schema()
        original = copy.deepcopy(schema)

        get_tool_parser_property_hints(schema)

        self.assertEqual(schema, original)


class TestRootCombinatorToolParsers(unittest.TestCase):
    def setUp(self):
        parameters = disjoint_schema()
        self.original_parameters = copy.deepcopy(parameters)
        self.tools = [
            Tool(
                type="function",
                function=Function(name="convert", parameters=parameters),
            )
        ]

    def parser_cases(self, argument: str, raw_value: str):
        ns = MINIMAX_NS_TOKEN
        minimax_value = [f"<{argument}>{raw_value}", f"</{argument}>"]
        if argument.endswith("List"):
            minimax_value = [f"<{argument}>"]
            for item in json.loads(raw_value):
                minimax_value.extend((f"<item>{item}", "</item>"))
            minimax_value.append(f"</{argument}>")

        return [
            (
                "qwen3_coder",
                [
                    "<tool_call>",
                    "<function=convert>",
                    f"<parameter={argument}>{raw_value}</parameter>",
                    "</function>",
                    "</tool_call>",
                ],
            ),
            (
                "glm",
                [
                    "<tool_call>convert\n",
                    (
                        f"<arg_key>{argument}</arg_key>\n"
                        f"<arg_value>{raw_value}</arg_value>\n"
                    ),
                    "</tool_call>",
                ],
            ),
            (
                "glm47",
                [
                    "<tool_call>convert",
                    (
                        f"<arg_key>{argument}</arg_key>"
                        f"<arg_value>{raw_value}</arg_value>"
                    ),
                    "</tool_call>",
                ],
            ),
            (
                "dots",
                [
                    '<dots_function_call><invoke name="convert">',
                    f'<parameter name="{argument}">{raw_value}</parameter>',
                    "</invoke></dots_function_call>",
                ],
            ),
            (
                "hunyuan",
                [
                    "<tool_calls><tool_call>convert<tool_sep>",
                    (
                        f"<arg_key>{argument}</arg_key>"
                        f"<arg_value>{raw_value}</arg_value>"
                    ),
                    "</tool_call></tool_calls>",
                ],
            ),
            (
                "mimo",
                [
                    "<tool_call><function=convert>",
                    f"<parameter={argument}>{raw_value}</parameter>",
                    "</function></tool_call>",
                ],
            ),
            (
                "minicpm5",
                [
                    '<function name="convert">',
                    f'<param name="{argument}">{raw_value}</param>',
                    "</function>",
                ],
            ),
            (
                "minimax-m2",
                [
                    '<minimax:tool_call><invoke name="convert">',
                    f'<parameter name="{argument}">{raw_value}</parameter>',
                    "</invoke></minimax:tool_call>",
                ],
            ),
            (
                "minimax-m3",
                [
                    ns + segment
                    for segment in (
                        "<tool_call>",
                        '<invoke name="convert">',
                        *minimax_value,
                        "</invoke>",
                        "</tool_call>",
                    )
                ],
            ),
            (
                "poolside_v1",
                [
                    "<tool_call>convert\n",
                    (
                        f"<arg_key>{argument}</arg_key>\n"
                        f"<arg_value>{raw_value}</arg_value>\n"
                    ),
                    "</tool_call>",
                ],
            ),
            (
                "step3",
                [
                    "<｜tool_calls_begin｜><｜tool_call_begin｜>function<｜tool_sep｜>",
                    '<steptml:invoke name="convert">',
                    (
                        f'<steptml:parameter name="{argument}">{raw_value}'
                        "</steptml:parameter>"
                    ),
                    "</steptml:invoke><｜tool_call_end｜><｜tool_calls_end｜>",
                ],
            ),
        ]

    def assert_arguments(self, serialized: str, expected: dict):
        self.assertNotIn("<", serialized)
        self.assertEqual(json.loads(serialized), expected)

    def test_streaming_and_non_streaming_parsers(self):
        parser_names = [
            parser_name for parser_name, _ in self.parser_cases("number", "1.5")
        ]
        for parser_name in parser_names:
            non_streaming_parser = FunctionCallParser(self.tools, parser_name)

            for argument, raw_value, expected_value in ARGUMENT_CASES:
                chunks = dict(self.parser_cases(argument, raw_value))[parser_name]
                expected = {argument: expected_value}

                with self.subTest(
                    parser=parser_name,
                    argument=argument,
                    mode="non_streaming",
                ):
                    _, calls = non_streaming_parser.parse_non_stream("".join(chunks))
                    self.assertEqual(len(calls), 1)
                    self.assert_arguments(calls[0].parameters, expected)

                with self.subTest(
                    parser=parser_name,
                    argument=argument,
                    mode="streaming",
                ):
                    streaming_parser = FunctionCallParser(self.tools, parser_name)
                    parameters = ""
                    for chunk in chunks:
                        _, calls = streaming_parser.parse_stream_chunk(chunk)
                        parameters += "".join(call.parameters for call in calls)
                    _, calls = streaming_parser.parse_stream_end()
                    parameters += "".join(call.parameters for call in calls)
                    self.assert_arguments(parameters, expected)

            self.assertEqual(
                self.tools[0].function.parameters,
                self.original_parameters,
            )

    def test_detector_tools_use_projected_properties(self):
        parser = FunctionCallParser(self.tools, "qwen3_coder")

        self.assertNotIn("properties", self.tools[0].function.parameters)
        self.assertEqual(
            parser.detector_tools[0].function.parameters["properties"],
            PROPERTY_SCHEMAS,
        )

    def test_conflicting_property_uses_conservative_string_fallback(self):
        parameters = {
            "oneOf": [
                {"properties": {"payload": {"type": "object"}}},
                {"properties": {"payload": {"type": "string"}}},
            ]
        }
        tools = [
            Tool(
                type="function",
                function=Function(name="convert", parameters=parameters),
            )
        ]
        parser = FunctionCallParser(tools, "qwen3_coder")

        _, calls = parser.parse_non_stream(
            "<tool_call><function=convert>"
            '<parameter=payload>{"value":1}</parameter>'
            "</function></tool_call>"
        )

        self.assertEqual(
            json.loads(calls[0].parameters),
            {"payload": '{"value":1}'},
        )


if __name__ == "__main__":
    unittest.main()
