import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.glm47_moe_detector import Glm47MoeDetector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestGlm47SchemaTypes(unittest.TestCase):
    def check_arguments(self, schema, expected, raw_values=None):
        tools = [
            Tool(type="function", function=Function(name="inspect", parameters=schema))
        ]
        pairs = []
        for key, value in expected.items():
            raw = (raw_values or {}).get(
                key, value if isinstance(value, str) else json.dumps(value)
            )
            pairs.append(f"<arg_key>{key}</arg_key><arg_value>{raw}</arg_value>")
        text = "<tool_call>inspect" + "".join(pairs) + "</tool_call>"
        for size in (0, 1, 7, len(text)):
            with self.subTest(schema=schema, expected=expected, chunk_size=size):
                detector = Glm47MoeDetector()
                if size == 0:
                    calls = detector.detect_and_parse(text, tools).calls
                else:
                    calls = []
                    for offset in range(0, len(text), size):
                        calls.extend(
                            detector.parse_streaming_increment(
                                text[offset : offset + size], tools
                            ).calls
                        )
                actual = json.loads("".join(call.parameters for call in calls))
                self.assertEqual(actual, expected)
                for key in expected:
                    self.assertIs(type(actual[key]), type(expected[key]))

    def test_field_types(self):
        cases = [
            ({"type": ["integer", "string"]}, "auto"),
            ({"type": ["string", "integer"]}, 7),
            ({"type": ["number", "string"]}, "auto"),
            ({"type": ["object", "string"]}, "auto"),
            ({"type": ["string", "object"]}, {"ok": False}),
            ({"type": ["boolean", "string"]}, "auto"),
            ({"type": ["string", "boolean"]}, False),
            ({"type": ["string", "null"]}, None),
            ({"type": ["null"]}, None),
            ({"type": "null"}, None),
            ({"enum": ["ok", None]}, None),
            ({"type": ["string", "null"], "enum": ["ok", None]}, None),
            ({"anyOf": [{"type": "string", "enum": ["auto"]}, {"type": "integer"}]}, 7),
            (
                {
                    "oneOf": [
                        {"type": "null"},
                        {"type": "array", "items": {"type": "integer"}},
                    ]
                },
                [1, 2],
            ),
            ({"const": 7}, 7),
            ({"const": {"ok": False}}, {"ok": False}),
            ({"const": "null"}, "null"),
            ({"allOf": [{"type": ["integer", "string"]}, {"type": "integer"}]}, 7),
            ({}, "123_456"),
            ({}, {"ok": False}),
        ]
        cases.extend(
            ({"type": "string"}, value)
            for value in (
                "null",
                "true",
                "1e2",
                "123_456",
                "\\d+",
                'line one\n"quoted" \\ café',
            )
        )
        for field, value in cases:
            self.check_arguments(
                {"type": "object", "properties": {"value": field}}, {"value": value}
            )
        self.check_arguments(
            {"type": "object", "properties": {"value": {"type": ["string", "null"]}}},
            {"value": "null"},
            {"value": '"null"'},
        )

    def test_local_references_and_cycles(self):
        for value, field in [
            (7, {"type": "integer"}),
            ({"ok": False}, {"type": "object"}),
            ("null", {"type": "string"}),
        ]:
            for defs in ("$defs", "definitions"):
                schema = {
                    defs: {"a/b~c": field},
                    "type": "object",
                    "properties": {"value": {"$ref": f"#/{defs}/a~1b~0c"}},
                }
                self.check_arguments(schema, {"value": value})
        self.check_arguments(
            {
                "$defs": {
                    "args": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                    }
                },
                "$ref": "#/$defs/args",
            },
            {"value": "null"},
        )
        self.check_arguments(
            {
                "$defs": {"cycle": {"$ref": "#/$defs/cycle"}},
                "properties": {"value": {"$ref": "#/$defs/cycle"}},
            },
            {"value": "auto"},
        )

    def test_root_unions_preserve_each_value_type(self):
        text_branch = {
            "type": "object",
            "properties": {"kind": {"const": "text"}, "value": {"type": "string"}},
        }
        count_branch = {
            "type": "object",
            "properties": {"kind": {"const": "count"}, "value": {"type": "integer"}},
        }
        for keyword in ("anyOf", "oneOf"):
            for branches in ([text_branch, count_branch], [count_branch, text_branch]):
                self.check_arguments(
                    {keyword: branches}, {"kind": "text", "value": "auto"}
                )
                self.check_arguments({keyword: branches}, {"kind": "count", "value": 7})
        self.check_arguments(
            {"allOf": [count_branch, {"required": ["kind", "value"]}]},
            {"kind": "count", "value": 7},
        )

    def test_plain_strings_still_stream_before_value_completes(self):
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="inspect",
                    parameters={
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                    },
                ),
            )
        ]
        detector = Glm47MoeDetector()
        first = detector.parse_streaming_increment(
            "<tool_call>inspect<arg_key>value</arg_key><arg_value>hello", tools
        )
        self.assertIn('"hello', "".join(call.parameters for call in first.calls))
        last = detector.parse_streaming_increment(
            " world</arg_value></tool_call>", tools
        )
        self.assertEqual(
            json.loads("".join(call.parameters for call in first.calls + last.calls)),
            {"value": "hello world"},
        )

    def test_native_strings_reject_disallowed_json_types_and_enum_values(self):
        for field, value in [
            ({"type": ["string", "object"]}, "7"),
            ({"type": ["object", "string"]}, "7"),
            ({"oneOf": [{"type": "string"}, {"type": "object"}]}, "7"),
            ({"type": ["string", "array"]}, "1e2"),
            ({"type": ["string", "integer"]}, "true"),
            ({"type": ["string", "integer"]}, "null"),
            ({"enum": ["7", None]}, "7"),
            ({"enum": ["7", 8]}, "7"),
            ({"enum": ["7", 8]}, 8),
            ({"enum": ["true", 7]}, "true"),
            ({"enum": ["null", 7]}, "null"),
            ({"enum": ["7", False]}, "7"),
            ({"type": ["string", "integer"]}, 7),
            ({"type": ["string", "boolean"]}, True),
            ({"type": ["string", "null"]}, None),
            ({"type": ["string", "object"]}, {"ok": True}),
            ({"type": ["number", "string"], "enum": [1, "auto"]}, 1),
            ({"type": ["number", "string"], "enum": [1, "auto"]}, 1.0),
        ]:
            self.check_arguments(
                {"type": "object", "properties": {"value": field}}, {"value": value}
            )
        self.check_arguments(
            {
                "$defs": {"value": {"type": ["string", "object"]}},
                "properties": {"value": {"$ref": "#/$defs/value"}},
            },
            {"value": "7"},
        )

    def test_json_looking_string_bytes_are_preserved(self):
        for value in ('{"a":1}', "[1,2]", '{ "a" : true }'):
            for field in ({"type": "string"}, {"const": value}):
                self.check_arguments({"properties": {"value": field}}, {"value": value})

    def test_nonstream_numeric_strings_with_integer_enum(self):
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="inspect",
                    parameters={
                        "properties": {"value": {"type": "number", "enum": [1, 2]}}
                    },
                ),
            )
        ]
        for raw in ('"1"', r"\"1\""):
            text = (
                "<tool_call>inspect<arg_key>value</arg_key><arg_value>"
                + raw
                + "</arg_value></tool_call>"
            )
            calls = Glm47MoeDetector().detect_and_parse(text, tools).calls
            self.assertEqual(json.loads(calls[0].parameters), {"value": 1})

    def test_numeric_values_are_serialized_after_the_value_closes(self):
        for value_type in ("number", "integer"):
            schema = {"properties": {"value": {"type": value_type}}}
            for raw, expected in (
                ("123abc", "123abc"),
                ('"123"', 123),
                (r"\"123\"", 123),
                ("-12", -12),
                ("1.25e2", 125.0),
                ("", ""),
            ):
                self.check_arguments(schema, {"value": expected}, {"value": raw})

    def test_numeric_prefix_does_not_commit_to_invalid_json(self):
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="inspect",
                    parameters={"properties": {"value": {"type": "number"}}},
                ),
            )
        ]
        detector = Glm47MoeDetector()
        prefix = detector.parse_streaming_increment(
            "<tool_call>inspect<arg_key>value</arg_key><arg_value>123", tools
        ).calls
        self.assertNotIn("123", "".join(call.parameters for call in prefix))
        suffix = detector.parse_streaming_increment(
            "abc</arg_value></tool_call>", tools
        ).calls
        self.assertEqual(
            json.loads("".join(call.parameters for call in prefix + suffix)),
            {"value": "123abc"},
        )

    def test_completed_siblings_disambiguate_root_union(self):
        for keyword in ("oneOf", "anyOf"):
            for discriminator in ("const", "enum"):
                branches = [
                    {
                        "properties": {
                            "kind": {
                                discriminator: [kind]
                                if discriminator == "enum"
                                else kind
                            },
                            "value": {"type": value_type},
                        }
                    }
                    for kind, value_type in (("text", "string"), ("count", "integer"))
                ]
                for ordered in (branches, list(reversed(branches))):
                    for schema in (
                        {keyword: ordered},
                        {keyword: [{"allOf": [branch]} for branch in ordered]},
                        {
                            "$defs": {
                                str(i): branch for i, branch in enumerate(ordered)
                            },
                            keyword: [{"$ref": f"#/$defs/{i}"} for i in range(2)],
                        },
                    ):
                        self.check_arguments(schema, {"kind": "text", "value": "7"})
                        self.check_arguments(schema, {"kind": "count", "value": 7})
                        self.check_arguments(schema, {"value": "7", "kind": "text"})
                        self.check_arguments(schema, {"value": 7, "kind": "count"})

    def test_completed_arguments_reset_between_tool_calls(self):
        schema = {
            "oneOf": [
                {
                    "properties": {
                        "kind": {"const": "text"},
                        "value": {"type": "string"},
                    }
                },
                {
                    "properties": {
                        "kind": {"const": "count"},
                        "value": {"type": "integer"},
                    }
                },
            ]
        }
        tools = [
            Tool(type="function", function=Function(name="inspect", parameters=schema))
        ]
        tools.append(
            Tool(
                type="function",
                function=Function(
                    name="plain",
                    parameters={"properties": {"value": {"type": "string"}}},
                ),
            )
        )
        detector = Glm47MoeDetector()
        outputs = {}
        for kind, discriminator_first in (
            ("text", True),
            ("count", False),
            ("text", False),
        ):
            tag = f"<arg_key>kind</arg_key><arg_value>{kind}</arg_value>"
            value = "<arg_key>value</arg_key><arg_value>7</arg_value>"
            text = (
                "<tool_call>inspect"
                + (tag + value if discriminator_first else value + tag)
                + "</tool_call>"
            )
            for char in text:
                for call in detector.parse_streaming_increment(char, tools).calls:
                    outputs[call.tool_index] = (
                        outputs.get(call.tool_index, "") + call.parameters
                    )
        self.assertEqual(json.loads(outputs[0]), {"kind": "text", "value": "7"})
        self.assertEqual(json.loads(outputs[1]), {"kind": "count", "value": 7})
        self.assertEqual(json.loads(outputs[2]), {"kind": "text", "value": "7"})
        first = detector.parse_streaming_increment(
            "<tool_call>plain<arg_key>value</arg_key><arg_value>hello", tools
        )
        self.assertIn('"hello', "".join(call.parameters for call in first.calls))
        last = detector.parse_streaming_increment("</arg_value></tool_call>", tools)
        self.assertEqual(
            json.loads("".join(call.parameters for call in first.calls + last.calls)),
            {"value": "hello"},
        )


if __name__ == "__main__":
    unittest.main()
