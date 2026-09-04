import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.k2_v3_detector import K2V3Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_weather_tool() -> Tool:
    return Tool(
        type="function",
        function=Function(
            name="get_weather",
            description="Get weather",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "options": {"type": "object"},
                },
            },
        ),
    )


def make_reasoning_with_tool_marker(tag: str) -> str:
    return (
        f"<ifm|{tag}>Consider "
        "<ifm|tool_call>get_weather"
        "<ifm|arg_key>city</ifm|arg_key>"
        "<ifm|arg_value>Boston</ifm|arg_value>"
        "</ifm|tool_call> as hypothetical text."
        f"</ifm|{tag}>\n"
    )


class TestK2V3Detector(CustomTestCase):
    def setUp(self):
        self.tools = [make_weather_tool()]

    def test_xml_and_typed_values(self):
        text = (
            "<ifm|tool_call>get_weather\n"
            "<ifm|arg_key>city</ifm|arg_key>\n"
            "<ifm|arg_type>string</ifm|arg_type>\n"
            "<ifm|arg_value>  Boston  </ifm|arg_value>\n"
            "<ifm|arg_key>days</ifm|arg_key>\n"
            "<ifm|arg_type>integer</ifm|arg_type>\n"
            "<ifm|arg_value>3</ifm|arg_value>\n"
            "<ifm|arg_key>options</ifm|arg_key>\n"
            '<ifm|arg_value>{"units": "metric"}</ifm|arg_value>\n'
            "</ifm|tool_call>"
        )
        result = K2V3Detector().detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, "")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(
            json.loads(result.calls[0].parameters),
            {
                "city": "  Boston  ",
                "days": 3,
                "options": {"units": "metric"},
            },
        )

    def test_json_content_is_detected_from_wire(self):
        wire = (
            "<ifm|tool_calls>\n<ifm|tool_call>"
            '{"name":"get_weather","arguments":{"city":"Tokyo","days":2}}'
            "</ifm|tool_call>\n</ifm|tool_calls>"
        )
        result = K2V3Detector().detect_and_parse(wire, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(
            json.loads(result.calls[0].parameters),
            {"city": "Tokyo", "days": 2},
        )

    def test_json_strings_are_not_reinterpreted_without_a_schema(self):
        wire = (
            "<ifm|tool_call>"
            '{"name":"get_weather","arguments":'
            '{"numeric":"123","boolean":"true","object":"{}"}}'
            "</ifm|tool_call>"
        )
        result = K2V3Detector().detect_and_parse(wire, self.tools)
        self.assertEqual(
            json.loads(result.calls[0].parameters),
            {"numeric": "123", "boolean": "true", "object": "{}"},
        )

    def test_local_schema_ref_preserves_xml_string_value(self):
        tool = Tool(
            type="function",
            function=Function(
                name="lookup",
                parameters={
                    "type": "object",
                    "$defs": {"Identifier": {"type": "string"}},
                    "properties": {"id": {"$ref": "#/$defs/Identifier"}},
                },
            ),
        )
        wire = (
            "<ifm|tool_call>lookup"
            "<ifm|arg_key>id</ifm|arg_key>"
            "<ifm|arg_value>123</ifm|arg_value>"
            "</ifm|tool_call>"
        )
        result = K2V3Detector().detect_and_parse(wire, [tool])
        self.assertEqual(json.loads(result.calls[0].parameters), {"id": "123"})

    def test_xml_typed_union_uses_wire_value_type(self):
        tool = Tool(
            type="function",
            function=Function(
                name="lookup",
                parameters={
                    "type": "object",
                    "properties": {
                        "id": {"anyOf": [{"type": "string"}, {"type": "integer"}]}
                    },
                },
            ),
        )
        wire = (
            "<ifm|tool_call>lookup"
            "<ifm|arg_key>id</ifm|arg_key>"
            "<ifm|arg_type>integer</ifm|arg_type>"
            "<ifm|arg_value>123</ifm|arg_value>"
            "</ifm|tool_call>"
        )
        result = K2V3Detector().detect_and_parse(wire, [tool])
        self.assertEqual(json.loads(result.calls[0].parameters), {"id": 123})

    def test_non_string_json_function_name_is_forwarded_as_text(self):
        wire = '<ifm|tool_call>{"name":123,"arguments":{}}</ifm|tool_call>'
        result = K2V3Detector().detect_and_parse(wire, self.tools)
        self.assertEqual(result.calls, [])
        self.assertEqual(result.normal_text, wire)

    def test_parallel_calls_keep_wire_order_and_indices(self):
        wire = (
            "<ifm|tool_calls>\n"
            "<ifm|tool_call>get_weather"
            "<ifm|arg_key>city</ifm|arg_key>"
            "<ifm|arg_value>Tokyo</ifm|arg_value>"
            "</ifm|tool_call>\n"
            "<ifm|tool_call>get_weather"
            "<ifm|arg_key>city</ifm|arg_key>"
            "<ifm|arg_value>Boston</ifm|arg_value>"
            "</ifm|tool_call>\n"
            "</ifm|tool_calls>"
        )
        result = K2V3Detector().detect_and_parse(wire, self.tools)
        self.assertEqual(result.normal_text, "")
        self.assertEqual([call.tool_index for call in result.calls], [0, 1])
        self.assertEqual(
            [json.loads(call.parameters)["city"] for call in result.calls],
            ["Tokyo", "Boston"],
        )

    def test_non_streaming_preserves_reasoning_for_ordinary_answer(self):
        wire = (
            " \n" + make_reasoning_with_tool_marker("think_fast") + "The answer is 42."
        )
        parser = FunctionCallParser(self.tools, "k2_horizon")
        normal, calls = parser.parse_non_stream(wire)
        self.assertEqual(normal, wire)
        self.assertEqual(calls, [])

    def test_streaming_preserves_reasoning_for_ordinary_answer(self):
        wire = (
            " \n"
            + make_reasoning_with_tool_marker("think_faster")
            + "The answer is 42."
        )
        parser = FunctionCallParser(self.tools, "k2_horizon")
        normal = ""
        calls = []
        for char in wire:
            new_normal, new_calls = parser.parse_stream_chunk(char)
            normal += new_normal
            calls.extend(new_calls)
        end_normal, end_calls = parser.parse_stream_end()
        normal += end_normal
        calls.extend(end_calls)

        self.assertEqual(normal, wire)
        self.assertEqual(calls, [])

    def test_non_streaming_tool_call_preserves_reasoning_prefix(self):
        reasoning = make_reasoning_with_tool_marker("think")
        wire = reasoning + (
            "<ifm|tool_calls>\n"
            "<ifm|tool_call>get_weather"
            "<ifm|arg_key>city</ifm|arg_key>"
            "<ifm|arg_value>Tokyo</ifm|arg_value>"
            "</ifm|tool_call>\n"
            "</ifm|tool_calls>"
        )
        parser = FunctionCallParser(self.tools, "k2_horizon")
        normal, calls = parser.parse_non_stream(wire)
        self.assertEqual(normal, reasoning)
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0].parameters), {"city": "Tokyo"})

    def test_forced_reasoning_without_opening_tag_is_preserved(self):
        reasoning = "work</ifm|think>\n"
        wire = reasoning + (
            "<ifm|tool_calls><ifm|tool_call>get_weather"
            "<ifm|arg_key>city</ifm|arg_key>"
            "<ifm|arg_value>Tokyo</ifm|arg_value>"
            "</ifm|tool_call></ifm|tool_calls>"
        )
        normal, calls = FunctionCallParser(self.tools, "k2_horizon").parse_non_stream(
            wire
        )
        self.assertEqual(normal, reasoning)
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0].parameters), {"city": "Tokyo"})

    def test_streaming_tool_call_preserves_reasoning_at_every_boundary(self):
        reasoning = make_reasoning_with_tool_marker("think")
        wire = reasoning + (
            "<ifm|tool_calls>\n"
            "<ifm|tool_call>get_weather"
            "<ifm|arg_key>city</ifm|arg_key>"
            "<ifm|arg_value>東京</ifm|arg_value>"
            "</ifm|tool_call>\n"
            "</ifm|tool_calls>"
        )
        parser = FunctionCallParser(self.tools, "k2_horizon")
        normal = ""
        calls = []
        for char in wire:
            new_normal, new_calls = parser.parse_stream_chunk(char)
            normal += new_normal
            calls.extend(new_calls)
        end_normal, end_calls = parser.parse_stream_end()
        normal += end_normal
        calls.extend(end_calls)

        self.assertEqual(normal, reasoning)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(json.loads(calls[0].parameters), {"city": "東京"})

    def test_malformed_complete_block_is_forwarded_as_text(self):
        wire = (
            "prefix<ifm|tool_call>get_weather"
            "<ifm|arg_key>city</ifm|arg_key>"
            "</ifm|tool_call>suffix"
        )
        result = K2V3Detector().detect_and_parse(wire, self.tools)
        self.assertEqual(result.calls, [])
        self.assertEqual(result.normal_text, wire)

    def test_malformed_canonical_group_is_forwarded_intact(self):
        wire = (
            "prefix<ifm|tool_calls>\n"
            "<ifm|tool_call>get_weather"
            "<ifm|arg_key>city</ifm|arg_key>"
            "</ifm|tool_call>\n"
            "</ifm|tool_calls>suffix"
        )
        result = K2V3Detector().detect_and_parse(wire, self.tools)
        self.assertEqual(result.calls, [])
        self.assertEqual(result.normal_text, wire)

    def test_unknown_tool_respects_forwarding_policy(self):
        wire = "<ifm|tool_call>unknown</ifm|tool_call>"
        with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(False):
            dropped = K2V3Detector().detect_and_parse(wire, self.tools)
        self.assertEqual(dropped.calls, [])

        with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(True):
            forwarded = K2V3Detector().detect_and_parse(wire, self.tools)
        self.assertEqual(len(forwarded.calls), 1)
        self.assertEqual(forwarded.calls[0].name, "unknown")

    def test_any_schema_preserves_json_value(self):
        tool = make_weather_tool()
        tool.function.parameters["properties"]["options"] = {"type": "any"}
        wire = (
            "<ifm|tool_call>"
            '{"name":"get_weather","arguments":{"options":{"units":"metric"}}}'
            "</ifm|tool_call>"
        )
        result = K2V3Detector().detect_and_parse(wire, [tool])
        self.assertEqual(
            json.loads(result.calls[0].parameters),
            {"options": {"units": "metric"}},
        )

    def test_unterminated_stream_is_released_on_finish(self):
        wire = "prefix<ifm|tool_call>get_weather<ifm|arg_key>city"
        detector = K2V3Detector()
        streamed = detector.parse_streaming_increment(wire, self.tools)
        self.assertEqual(streamed.normal_text, "prefix")
        self.assertEqual(streamed.calls, [])
        finished = detector.finish(self.tools)
        self.assertEqual(
            finished.normal_text,
            "<ifm|tool_call>get_weather<ifm|arg_key>city",
        )
        self.assertEqual(finished.calls, [])

    def test_function_call_parser_registry(self):
        parser = FunctionCallParser(
            tools=self.tools,
            tool_call_parser="k2_horizon",
        )
        self.assertIsInstance(parser.detector, K2V3Detector)


if __name__ == "__main__":
    unittest.main()
