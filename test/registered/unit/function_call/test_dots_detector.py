import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.dots_detector import DotsToolDetector
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.parser.reasoning_parser import Qwen3Detector, ReasoningParser
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _tool(name: str, properties: dict) -> Tool:
    return Tool(
        type="function",
        function=Function(
            name=name,
            description="test tool",
            parameters={"type": "object", "properties": properties},
        ),
    )


class TestDotsToolDetector(unittest.TestCase):
    def test_dots_parsers_are_registered(self):
        self.assertIs(ReasoningParser.DetectorMap["dots"], Qwen3Detector)
        self.assertIs(FunctionCallParser.ToolCallParserEnum["dots"], DotsToolDetector)

    def test_dots_reasoning_uses_qwen3_format(self):
        parser = ReasoningParser("dots", stream_reasoning=False, force_reasoning=True)
        reasoning, content = parser.parse_non_stream(
            "Need to inspect inputs.</think>Final answer"
        )
        self.assertEqual(reasoning, "Need to inspect inputs.")
        self.assertEqual(content, "Final answer")

    def test_non_stream_xml_converts_schema_types_and_resolves_ref(self):
        tool = Tool(
            type="function",
            function=Function(
                name="set_location",
                description="Set location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"$ref": "#/$defs/Location"},
                        "days": {"type": "integer"},
                        "include_weather": {"type": "boolean"},
                    },
                    "$defs": {
                        "Location": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        }
                    },
                },
            ),
        )
        parser = FunctionCallParser([tool], "dots")
        text = (
            "ok<dots_function_call>"
            '<invoke name="set_location">'
            '<parameter name="location">{"city": "Shanghai"}</parameter>'
            '<parameter name="days">3</parameter>'
            '<parameter name="include_weather">true</parameter>'
            "</invoke>"
            "</dots_function_call>"
        )

        normal_text, calls = parser.parse_non_stream(text)

        self.assertEqual(normal_text, "ok")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "set_location")
        self.assertEqual(
            json.loads(calls[0].parameters),
            {
                "location": {"city": "Shanghai"},
                "days": 3,
                "include_weather": True,
            },
        )

    def test_non_stream_supports_multiple_invokes_and_json_fallback(self):
        tools = [
            _tool("search", {"query": {"type": "string"}}),
            _tool("open", {"id": {"type": "integer"}}),
        ]
        parser = FunctionCallParser(tools, "dots")
        text = (
            "<dots_function_call>"
            '<invoke name="search"><parameter name="query">chairs</parameter></invoke>'
            '<invoke name="open"><parameter name="id">7</parameter></invoke>'
            "</dots_function_call>"
            '<dots_function_call>{"name":"search","arguments":{"query":"tables"}}'
            "</dots_function_call>"
        )

        _, calls = parser.parse_non_stream(text)

        self.assertEqual([call.name for call in calls], ["search", "open", "search"])
        self.assertEqual(
            [json.loads(call.parameters) for call in calls],
            [
                {"query": "chairs"},
                {"id": 7},
                {"query": "tables"},
            ],
        )

    def test_streaming_buffers_partial_marker_and_emits_all_complete_calls(self):
        tools = [_tool("search", {"query": {"type": "string"}})]
        detector = DotsToolDetector()
        chunks = [
            "visible<dots_func",
            (
                "tion_call>"
                '<invoke name="search"><parameter name="query">chairs</parameter></invoke>'
                "</dots_function_call>"
                "<dots_function_call>"
                '<invoke name="search"><parameter name="query">tables</parameter></invoke>'
                "</dots_function_call>"
            ),
        ]

        results = [detector.parse_streaming_increment(chunk, tools) for chunk in chunks]

        self.assertEqual("".join(result.normal_text for result in results), "visible")
        calls = [call for result in results for call in result.calls]
        self.assertEqual([call.tool_index for call in calls], [0, 1])
        self.assertEqual(
            [json.loads(call.parameters) for call in calls],
            [
                {"query": "chairs"},
                {"query": "tables"},
            ],
        )

    def test_streaming_filters_unknown_tools_and_surfaces_the_content(self):
        tools = [_tool("search", {"query": {"type": "string"}})]
        detector = DotsToolDetector()
        text = (
            "<dots_function_call>"
            '<invoke name="ghost"><parameter name="query">chairs</parameter></invoke>'
            "</dots_function_call>"
        )

        result = detector.parse_streaming_increment(text, tools)

        self.assertEqual(result.calls, [])
        self.assertIn("ghost", result.normal_text)
        self.assertEqual(detector._buffer, "")

    def test_streaming_malformed_block_does_not_block_a_later_valid_call(self):
        tools = [_tool("search", {"query": {"type": "string"}})]
        detector = DotsToolDetector()

        malformed = detector.parse_streaming_increment(
            "<dots_function_call>garbage</dots_function_call>", tools
        )
        valid = detector.parse_streaming_increment(
            "<dots_function_call>"
            '<invoke name="search"><parameter name="query">chairs</parameter></invoke>'
            "</dots_function_call>",
            tools,
        )

        self.assertEqual(malformed.calls, [])
        self.assertEqual(malformed.normal_text, "garbage")
        self.assertEqual([call.name for call in valid.calls], ["search"])

    def test_streaming_strips_stray_end_marker_from_normal_text(self):
        detector = DotsToolDetector()

        result = detector.parse_streaming_increment(
            "some text </dots_function_call>", []
        )

        self.assertEqual(result.calls, [])
        self.assertEqual(result.normal_text, "some text ")

    def test_streaming_flushes_partial_opening_marker_at_eof(self):
        tools = [_tool("search", {"query": {"type": "string"}})]
        detector = DotsToolDetector()

        result = detector.parse_streaming_increment("answer <dots_func", tools)

        self.assertEqual(result.calls, [])
        self.assertEqual(result.normal_text, "answer ")
        self.assertEqual(detector.flush_pending_normal_text(), "<dots_func")
        self.assertEqual(detector.flush_pending_normal_text(), "")

    def test_streaming_emits_complete_json_body_before_end_marker_without_duplication(
        self,
    ):
        tools = [_tool("search", {"query": {"type": "string"}})]
        detector = DotsToolDetector()

        opening = detector.parse_streaming_increment("<dots_function_call>", tools)
        body = detector.parse_streaming_increment(
            '{"name":"search","arguments":{"query":"chairs"}}', tools
        )
        closing = detector.parse_streaming_increment("</dots_function_call>", tools)

        self.assertEqual(opening.calls, [])
        self.assertEqual([call.name for call in body.calls], ["search", None])
        self.assertEqual(
            "".join(call.parameters for call in body.calls), '{"query": "chairs"}'
        )
        self.assertEqual(closing.calls, [])


if __name__ == "__main__":
    unittest.main()
