"""Unit tests for MinimaxM2Detector -- no server, no model loading."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.minimax_m2 import MinimaxM2Detector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1.0, "stage-a-test-cpu")


class TestMinimaxM2Detector(unittest.TestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "days": {"type": "integer"},
                            "include_hourly": {"type": "boolean"},
                            "tags": {"type": "array"},
                        },
                        "required": ["location"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="search",
                    description="Search the web",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                        "required": ["query"],
                    },
                ),
            ),
        ]
        self.detector = MinimaxM2Detector()

    def test_detect_and_parse_converts_schema_typed_parameters(self):
        text = (
            "before "
            "<minimax:tool_call>"
            '<invoke name="get_weather">'
            '<parameter name="location">Paris</parameter>'
            '<parameter name="days">3</parameter>'
            '<parameter name="include_hourly">true</parameter>'
            '<parameter name="tags">["rain", "wind"]</parameter>'
            "</invoke>"
            "</minimax:tool_call>"
            " after"
        )

        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, "before  after")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].tool_index, 0)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(
            json.loads(result.calls[0].parameters),
            {
                "location": "Paris",
                "days": 3,
                "include_hourly": True,
                "tags": ["rain", "wind"],
            },
        )

    def test_detect_and_parse_incomplete_block_stays_normal_text(self):
        text = (
            "prefix "
            "<minimax:tool_call>"
            '<invoke name="get_weather">'
            '<parameter name="location">Rome</parameter>'
        )

        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, text)
        self.assertEqual(result.calls, [])

    def test_streaming_consumes_outer_end_token_and_emits_trailing_text(self):
        chunks = [
            "Before ",
            '<minimax:tool_call><invoke name="get_weather">',
            '<parameter name="location">Rome</parameter>',
            '<parameter name="days">2</parameter></invoke>',
            "</minimax:tool_call> after",
        ]
        calls = []
        normal_text = ""

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            calls.extend(result.calls)
            normal_text += result.normal_text

        self.assertEqual(normal_text, "Before  after")
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(calls[0].parameters, "")
        self.assertEqual(
            json.loads("".join(call.parameters for call in calls if call.parameters)),
            {
                "location": "Rome",
                "days": 2,
            },
        )
        self.assertEqual(self.detector._buf, "")
        self.assertFalse(self.detector._in_tool_call)


if __name__ == "__main__":
    unittest.main()
