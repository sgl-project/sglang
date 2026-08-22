"""Unit tests for Qwen25Detector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.qwen25_detector import Qwen25Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestQwen25Detector(CustomTestCase):
    """Test Qwen25Detector streaming and non-streaming multi-tool-call parsing."""

    def setUp(self):
        self.detector = Qwen25Detector()
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_current_weather",
                    description="Get the current weather in a given location",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "The city name",
                            },
                            "state": {
                                "type": "string",
                                "description": "Two-letter state abbreviation",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["city", "state", "unit"],
                    },
                ),
            ),
        ]

    # -- Non-streaming tests --

    def test_detect_and_parse_multiple_tool_calls(self):
        text = (
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"city": "NYC", "state": "NY", "unit": "fahrenheit"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"city": "Baltimore", "state": "MD", "unit": "fahrenheit"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"city": "Minneapolis", "state": "MN", "unit": "fahrenheit"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"city": "Los Angeles", "state": "CA", "unit": "fahrenheit"}}\n</tool_call>'
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 4)
        cities = [json.loads(c.parameters)["city"] for c in result.calls]
        self.assertEqual(cities, ["NYC", "Baltimore", "Minneapolis", "Los Angeles"])

    def test_detect_and_parse_with_normal_text_prefix(self):
        text = (
            "Sure, let me check the weather.\n"
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"city": "NYC", "state": "NY", "unit": "celsius"}}\n</tool_call>'
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertIn("let me check", result.normal_text)

    # -- Streaming tests --

    def _collect_streaming_tool_calls(self, chunks):
        """Helper: feed chunks through streaming parser and collect tool calls by index."""
        tool_calls_by_index = {}
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            for call in result.calls:
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }
                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters
        return tool_calls_by_index

    def test_streaming_multiple_tool_calls(self):
        """Core regression test: multiple tool calls must all be parsed in streaming mode."""
        chunks = [
            "<tool_call>\n",
            '{"name": "get_current_weather",',
            ' "arguments": {"city": "NYC", "state": "NY", "unit": "fahrenheit"}}',
            "\n</tool_call>\n",
            "<tool_call>\n",
            '{"name": "get_current_weather",',
            ' "arguments": {"city": "Baltimore", "state": "MD", "unit": "fahrenheit"}}',
            "\n</tool_call>\n",
            "<tool_call>\n",
            '{"name": "get_current_weather",',
            ' "arguments": {"city": "LA", "state": "CA", "unit": "fahrenheit"}}',
            "\n</tool_call>",
        ]
        result = self._collect_streaming_tool_calls(chunks)
        self.assertEqual(len(result), 3, f"Expected 3 tool calls, got {len(result)}")
        cities = [json.loads(result[i]["parameters"])["city"] for i in sorted(result)]
        self.assertEqual(cities, ["NYC", "Baltimore", "LA"])

    def test_streaming_multiple_tool_calls_fused_chunks(self):
        """Test when separator and next bot_token arrive in a single chunk."""
        chunks = [
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"city": "NYC", "state": "NY", "unit": "fahrenheit"}}',
            '\n</tool_call>\n<tool_call>\n{"name": "get_current_weather",',
            ' "arguments": {"city": "LA", "state": "CA", "unit": "fahrenheit"}}',
            "\n</tool_call>",
        ]
        result = self._collect_streaming_tool_calls(chunks)
        self.assertEqual(len(result), 2, f"Expected 2 tool calls, got {len(result)}")
        cities = [json.loads(result[i]["parameters"])["city"] for i in sorted(result)]
        self.assertEqual(cities, ["NYC", "LA"])

    def test_streaming_multiple_tool_calls_char_by_char_separator(self):
        """Test when the separator between tool calls arrives character by character."""
        call1 = '{"name": "get_current_weather", "arguments": {"city": "NYC", "state": "NY", "unit": "fahrenheit"}}'
        call2 = '{"name": "get_current_weather", "arguments": {"city": "LA", "state": "CA", "unit": "celsius"}}'
        separator = "\n</tool_call>\n<tool_call>\n"

        chunks = ["<tool_call>\n", call1]
        for ch in separator:
            chunks.append(ch)
        chunks.append(call2)
        chunks.append("\n</tool_call>")

        result = self._collect_streaming_tool_calls(chunks)
        self.assertEqual(len(result), 2, f"Expected 2 tool calls, got {len(result)}")
        cities = [json.loads(result[i]["parameters"])["city"] for i in sorted(result)]
        self.assertEqual(cities, ["NYC", "LA"])


if __name__ == "__main__":
    import unittest

    unittest.main()
