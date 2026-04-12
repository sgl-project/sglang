"""Unit tests for Llama32Detector — no server, no model loading."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.llama32_detector import Llama32Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "stage-a-test-cpu")


class TestLlama32Detector(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="get_time",
                    description="Get current time",
                    parameters={
                        "type": "object",
                        "properties": {"timezone": {"type": "string"}},
                        "required": ["timezone"],
                    },
                ),
            ),
        ]
        self.detector = Llama32Detector()

    # ==================== has_tool_call ====================

    def test_has_tool_call_python_tag(self):
        self.assertTrue(
            self.detector.has_tool_call(
                '<|python_tag|>{"name": "get_weather", "arguments": {"city": "Paris"}}'
            )
        )

    def test_has_tool_call_bare_json(self):
        """Text starting with '{' is treated as a tool call (no python_tag needed)."""
        self.assertTrue(
            self.detector.has_tool_call(
                '{"name": "get_weather", "arguments": {"city": "Paris"}}'
            )
        )

    def test_has_tool_call_false(self):
        self.assertFalse(self.detector.has_tool_call("The weather is nice today."))

    def test_has_tool_call_false_json_not_at_start(self):
        """JSON not at position 0 without python_tag is not a tool call."""
        self.assertFalse(self.detector.has_tool_call('Sure! {"name": "get_weather"}'))

    # ==================== detect_and_parse ====================

    def test_single_tool_call_with_tag(self):
        text = '<|python_tag|>{"name": "get_weather", "arguments": {"city": "Paris"}}'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[0].tool_index, 0)
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["city"], "Paris")

    def test_single_tool_call_bare_json(self):
        """Llama32 accepts a bare JSON object (no python_tag) as a tool call."""
        text = '{"name": "get_weather", "arguments": {"city": "Paris"}}'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")

    def test_multiple_tool_calls_semicolon_separator(self):
        """Multiple tool calls are separated by ';' (tool_call_separator)."""
        text = (
            '<|python_tag|>'
            '{"name": "get_weather", "arguments": {"city": "Paris"}}'
            ';'
            '{"name": "get_time", "arguments": {"timezone": "UTC"}}'
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "get_time")
        self.assertEqual(result.calls[1].tool_index, 1)

    def test_no_tool_call(self):
        text = "The weather in Paris is sunny today."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, text)

    def test_normal_text_before_python_tag(self):
        text = 'Sure!<|python_tag|>{"name": "get_weather", "arguments": {"city": "Paris"}}'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertIn("Sure!", result.normal_text)

    def test_python_dict_fallback(self):
        """Single-quoted Python dict is converted to JSON via ast.literal_eval."""
        text = "<|python_tag|>{'name': 'get_weather', 'arguments': {'city': 'Paris'}}"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["city"], "Paris")

    def test_unknown_tool_skipped(self):
        text = '<|python_tag|>{"name": "undefined_func", "arguments": {}}'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)

    def test_parameters_key_accepted(self):
        """parse_base_json accepts 'parameters' in addition to 'arguments'."""
        text = '<|python_tag|>{"name": "get_weather", "parameters": {"city": "Berlin"}}'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["city"], "Berlin")

    def test_trailing_garbage_after_separator(self):
        """Trailing garbage after a valid call is returned as normal_text via safe_idx."""
        text = (
            '<|python_tag|>'
            '{"name": "get_weather", "arguments": {"city": "Paris"}}'
            ';not json garbage'
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertIn("not json", result.normal_text)

    # ==================== structure_info ====================

    def test_structure_info(self):
        get_info = self.detector.structure_info()
        info = get_info("get_weather")
        self.assertIn("<|python_tag|>", info.begin)
        self.assertIn("get_weather", info.begin)
        self.assertEqual(info.end, "}")
        self.assertEqual(info.trigger, "<|python_tag|>")

    # ==================== Streaming ====================

    def test_streaming_plain_text(self):
        """Text with no tool markers is returned immediately as normal_text."""
        result = self.detector.parse_streaming_increment("Hello, world!", self.tools)
        self.assertEqual(result.normal_text, "Hello, world!")
        self.assertEqual(result.calls, [])

    def test_streaming_partial_bot_token_buffered(self):
        """A chunk ending with a partial <|python_tag|> is held in the buffer."""
        result = self.detector.parse_streaming_increment("<|python_t", self.tools)
        self.assertEqual(result.normal_text, "")
        self.assertEqual(result.calls, [])

    def test_streaming_python_dict_conversion(self):
        """Python single-quote dict is regex-converted before base streaming parse."""
        # Single-quote values are converted in the buffer before parsing.
        # Sending a complete chunk allows the base streaming to parse the full call.
        text = "<|python_tag|>{'name': 'get_weather', 'arguments': {'city': 'Paris'}}"
        result = self.detector.parse_streaming_increment(text, self.tools)
        # After regex conversion, the buffer becomes valid JSON and the call is parsed.
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")


if __name__ == "__main__":
    unittest.main()
