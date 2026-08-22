"""Unit tests for Lfm2Detector — no server, no model loading."""

import json
import warnings

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.lfm2_detector import Lfm2Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestLfm2Detector(CustomTestCase):
    """Tests for LFM2 (Liquid Foundation Model 2) function call detector."""

    def setUp(self):
        """Set up test tools and detector."""
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name",
                            },
                            "unit": {
                                "type": "string",
                                "description": "Temperature unit",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["city"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="search",
                    description="Search for information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query",
                            },
                        },
                        "required": ["query"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="calculator",
                    description="Perform calculations",
                    parameters={
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Math expression",
                            },
                        },
                        "required": ["expression"],
                    },
                ),
            ),
        ]
        self.detector = Lfm2Detector()

    # ==================== has_tool_call tests ====================

    def test_has_tool_call_false(self):
        """Test no false positives for regular text."""
        text = "The weather in Paris is nice today."
        self.assertFalse(self.detector.has_tool_call(text))

    def test_has_tool_call_partial_marker(self):
        """Test that partial markers are detected (start token present)."""
        text = '<|tool_call_start|>[get_weather(city="Paris")'
        self.assertTrue(self.detector.has_tool_call(text))

    # ==================== detect_and_parse tests (Pythonic format) ====================

    def test_detect_and_parse_pythonic_simple(self):
        """Test parsing a simple Pythonic format tool call."""
        text = '<|tool_call_start|>[get_weather(city="Paris")]<|tool_call_end|>'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[0].tool_index, 0)

        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["city"], "Paris")

    def test_detect_and_parse_pythonic_invalid_escape(self):
        """An invalid Python escape (e.g. "\\d+") must not drop the tool call."""
        text = '<|tool_call_start|>[search(query="\\d+")]<|tool_call_end|>'
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", SyntaxWarning)
            result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "search")
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["query"], "\\d+")
        self.assertFalse(
            any(isinstance(w.message, SyntaxWarning) for w in caught),
            [str(w.message) for w in caught],
        )

    def test_detect_and_parse_pythonic_multiple_args(self):
        """Test parsing with multiple arguments."""
        text = '<|tool_call_start|>[get_weather(city="London", unit="celsius")]<|tool_call_end|>'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")

        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["city"], "London")
        self.assertEqual(params["unit"], "celsius")

    def test_detect_and_parse_pythonic_no_args(self):
        """Test parsing function with no arguments."""
        # Add a no-arg tool for this test
        tools_with_noarg = self.tools + [
            Tool(
                type="function",
                function=Function(
                    name="get_time",
                    description="Get current time",
                    parameters={"type": "object", "properties": {}},
                ),
            ),
        ]
        text = "<|tool_call_start|>[get_time()]<|tool_call_end|>"
        result = self.detector.detect_and_parse(text, tools_with_noarg)

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_time")

    def test_detect_and_parse_pythonic_multiple_calls(self):
        """Test parsing multiple tool calls in one block."""
        text = '<|tool_call_start|>[get_weather(city="Paris"), search(query="restaurants")]<|tool_call_end|>'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "search")

        params1 = json.loads(result.calls[0].parameters)
        params2 = json.loads(result.calls[1].parameters)
        self.assertEqual(params1["city"], "Paris")
        self.assertEqual(params2["query"], "restaurants")

    def test_detect_and_parse_with_normal_text_before(self):
        """Test parsing with normal text before the tool call."""
        text = 'Let me check the weather for you. <|tool_call_start|>[get_weather(city="Tokyo")]<|tool_call_end|>'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, "Let me check the weather for you.")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")

    def test_detect_and_parse_special_characters_in_value(self):
        """Test parsing with special characters in argument values."""
        text = (
            '<|tool_call_start|>[search(query="what\'s the weather?")]<|tool_call_end|>'
        )
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        params = json.loads(result.calls[0].parameters)
        self.assertIn("weather", params["query"])

    # ==================== detect_and_parse tests (JSON format) ====================

    def test_detect_and_parse_json_simple(self):
        """Test parsing JSON format tool call."""
        text = '<|tool_call_start|>[{"name": "get_weather", "arguments": {"city": "Berlin"}}]<|tool_call_end|>'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")

        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["city"], "Berlin")

    def test_detect_and_parse_json_multiple_calls(self):
        """Test parsing multiple JSON format tool calls."""
        text = '<|tool_call_start|>[{"name": "get_weather", "arguments": {"city": "Paris"}}, {"name": "search", "arguments": {"query": "hotels"}}]<|tool_call_end|>'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "search")

    def test_detect_and_parse_json_with_parameters_key(self):
        """Test parsing JSON format with 'parameters' key instead of 'arguments'."""
        text = '<|tool_call_start|>[{"name": "get_weather", "parameters": {"city": "Madrid"}}]<|tool_call_end|>'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["city"], "Madrid")

    # ==================== Edge cases ====================

    def test_detect_and_parse_no_tool_call(self):
        """Test parsing text with no tool calls."""
        text = "This is just regular text without any tool calls."
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, text)
        self.assertEqual(result.calls, [])

    def test_detect_and_parse_unknown_function(self):
        """Test parsing with unknown function name - skipped by default (SGLANG_FORWARD_UNKNOWN_TOOLS=false)."""
        text = '<|tool_call_start|>[unknown_function(arg="value")]<|tool_call_end|>'
        result = self.detector.detect_and_parse(text, self.tools)

        # By default, unknown functions are skipped (consistent with other detectors)
        self.assertEqual(len(result.calls), 0)

    def test_detect_and_parse_empty_content(self):
        """Test parsing with empty content between markers."""
        text = "<|tool_call_start|><|tool_call_end|>"
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.calls, [])

    def test_detect_and_parse_multiple_blocks(self):
        """Test parsing multiple separate tool call blocks."""
        text = '<|tool_call_start|>[get_weather(city="Paris")]<|tool_call_end|> Some text <|tool_call_start|>[search(query="food")]<|tool_call_end|>'
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "search")

    # ==================== Streaming tests ====================
    # The LFM2 detector buffers until it sees complete <|tool_call_start|>...<|tool_call_end|>
    # blocks, then parses the complete block. This allows proper handling of both
    # JSON and Pythonic formats.

    def test_streaming_json_split_across_chunks(self):
        """Test streaming with JSON tool call split across multiple chunks - waits for complete block."""
        # Reset detector state
        self.detector = Lfm2Detector()

        # First chunk: start marker and partial JSON (no end token)
        chunk1 = '<|tool_call_start|>{"name": "get_weather", "arguments": {"city": '
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)

        # Should buffer and not emit calls yet (waiting for complete block)
        self.assertEqual(len(result1.calls), 0)
        self.assertEqual(result1.normal_text, "")

        # Second chunk: complete the JSON and end token
        chunk2 = '"Vienna"}}<|tool_call_end|>'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)

        # Now should have the complete tool call
        self.assertEqual(len(result2.calls), 1)
        self.assertEqual(result2.calls[0].name, "get_weather")

    def test_streaming_json_normal_text_before_tool_call(self):
        """Test streaming with normal text before JSON tool call."""
        # Reset detector state
        self.detector = Lfm2Detector()

        chunk1 = "I'll check the weather. "
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)

        # Normal text should be returned
        self.assertIn("check the weather", result1.normal_text)

        chunk2 = '<|tool_call_start|>{"name": "get_weather", "arguments": {"city": "Amsterdam"}}<|tool_call_end|>'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)

        self.assertEqual(len(result2.calls), 1)

    def test_streaming_eot_token_filtering(self):
        """Test that end-of-turn token is filtered from normal text."""
        # Reset detector state
        self.detector = Lfm2Detector()

        # Send text that ends with tool call end token (JSON format)
        text = '<|tool_call_start|>{"name": "get_weather", "arguments": {"city": "Oslo"}}<|tool_call_end|>'
        result = self.detector.parse_streaming_increment(text, self.tools)

        # The normal_text should not contain the eot_token
        self.assertNotIn("<|tool_call_end|>", result.normal_text)

    # ==================== Pythonic streaming tests ====================

    def test_streaming_pythonic_split_across_chunks(self):
        """Test streaming with Pythonic tool call split across multiple chunks."""
        self.detector = Lfm2Detector()

        # First chunk: start marker and partial call
        chunk1 = '<|tool_call_start|>[get_weather(city="'
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)

        # Should buffer and not emit calls yet
        self.assertEqual(len(result1.calls), 0)

        # Second chunk: complete the call
        chunk2 = 'Munich")]<|tool_call_end|>'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)

        # Now should have the complete tool call
        self.assertEqual(len(result2.calls), 1)
        self.assertEqual(result2.calls[0].name, "get_weather")
        self.assertEqual(json.loads(result2.calls[0].parameters), {"city": "Munich"})

    def test_streaming_pythonic_multiple_calls(self):
        """Test streaming with multiple Pythonic tool calls."""
        self.detector = Lfm2Detector()

        text = '<|tool_call_start|>[get_weather(city="Paris"), search(query="hotels")]<|tool_call_end|>'
        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "search")

    # ==================== structure_info tests ====================

    def test_supports_structural_tag(self):
        """Test that LFM2 does not support structural tags (Pythonic format)."""
        # LFM2 uses Pythonic format which is not JSON-compatible,
        # so structural_tag constrained generation cannot be used
        self.assertFalse(self.detector.supports_structural_tag())

    def test_structure_info(self):
        """Test structure info for constrained generation."""
        info_func = self.detector.structure_info()
        info = info_func("get_weather")

        self.assertEqual(info.begin, "<|tool_call_start|>[get_weather(")
        self.assertEqual(info.end, ")]<|tool_call_end|>")
        self.assertEqual(info.trigger, "<|tool_call_start|>")


if __name__ == "__main__":
    import unittest

    unittest.main()
