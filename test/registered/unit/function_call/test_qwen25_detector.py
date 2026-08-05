"""Unit tests for Qwen25Detector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.qwen25_detector import Qwen25Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestQwen25Detector(CustomTestCase):
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
                            "city": {"type": "string", "description": "City name"},
                            "unit": {
                                "type": "string",
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
                    description="Search the web",
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
        ]
        self.detector = Qwen25Detector()

    # ==================== has_tool_call Tests ====================

    def test_has_tool_call_true(self):
        text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Beijing"}}\n</tool_call>'
        self.assertTrue(self.detector.has_tool_call(text))

    def test_has_tool_call_false(self):
        text = "The weather in Beijing is sunny today."
        self.assertFalse(self.detector.has_tool_call(text))

    def test_has_tool_call_partial_tag(self):
        """Partial tag should not be detected as a tool call."""
        text = "<tool_call>no newline so not matching bot_token"
        self.assertFalse(self.detector.has_tool_call(text))

    # ==================== detect_and_parse Tests ====================

    def test_single_tool_call(self):
        text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Beijing"}}\n</tool_call>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Beijing")
        self.assertEqual(result.normal_text, "")

    def test_multiple_tool_calls(self):
        text = (
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Beijing"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "search", "arguments": {"query": "restaurants"}}\n</tool_call>'
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "search")

    def test_tool_call_with_preceding_text(self):
        """Normal text before the tool call should be captured."""
        text = 'Let me check the weather.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Shanghai"}}\n</tool_call>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertIn("check the weather", result.normal_text)

    def test_no_tool_call(self):
        """Plain text without tool call tags returns empty calls."""
        text = "Hello, how can I help you today?"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, text)

    def test_malformed_json(self):
        """Malformed JSON inside tool_call tags should be skipped gracefully."""
        text = "<tool_call>\n{invalid json}\n</tool_call>"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)

    def test_tool_call_with_extra_whitespace(self):
        """Extra whitespace inside the JSON block should still parse."""
        text = '<tool_call>\n  {"name": "get_weather", "arguments": {"city": "Tokyo"}}  \n</tool_call>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Tokyo")

    # ==================== Streaming Tests ====================

    def test_streaming_single_call(self):
        """Streaming parse should accumulate and detect complete tool call."""
        chunks = [
            "<tool_call>\n",
            '{"name": "get_weather",',
            ' "arguments": {"city": "Beijing"}}',
            "\n</tool_call>",
        ]
        all_calls = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            if result.calls:
                all_calls.extend(result.calls)

        # Across all chunks, we should have accumulated one parsed call
        self.assertEqual(len(all_calls), 1)
        self.assertEqual(all_calls[0].name, "get_weather")

    def test_streaming_normal_text_before_call(self):
        """Normal text streamed before a tool call is captured."""
        # Reset detector state for clean test
        detector = Qwen25Detector()
        result = detector.parse_streaming_increment("Hello there! ", self.tools)
        self.assertIn("Hello", result.normal_text)
        self.assertEqual(len(result.calls), 0)
