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
                            "city": {"type": "string"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
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
                        "properties": {"query": {"type": "string"}},
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
        text = "The weather is sunny today."
        self.assertFalse(self.detector.has_tool_call(text))

    # ==================== detect_and_parse Tests ====================

    def test_single_tool_call(self):
        text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Beijing"}}\n</tool_call>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Beijing")

    def test_multiple_tool_calls(self):
        text = (
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Beijing"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "search", "arguments": {"query": "restaurants"}}\n</tool_call>'
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "search")

    def test_tool_call_with_leading_text(self):
        text = (
            'I will check the weather.\n'
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n</tool_call>'
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.normal_text, "I will check the weather.")

    def test_no_tool_call(self):
        text = "The weather is nice today."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, "The weather is nice today.")

    def test_malformed_json_returns_no_calls(self):
        text = "<tool_call>\nnot valid json\n</tool_call>"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)

    # ==================== Streaming Tests ====================

    def test_streaming_single_tool_call(self):
        detector = Qwen25Detector()
        chunks = [
            '<tool_call>\n',
            '{"name": "get_weather"',
            ', "arguments": {"city": "Beijing"',
            "}}\n</tool_call>",
        ]
        all_calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)

        func_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(func_calls), 1)
        self.assertEqual(func_calls[0].name, "get_weather")

    def test_streaming_normal_text_first(self):
        detector = Qwen25Detector()
        result = detector.parse_streaming_increment("Hello! ", self.tools)
        self.assertEqual(result.normal_text, "Hello! ")

    def test_streaming_text_then_tool_call(self):
        detector = Qwen25Detector()
        chunks = [
            "Sure, let me check. ",
            '<tool_call>\n',
            '{"name": "get_weather"',
            ', "arguments": {"city": "Paris"',
            "}}\n</tool_call>",
        ]
        all_normal_text = ""
        all_calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_normal_text += result.normal_text
            all_calls.extend(result.calls)

        self.assertIn("Sure, let me check.", all_normal_text)
        func_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(func_calls), 1)
        self.assertEqual(func_calls[0].name, "get_weather")


if __name__ == "__main__":
    import unittest

    unittest.main()
