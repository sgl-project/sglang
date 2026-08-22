"""Unit tests for Qwen25Detector — no server, no model loading."""

import json
import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.qwen25_detector import Qwen25Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


def _make_tools():
    return [
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
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            ),
        ),
    ]


class TestQwen25DetectorHasToolCall(CustomTestCase):
    def setUp(self):
        self.detector = Qwen25Detector()

    def test_has_tool_call_true(self):
        text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Beijing"}}\n</tool_call>'
        self.assertTrue(self.detector.has_tool_call(text))

    def test_has_tool_call_false_plain_text(self):
        self.assertFalse(self.detector.has_tool_call("The weather in Beijing is sunny."))

    def test_has_tool_call_false_only_closing_tag(self):
        self.assertFalse(self.detector.has_tool_call("\n</tool_call>"))


class TestQwen25DetectorDetectAndParse(CustomTestCase):
    def setUp(self):
        self.tools = _make_tools()
        self.detector = Qwen25Detector()

    def test_single_tool_call(self):
        text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Beijing"}}\n</tool_call>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Beijing")
        self.assertEqual(result.normal_text, "")

    def test_single_tool_call_with_multiple_args(self):
        text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "London", "unit": "celsius"}}\n</tool_call>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "London")
        self.assertEqual(args["unit"], "celsius")

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
        text = 'I will check the weather.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n</tool_call>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertIn("I will check the weather", result.normal_text)

    def test_no_tool_call(self):
        text = "The weather is nice today."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, text)

    def test_malformed_json_skipped(self):
        text = "<tool_call>\nnot valid json\n</tool_call>"
        result = self.detector.detect_and_parse(text, self.tools)
        # Malformed JSON is logged and skipped; no calls returned
        self.assertEqual(len(result.calls), 0)


class TestQwen25DetectorStructureInfo(CustomTestCase):
    def setUp(self):
        self.detector = Qwen25Detector()

    def test_structure_info_contains_name(self):
        info_func = self.detector.structure_info()
        info = info_func("get_weather")
        self.assertIn("get_weather", info.begin)

    def test_structure_info_trigger(self):
        info_func = self.detector.structure_info()
        info = info_func("get_weather")
        self.assertEqual(info.trigger, "<tool_call>")

    def test_structure_info_end(self):
        info_func = self.detector.structure_info()
        info = info_func("get_weather")
        self.assertIn("</tool_call>", info.end)


class TestQwen25DetectorStreaming(CustomTestCase):
    def setUp(self):
        self.tools = _make_tools()

    def test_streaming_single_tool_call(self):
        detector = Qwen25Detector()
        chunks = [
            "<tool_call>\n",
            '{"name": "get_weather",',
            ' "arguments": {"city": "Beijing"',
            "}}\n</tool_call>",
        ]
        all_calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)

        named = [c for c in all_calls if c.name]
        self.assertEqual(len(named), 1)
        self.assertEqual(named[0].name, "get_weather")

        full_params = "".join(c.parameters for c in all_calls if c.parameters)
        params = json.loads(full_params)
        self.assertEqual(params["city"], "Beijing")

    def test_streaming_normal_text_before_tool(self):
        detector = Qwen25Detector()
        result = detector.parse_streaming_increment("Hello! Let me help. ", self.tools)
        self.assertEqual(result.normal_text, "Hello! Let me help. ")
        self.assertEqual(len(result.calls), 0)

    def test_streaming_text_then_tool_call(self):
        detector = Qwen25Detector()
        chunks = [
            "Sure, let me check. ",
            '<tool_call>\n{"name": "get_weather",',
            ' "arguments": {"city": "Tokyo"',
            "}}\n</tool_call>",
        ]
        all_calls = []
        all_normal_text = ""
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)
            all_normal_text += result.normal_text

        self.assertIn("Sure, let me check.", all_normal_text)
        named = [c for c in all_calls if c.name]
        self.assertEqual(len(named), 1)
        self.assertEqual(named[0].name, "get_weather")
        full_params = "".join(c.parameters for c in all_calls if c.parameters)
        params = json.loads(full_params)
        self.assertEqual(params["city"], "Tokyo")

    def test_streaming_multiple_tool_calls(self):
        detector = Qwen25Detector()
        full_text = (
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Beijing"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "search", "arguments": {"query": "food"}}\n</tool_call>'
        )
        all_calls = []
        # Feed in character-by-character to stress the buffering logic
        for char in full_text:
            result = detector.parse_streaming_increment(char, self.tools)
            all_calls.extend(result.calls)

        named = [c for c in all_calls if c.name]
        names = [c.name for c in named]
        self.assertIn("get_weather", names)
        self.assertIn("search", names)

        # Reconstruct and verify parameters for each tool call by tool_index
        params_by_index = {}
        for c in all_calls:
            if c.parameters:
                params_by_index[c.tool_index] = (
                    params_by_index.get(c.tool_index, "") + c.parameters
                )

        self.assertEqual(json.loads(params_by_index[0]), {"city": "Beijing"})
        self.assertEqual(json.loads(params_by_index[1]), {"query": "food"})


if __name__ == "__main__":
    import unittest

    unittest.main()