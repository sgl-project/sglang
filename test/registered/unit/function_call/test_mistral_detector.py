"""Unit tests for MistralDetector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.mistral_detector import MistralDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "stage-a-test-cpu")


class TestMistralDetector(CustomTestCase):
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
        self.detector = MistralDetector()

    # ==================== has_tool_call Tests ====================

    def test_has_tool_call_json_array_format(self):
        text = (
            '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Beijing"}}]'
        )
        self.assertTrue(self.detector.has_tool_call(text))

    def test_has_tool_call_compact_format(self):
        text = '[TOOL_CALLS]get_weather[ARGS]{"city": "Beijing"}'
        self.assertTrue(self.detector.has_tool_call(text))

    def test_has_tool_call_false(self):
        text = "The weather in Beijing is sunny today."
        self.assertFalse(self.detector.has_tool_call(text))

    # ==================== JSON Array Format Tests ====================

    def test_json_array_single_tool_call(self):
        text = (
            '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Beijing"}}]'
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Beijing")
        self.assertEqual(result.normal_text, "")

    def test_json_array_multiple_tool_calls(self):
        text = '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Beijing"}}, {"name": "search", "arguments": {"query": "restaurants"}}]'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "search")

    def test_json_array_with_leading_text(self):
        text = 'I will check. [TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.normal_text, "I will check.")

    # ==================== Compact Format Tests ====================

    def test_compact_format_single_tool_call(self):
        text = '[TOOL_CALLS]get_weather[ARGS]{"city": "Beijing"}'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Beijing")

    def test_compact_format_with_leading_text(self):
        text = 'Let me help. [TOOL_CALLS]get_weather[ARGS]{"city": "Tokyo"}'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.normal_text, "Let me help.")

    # ==================== No Tool Call Tests ====================

    def test_no_tool_call(self):
        text = "The weather is nice today."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, "The weather is nice today.")

    # ==================== Edge Cases ====================

    def test_tool_call_with_nested_json(self):
        text = '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Beijing", "options": {"detailed": true}}}]'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["options"]["detailed"], True)

    def test_json_array_with_invalid_json(self):
        text = "[TOOL_CALLS] [not valid json]"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, "")

    # ==================== Internal Methods Tests ====================

    def test_extract_json_array(self):
        text = (
            '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Beijing"}}]'
        )
        result = self.detector._extract_json_array(text)
        self.assertIsNotNone(result)
        parsed = json.loads(result)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["name"], "get_weather")

    def test_extract_json_array_nested_brackets(self):
        text = (
            '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"tags": ["a", "b"]}}]'
        )
        result = self.detector._extract_json_array(text)
        self.assertIsNotNone(result)
        parsed = json.loads(result)
        self.assertEqual(parsed[0]["arguments"]["tags"], ["a", "b"])

    def test_extract_json_array_no_marker(self):
        text = "no tool calls here"
        result = self.detector._extract_json_array(text)
        self.assertIsNone(result)

    # ==================== structure_info Tests ====================

    def test_structure_info(self):
        info_func = self.detector.structure_info()
        info = info_func("get_weather")
        self.assertIn("get_weather", info.begin)
        self.assertIn("[TOOL_CALLS]", info.trigger)
        self.assertEqual(info.end, "}]")

    # ==================== Streaming Tests ====================

    def test_streaming_compact_format(self):
        detector = MistralDetector()
        chunks = [
            "[TOOL_",
            "CALLS]get_weather",
            '[ARGS]{"city": "Beijing"}',
        ]
        all_calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)

        func_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(func_calls), 1)
        self.assertEqual(func_calls[0].name, "get_weather")

        full_params = "".join(c.parameters for c in all_calls if c.parameters)
        params = json.loads(full_params)
        self.assertEqual(params["city"], "Beijing")

    def test_streaming_normal_text_before_tool(self):
        detector = MistralDetector()
        result = detector.parse_streaming_increment("Let me check. ", self.tools)
        self.assertEqual(result.normal_text, "Let me check. ")
        self.assertEqual(len(result.calls), 0)

    def test_streaming_text_then_tool_call(self):
        detector = MistralDetector()
        chunks = [
            "Sure! ",
            "[TOOL_CALLS]get_weather",
            '[ARGS]{"city": "Tokyo"}',
        ]
        all_calls = []
        all_normal_text = ""
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)
            all_normal_text += result.normal_text

        self.assertEqual(all_normal_text, "Sure! ")
        func_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(func_calls), 1)
        self.assertEqual(func_calls[0].name, "get_weather")
        full_params = "".join(c.parameters for c in all_calls if c.parameters)
        params = json.loads(full_params)
        self.assertEqual(params["city"], "Tokyo")


if __name__ == "__main__":
    import unittest

    unittest.main()
