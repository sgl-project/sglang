"""Unit tests for DeepSeekV3Detector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv3_detector import DeepSeekV3Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


def _make_call(name: str, args_json: str) -> str:
    """Build one DeepSeekV3 tool call segment."""
    return (
        "<｜tool▁call▁begin｜>function<｜tool▁sep｜>"
        + name
        + "\n```json\n"
        + args_json
        + "\n```<｜tool▁call▁end｜>"
    )


def _wrap(*calls: str) -> str:
    """Wrap one or more call segments in the begin/end tokens."""
    return "<｜tool▁calls▁begin｜>" + "".join(calls) + "<｜tool▁calls▁end｜>"


class TestDeepSeekV3Detector(CustomTestCase):
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
        self.detector = DeepSeekV3Detector()

    # ==================== has_tool_call Tests ====================

    def test_has_tool_call_true(self):
        text = _wrap(_make_call("get_weather", '{"city": "Beijing"}'))
        self.assertTrue(self.detector.has_tool_call(text))

    def test_has_tool_call_false(self):
        text = "The weather in Beijing is sunny today."
        self.assertFalse(self.detector.has_tool_call(text))

    # ==================== detect_and_parse Tests ====================

    def test_single_tool_call(self):
        text = _wrap(_make_call("get_weather", '{"city": "Beijing"}'))
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Beijing")
        self.assertEqual(result.normal_text, "")

    def test_multiple_tool_calls(self):
        text = _wrap(
            _make_call("get_weather", '{"city": "Beijing"}'),
            _make_call("search", '{"query": "restaurants"}'),
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "search")

    def test_tool_call_with_leading_text(self):
        text = "I will check the weather for you. " + _wrap(
            _make_call("get_weather", '{"city": "Tokyo"}')
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.normal_text, "I will check the weather for you.")

    def test_no_tool_call(self):
        text = "The weather is nice today."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, "The weather is nice today.")

    def test_tool_call_with_multiple_arguments(self):
        text = _wrap(_make_call("get_weather", '{"city": "London", "unit": "celsius"}'))
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "London")
        self.assertEqual(args["unit"], "celsius")

    # ==================== structure_info Tests ====================

    def test_structure_info(self):
        info_func = self.detector.structure_info()
        info = info_func("get_weather")
        self.assertEqual(info.trigger, "<｜tool▁calls▁begin｜>")
        self.assertIn("get_weather", info.begin)
        self.assertIn("```json", info.begin)
        self.assertIn("<｜tool▁call▁end｜>", info.end)

    # ==================== Streaming Tests ====================

    def test_streaming_single_tool_call(self):
        detector = DeepSeekV3Detector()
        chunks = [
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n",
            '{"city": "Beijing"}\n```',
            "<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
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
        detector = DeepSeekV3Detector()
        result = detector.parse_streaming_increment("Hello! Let me help. ", self.tools)
        self.assertEqual(result.normal_text, "Hello! Let me help. ")
        self.assertEqual(len(result.calls), 0)

    def test_streaming_text_then_tool_call(self):
        detector = DeepSeekV3Detector()
        chunks = [
            "Sure, let me check. ",
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n",
            '{"city": "Tokyo"}\n```',
            "<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
        ]
        all_calls = []
        all_normal_text = ""
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)
            all_normal_text += result.normal_text

        self.assertEqual(all_normal_text, "Sure, let me check. ")
        func_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(func_calls), 1)
        self.assertEqual(func_calls[0].name, "get_weather")
        full_params = "".join(c.parameters for c in all_calls if c.parameters)
        params = json.loads(full_params)
        self.assertEqual(params["city"], "Tokyo")


if __name__ == "__main__":
    import unittest

    unittest.main()
