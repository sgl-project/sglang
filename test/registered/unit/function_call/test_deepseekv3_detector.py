"""Unit tests for DeepSeekV3Detector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv3_detector import DeepSeekV3Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


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
        text = '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"city": "Beijing"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
        self.assertTrue(self.detector.has_tool_call(text))

    def test_has_tool_call_false(self):
        text = "The weather in Beijing is sunny today."
        self.assertFalse(self.detector.has_tool_call(text))

    # ==================== detect_and_parse Tests ====================

    def test_single_tool_call(self):
        text = '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"city": "Beijing"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Beijing")

    def test_multiple_tool_calls(self):
        text = (
            "<｜tool▁calls▁begin｜>"
            '<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"city": "Beijing"}\n```<｜tool▁call▁end｜>\n'
            '<｜tool▁call▁begin｜>function<｜tool▁sep｜>search\n```json\n{"query": "restaurants"}\n```<｜tool▁call▁end｜>'
            "<｜tool▁calls▁end｜>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "search")

    def test_tool_call_with_preceding_text(self):
        text = (
            "Let me check that for you."
            '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"city": "Tokyo"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertIn("check that", result.normal_text)

    def test_no_tool_call(self):
        text = "Hello, how can I help you today?"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, text)

    def test_malformed_json(self):
        """Malformed JSON in arguments should be handled gracefully."""
        text = '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{invalid}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
        result = self.detector.detect_and_parse(text, self.tools)
        # Should not crash; returns empty calls or the raw text
        self.assertIsNotNone(result)
