"""Unit tests for CohereCommand4Detector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.cohere_command4_detector import CohereCommand4Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestCohereCommand4Detector(CustomTestCase):
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
                        },
                        "required": ["city"],
                    },
                ),
            ),
        ]
        self.detector = CohereCommand4Detector()

    # ==================== has_tool_call Tests ====================

    def test_has_tool_call_true(self):
        text = '<|START_ACTION|>[{"tool_name": "get_weather", "parameters": {"city": "Beijing"}}]<|END_ACTION|>'
        self.assertTrue(self.detector.has_tool_call(text))

    def test_has_tool_call_false(self):
        text = "The weather is nice today."
        self.assertFalse(self.detector.has_tool_call(text))

    # ==================== detect_and_parse Tests ====================

    def test_single_tool_call(self):
        text = '<|START_ACTION|>[{"tool_name": "get_weather", "parameters": {"city": "Beijing"}}]<|END_ACTION|>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Beijing")

    def test_multiple_tool_calls(self):
        text = (
            '<|START_ACTION|>['
            '{"tool_name": "get_weather", "parameters": {"city": "Beijing"}},'
            '{"tool_name": "search", "tool_call_id": "id1", "parameters": {"query": "restaurants"}}'
            ']<|END_ACTION|>'
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "search")

    def test_tool_call_with_leading_text(self):
        text = (
            'I will check the weather. '
            '<|START_ACTION|>[{"tool_name": "get_weather", "parameters": {"city": "Tokyo"}}]<|END_ACTION|>'
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.normal_text, "I will check the weather. ")

    def test_no_tool_call(self):
        text = "The weather is nice today."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, "The weather is nice today.")

    def test_malformed_json_returns_no_calls(self):
        text = "<|START_ACTION|>not valid json<|END_ACTION|>"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)


if __name__ == "__main__":
    import unittest

    unittest.main()
