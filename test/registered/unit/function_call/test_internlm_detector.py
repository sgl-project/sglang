"""Unit tests for InternlmDetector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.internlm_detector import InternlmDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestInternlmDetector(CustomTestCase):
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
        self.detector = InternlmDetector()

    # ==================== has_tool_call Tests ====================

    def test_has_tool_call_true(self):
        text = '<|action_start|> <|plugin|>\n{"name": "get_weather", "parameters": {"city": "Beijing"}}<|action_end|>'
        self.assertTrue(self.detector.has_tool_call(text))

    def test_has_tool_call_false(self):
        text = "The weather in Beijing is sunny today."
        self.assertFalse(self.detector.has_tool_call(text))

    # ==================== detect_and_parse Tests ====================

    def test_single_tool_call(self):
        text = '<|action_start|> <|plugin|>\n{"name": "get_weather", "parameters": {"city": "Beijing"}}<|action_end|>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Beijing")

    def test_tool_call_with_arguments_key(self):
        """InternLM supports both 'parameters' and 'arguments' keys."""
        text = '<|action_start|> <|plugin|>\n{"name": "search", "arguments": {"query": "python tutorials"}}<|action_end|>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "search")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["query"], "python tutorials")

    def test_multiple_tool_calls(self):
        text = (
            '<|action_start|> <|plugin|>\n{"name": "get_weather", "parameters": {"city": "Beijing"}}<|action_end|>'
            '<|action_start|> <|plugin|>\n{"name": "search", "parameters": {"query": "restaurants"}}<|action_end|>'
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "search")

    def test_tool_call_with_preceding_text(self):
        text = 'Let me check the weather for you.<|action_start|> <|plugin|>\n{"name": "get_weather", "parameters": {"city": "Shanghai"}}<|action_end|>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertIn("check the weather", result.normal_text)

    def test_no_tool_call(self):
        text = "Hello, how can I help you today?"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, text)

    def test_malformed_json(self):
        """Malformed JSON should be handled gracefully."""
        text = "<|action_start|> <|plugin|>\n{not valid json}<|action_end|>"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
