"""Unit tests for PythonicDetector (Llama-4 format) — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.pythonic_detector import PythonicDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestPythonicDetector(CustomTestCase):
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
        self.detector = PythonicDetector()

    # ==================== has_tool_call Tests ====================

    def test_has_tool_call_true(self):
        text = '[get_weather(city="Beijing")]'
        self.assertTrue(self.detector.has_tool_call(text))

    def test_has_tool_call_false(self):
        text = "The weather in Beijing is sunny today."
        self.assertFalse(self.detector.has_tool_call(text))

    def test_has_tool_call_with_python_tags(self):
        """Llama-4 sometimes emits <|python_start|> / <|python_end|> tokens."""
        text = '<|python_start|>[get_weather(city="Tokyo")]<|python_end|>'
        self.assertTrue(self.detector.has_tool_call(text))

    # ==================== detect_and_parse Tests ====================

    def test_single_tool_call(self):
        text = '[get_weather(city="Beijing")]'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Beijing")

    def test_single_tool_call_multiple_args(self):
        text = '[get_weather(city="Beijing", unit="celsius")]'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Beijing")
        self.assertEqual(args["unit"], "celsius")

    def test_multiple_tool_calls(self):
        text = '[get_weather(city="Beijing"), search(query="restaurants")]'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "search")

    def test_tool_call_with_preceding_text(self):
        """Normal text before the tool call should be captured."""
        text = 'Let me check that. [get_weather(city="Shanghai")]'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertIn("check that", result.normal_text)

    def test_no_tool_call(self):
        text = "Hello, how can I help you today?"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, text)

    def test_python_tags_stripped(self):
        """<|python_start|> and <|python_end|> tokens should be removed."""
        text = '<|python_start|>[get_weather(city="Tokyo")]<|python_end|>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")

    def test_invalid_syntax(self):
        """Invalid Python syntax should not crash."""
        text = "[get_weather(city=)]"
        result = self.detector.detect_and_parse(text, self.tools)
        # Should gracefully handle parse error
        self.assertIsNotNone(result)
