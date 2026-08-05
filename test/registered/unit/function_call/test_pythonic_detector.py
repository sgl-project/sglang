"""Unit tests for PythonicDetector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.pythonic_detector import PythonicDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "stage-a-test-cpu")

PYTHON_START = "<|python_start|>"
PYTHON_END = "<|python_end|>"


class TestPythonicDetector(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
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
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                ),
            ),
        ]
        self.detector = PythonicDetector()

    def test_has_tool_call_true(self):
        self.assertTrue(self.detector.has_tool_call('[get_weather(city="Tokyo")]'))

    def test_has_tool_call_false(self):
        self.assertFalse(self.detector.has_tool_call("The weather is fine today."))

    def test_has_tool_call_with_python_tokens(self):
        text = PYTHON_START + '[get_weather(city="Tokyo")]' + PYTHON_END
        self.assertTrue(self.detector.has_tool_call(text))

    def test_single_call(self):
        text = '[get_weather(city="Tokyo")]'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Tokyo")

    def test_multiple_calls(self):
        text = '[get_weather(city="Tokyo"), search(query="restaurants")]'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "search")

    def test_multiple_args(self):
        text = '[get_weather(city="London", unit="celsius")]'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "London")
        self.assertEqual(args["unit"], "celsius")

    def test_no_tool_call(self):
        text = "The weather is fine today."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertIn("weather", result.normal_text)

    def test_strips_python_tokens(self):
        text = PYTHON_START + '[get_weather(city="Paris")]' + PYTHON_END
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")

    def test_unknown_tool_skipped(self):
        text = '[undefined_func(arg="val")]'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)

    def test_streaming_plain_text(self):
        detector = PythonicDetector()
        result = detector.parse_streaming_increment("Hello world.", self.tools)
        self.assertEqual(result.normal_text, "Hello world.")
        self.assertEqual(len(result.calls), 0)

    def test_streaming_complete_call_one_chunk(self):
        detector = PythonicDetector()
        result = detector.parse_streaming_increment(
            '[get_weather(city="Tokyo")]', self.tools
        )
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")

    def test_streaming_split_across_chunks(self):
        detector = PythonicDetector()
        all_calls = []
        for chunk in ['[get_weather(city="', 'Oslo")]']:
            r = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(r.calls)
        named = [c for c in all_calls if c.name]
        self.assertEqual(len(named), 1)
        self.assertEqual(named[0].name, "get_weather")
        args = json.loads(named[0].parameters)
        self.assertEqual(args["city"], "Oslo")

    def test_supports_structural_tag_false(self):
        self.assertFalse(self.detector.supports_structural_tag())

    def test_structure_info_raises(self):
        with self.assertRaises(NotImplementedError):
            self.detector.structure_info()


if __name__ == "__main__":
    import unittest

    unittest.main()
