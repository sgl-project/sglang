"""Unit tests for Lfm2Detector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.lfm2_detector import Lfm2Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "stage-a-test-cpu")

BOT = "<|tool_call_start|>"
EOT = "<|tool_call_end|>"


def _pythonic(content: str) -> str:
    return BOT + content + EOT


def _json_call(name: str, args: dict) -> str:
    return BOT + json.dumps([{"name": name, "arguments": args}]) + EOT


class TestLfm2Detector(CustomTestCase):
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
        self.detector = Lfm2Detector()

    def test_has_tool_call_true(self):
        self.assertTrue(
            self.detector.has_tool_call(BOT + '[get_weather(city="Tokyo")]' + EOT)
        )

    def test_has_tool_call_false(self):
        self.assertFalse(self.detector.has_tool_call("The weather is fine today."))

    def test_single_call_pythonic(self):
        text = _pythonic('[get_weather(city="Tokyo")]')
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Tokyo")

    def test_single_call_json(self):
        text = _json_call("search", {"query": "sglang"})
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "search")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["query"], "sglang")

    def test_multiple_calls_pythonic(self):
        text = _pythonic('[get_weather(city="Tokyo"), search(query="restaurants")]')
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "search")

    def test_multiple_args_pythonic(self):
        text = _pythonic('[get_weather(city="London", unit="celsius")]')
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

    def test_normal_text_before_call(self):
        text = "Sure! " + _pythonic('[get_weather(city="Paris")]')
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertIn("Sure", result.normal_text)

    def test_streaming_plain_text(self):
        detector = Lfm2Detector()
        result = detector.parse_streaming_increment("Hello world.", self.tools)
        self.assertEqual(result.normal_text, "Hello world.")
        self.assertEqual(len(result.calls), 0)

    def test_streaming_complete_call_pythonic(self):
        detector = Lfm2Detector()
        text = _pythonic('[get_weather(city="Tokyo")]')
        result = detector.parse_streaming_increment(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")

    def test_streaming_split_chunks(self):
        detector = Lfm2Detector()
        chunks = [BOT, '[get_weather(city="Oslo")]', EOT]
        all_calls = []
        for chunk in chunks:
            r = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(r.calls)
        named = [c for c in all_calls if c.name]
        self.assertEqual(len(named), 1)
        self.assertEqual(named[0].name, "get_weather")

    def test_streaming_json_format(self):
        detector = Lfm2Detector()
        text = _json_call("search", {"query": "test"})
        result = detector.parse_streaming_increment(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "search")

    def test_supports_structural_tag_false(self):
        self.assertFalse(self.detector.supports_structural_tag())

    def test_structure_info_not_none(self):
        info_fn = self.detector.structure_info()
        self.assertIsNotNone(info_fn)
        info = info_fn("get_weather")
        self.assertIn("get_weather", info.begin)
        self.assertIn(BOT, info.trigger)


if __name__ == "__main__":
    import unittest

    unittest.main()
