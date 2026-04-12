"""Unit tests for Qwen25Detector — no server, no model loading."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.qwen25_detector import Qwen25Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "stage-a-test-cpu")


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
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="get_time",
                    description="Get current time",
                    parameters={
                        "type": "object",
                        "properties": {"timezone": {"type": "string"}},
                        "required": ["timezone"],
                    },
                ),
            ),
        ]
        self.detector = Qwen25Detector()

    # ==================== has_tool_call ====================

    def test_has_tool_call_true(self):
        text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
        self.assertTrue(self.detector.has_tool_call(text))

    def test_has_tool_call_false(self):
        self.assertFalse(self.detector.has_tool_call("The weather is nice today."))

    def test_has_tool_call_requires_newline(self):
        """<tool_call> without the trailing newline does not match bot_token."""
        self.assertFalse(self.detector.has_tool_call("<tool_call>"))

    # ==================== detect_and_parse ====================

    def test_single_tool_call(self):
        text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[0].tool_index, 0)
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["city"], "Paris")

    def test_multiple_tool_calls(self):
        text = (
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
            '<tool_call>\n{"name": "get_time", "arguments": {"timezone": "UTC"}}\n</tool_call>'
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "get_time")
        self.assertEqual(result.calls[1].tool_index, 1)

    def test_no_tool_call(self):
        text = "The weather in Paris is sunny today."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, text)

    def test_normal_text_before_call(self):
        text = 'Sure!\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertIn("Sure", result.normal_text)

    def test_invalid_json_skipped(self):
        """A malformed JSON block inside <tool_call> is skipped with a warning."""
        text = "<tool_call>\nnot valid json\n</tool_call>"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)

    def test_unknown_tool_skipped(self):
        text = '<tool_call>\n{"name": "undefined_func", "arguments": {}}\n</tool_call>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)

    def test_parameters_key_accepted(self):
        """parse_base_json accepts 'parameters' in addition to 'arguments'."""
        text = '<tool_call>\n{"name": "get_weather", "parameters": {"city": "Berlin"}}\n</tool_call>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["city"], "Berlin")

    # ==================== structure_info ====================

    def test_structure_info(self):
        get_info = self.detector.structure_info()
        info = get_info("get_weather")
        self.assertIn("<tool_call>", info.begin)
        self.assertIn("get_weather", info.begin)
        self.assertIn("</tool_call>", info.end)
        self.assertEqual(info.trigger, "<tool_call>")

    def test_structure_info_newlines(self):
        """Qwen25 format uses newlines around the JSON body."""
        get_info = self.detector.structure_info()
        info = get_info("get_weather")
        self.assertTrue(info.begin.startswith("<tool_call>\n"))
        self.assertTrue(info.end.endswith("</tool_call>"))

    # ==================== Streaming ====================

    def test_streaming_plain_text(self):
        """Text with no tool markers is returned immediately as normal_text."""
        result = self.detector.parse_streaming_increment("Hello, world!", self.tools)
        self.assertEqual(result.normal_text, "Hello, world!")
        self.assertEqual(result.calls, [])

    def test_streaming_partial_bot_token_buffered(self):
        """A chunk ending with a partial <tool_call>\n is held in the buffer."""
        result = self.detector.parse_streaming_increment("<tool_ca", self.tools)
        self.assertEqual(result.normal_text, "")
        self.assertEqual(result.calls, [])

    def test_streaming_eot_stripped_from_normal_text(self):
        """</tool_call> (without leading newline) appearing in normal text is stripped."""
        r1 = self.detector.parse_streaming_increment("After call: ", self.tools)
        self.assertEqual(r1.normal_text, "After call: ")
        r2 = self.detector.parse_streaming_increment("</tool_call>", self.tools)
        self.assertNotIn("</tool_call>", r2.normal_text)

    def test_streaming_partial_eot_held(self):
        """A partial </tool_call> at the end of normal text is buffered."""
        result = self.detector.parse_streaming_increment("</tool_ca", self.tools)
        # Partial eot must not appear in output
        self.assertNotIn("</tool_call>", result.normal_text)


if __name__ == "__main__":
    unittest.main()
