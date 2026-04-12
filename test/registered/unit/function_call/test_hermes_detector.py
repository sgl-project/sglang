"""Unit tests for HermesDetector — no server, no model loading."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.hermes_detector import HermesDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "stage-a-test-cpu")


class TestHermesDetector(CustomTestCase):
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
        self.detector = HermesDetector()

    # ==================== has_tool_call ====================

    def test_has_tool_call_true(self):
        self.assertTrue(
            self.detector.has_tool_call(
                '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
            )
        )

    def test_has_tool_call_false(self):
        self.assertFalse(self.detector.has_tool_call("The weather is nice today."))

    def test_has_tool_call_partial_tag(self):
        self.assertFalse(self.detector.has_tool_call("<tool_cal"))

    # ==================== detect_and_parse ====================

    def test_single_tool_call(self):
        text = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[0].tool_index, 0)
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["city"], "Paris")

    def test_multiple_tool_calls(self):
        text = (
            '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
            '<tool_call>{"name": "get_time", "arguments": {"timezone": "UTC"}}</tool_call>'
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
        text = 'Sure!<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertIn("Sure", result.normal_text)

    def test_invalid_json_fallback(self):
        text = "<tool_call>not valid json</tool_call>"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, text)

    def test_unknown_tool_skipped(self):
        text = '<tool_call>{"name": "undefined_func", "arguments": {}}</tool_call>'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)

    def test_parameters_key_accepted(self):
        """parse_base_json accepts 'parameters' in addition to 'arguments'."""
        text = '<tool_call>{"name": "get_weather", "parameters": {"city": "Berlin"}}</tool_call>'
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

    # ==================== Streaming ====================

    def test_streaming_plain_text(self):
        """Text with no tool markers is returned immediately as normal_text."""
        result = self.detector.parse_streaming_increment("Hello, world!", self.tools)
        self.assertEqual(result.normal_text, "Hello, world!")
        self.assertEqual(result.calls, [])

    def test_streaming_partial_bot_token_buffered(self):
        """A chunk ending with a partial <tool_call> is held in the buffer."""
        result = self.detector.parse_streaming_increment("<tool_ca", self.tools)
        self.assertEqual(result.normal_text, "")
        self.assertEqual(result.calls, [])

    def test_streaming_normal_text_before_bot_token(self):
        """Text before <tool_call> is emitted as normal_text; bot_token stays buffered."""
        result = self.detector.parse_streaming_increment("Sure! <tool_call>", self.tools)
        self.assertIn("Sure!", result.normal_text)

    def test_streaming_eot_stripped_from_normal_text(self):
        """</tool_call> appearing as normal text is stripped by _clean_normal_text."""
        r1 = self.detector.parse_streaming_increment("After call: ", self.tools)
        self.assertEqual(r1.normal_text, "After call: ")
        r2 = self.detector.parse_streaming_increment("</tool_call>", self.tools)
        self.assertNotIn("</tool_call>", r2.normal_text)

    def test_streaming_partial_eot_buffered(self):
        """A partial </tool_call> at the end of normal text is held until complete."""
        result = self.detector.parse_streaming_increment("</tool_ca", self.tools)
        self.assertNotIn("</tool_call>", result.normal_text)
        # Partial was held — nothing output yet
        self.assertEqual(result.normal_text, "")


if __name__ == "__main__":
    unittest.main()
