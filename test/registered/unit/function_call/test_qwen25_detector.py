"""Unit tests for Qwen25Detector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.qwen25_detector import Qwen25Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


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
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                        },
                        "required": ["city"],
                    },
                ),
            ),
        ]
        self.detector = Qwen25Detector()

    def test_single_tool_call(self):
        text = (
            "<tool_call>\n"
            '{"name": "get_weather", "arguments": {"city": "Beijing"}}'
            "\n</tool_call>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "Beijing"})

    # ==================== truncated tool call Tests ====================
    # A tool call cut off by max_tokens must not leak raw <tool_call>
    # markup into content: the streaming path drops the markup, and
    # finish_reason ("length") already tells the client the call is
    # incomplete, so non-streaming must drop it too.

    def test_bare_opener_at_end_dropped_from_content(self):
        # Cut off right at the opener, before the "\n" of the full
        # bot_token ("<tool_call>\n") was generated.
        text = "I will check.\n<tool_call>"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, "I will check.")
        self.assertEqual(len(result.calls), 0)

    def test_truncated_arguments_dropped_from_content(self):
        text = (
            "I will check.\n<tool_call>\n"
            '{"name": "get_weather", "arguments": {"city": "San Fr'
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, "I will check.")
        self.assertEqual(len(result.calls), 0)

    def test_opener_inside_content_not_stripped(self):
        # A bare opener only counts as truncated markup at the very end
        # of the output; elsewhere the text is returned unchanged.
        text = "The literal tag <tool_call> is mentioned here."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, text)
        self.assertEqual(len(result.calls), 0)


if __name__ == "__main__":
    import unittest

    unittest.main()
