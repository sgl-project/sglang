"""Unit tests for DeepSeekV3Detector / DeepSeekV31Detector — no server, no model loading."""

import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv3_detector import DeepSeekV3Detector
from sglang.srt.function_call.deepseekv31_detector import DeepSeekV31Detector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1.0, "base-a-test-cpu")


def _tools():
    return [
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


class TestDeepSeekV31StreamingPartialBotToken(unittest.TestCase):
    """bot_token/tool_call_begin are multi-codepoint markers, not single vocab
    tokens, so they routinely arrive split across streaming increments. The
    detector must buffer a possible partial prefix instead of flushing it as
    normal_text and wiping the buffer.
    """

    def setUp(self):
        self.tools = _tools()

    def test_bot_token_split_across_increments_is_not_leaked(self):
        detector = DeepSeekV31Detector()
        bot = "<｜tool▁calls▁begin｜>"
        rest = (
            '<｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"city": "Tokyo"}'
            "<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
        )
        # Split the bot_token itself across three increments, matching how a
        # real tokenizer emits this marker piecewise.
        chunks = [bot[0:4], bot[4:9], bot[9:], rest]

        all_normal = ""
        all_calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_normal += result.normal_text
            all_calls.extend(result.calls)

        self.assertEqual(all_normal, "")
        self.assertEqual(len(all_calls), 1)
        self.assertEqual(all_calls[0].name, "get_weather")

    def test_plain_text_without_any_tool_call_still_flushes(self):
        detector = DeepSeekV31Detector()
        result = detector.parse_streaming_increment(
            "Hello, how can I help you today?", self.tools
        )
        self.assertEqual(result.normal_text, "Hello, how can I help you today?")
        self.assertEqual(result.calls, [])


class TestDeepSeekV3StreamingPartialBotToken(unittest.TestCase):
    def setUp(self):
        self.tools = _tools()

    def test_bot_token_split_across_increments_is_not_leaked(self):
        detector = DeepSeekV3Detector()
        bot = "<｜tool▁calls▁begin｜>"
        rest = (
            "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
            '```json\n{"city": "Tokyo"}\n```'
            "<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
        )
        chunks = [bot[0:4], bot[4:9], bot[9:], rest]

        all_normal = ""
        all_calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_normal += result.normal_text
            all_calls.extend(result.calls)

        self.assertEqual(all_normal, "")
        self.assertEqual(len(all_calls), 1)
        self.assertEqual(all_calls[0].name, "get_weather")

    def test_plain_text_without_any_tool_call_still_flushes(self):
        detector = DeepSeekV3Detector()
        result = detector.parse_streaming_increment(
            "Hello, how can I help you today?", self.tools
        )
        self.assertEqual(result.normal_text, "Hello, how can I help you today?")
        self.assertEqual(result.calls, [])


if __name__ == "__main__":
    unittest.main()
