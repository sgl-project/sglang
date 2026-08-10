"""Streaming regression tests for the DeepSeek V3.2 / V4 DSML detectors.

Guards the bug reported in #34214: when a response mixes assistant content with
tool calls, the streaming parser dropped every character of content that was
still buffered when the tool-call opener arrived, so clients saw a truncated
prefix of the text and never received the rest. The one-shot path
(`detect_and_parse`) always emitted that leading text; only streaming diverged.
"""

import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


def _make_tools():
    return [
        Tool(
            type="function",
            function=Function(
                name="get_current_weather",
                description="Get weather information",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            ),
        ),
    ]


def _invoke_block(opener: str, closer: str) -> str:
    return (
        f"{opener}\n"
        '<｜DSML｜invoke name="get_current_weather">\n'
        '<｜DSML｜parameter name="city" string="true">Beijing</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n"
        f"{closer}"
    )


def _feed(detector, chunks, tools):
    """Stream `chunks` through the detector, returning (normal_text, calls)."""
    normal_text_parts = []
    calls = []
    for chunk in chunks:
        result = detector.parse_streaming_increment(chunk, tools)
        if result.normal_text:
            normal_text_parts.append(result.normal_text)
        calls.extend(result.calls)
    return "".join(normal_text_parts), calls


def _tool_call_summary(calls):
    name = next((c.name for c in calls if c.name), None)
    arguments = "".join(c.parameters or "" for c in calls)
    return name, arguments


class TestDeepSeekV4StreamingNormalText(CustomTestCase):
    """V4 spells the section `tool_calls`; this is the shape #34214 reported."""

    OPENER = "<｜DSML｜tool_calls>"
    CLOSER = "</｜DSML｜tool_calls>"

    def _detector(self):
        return DeepSeekV4Detector()

    def test_content_preceding_tool_call_is_streamed(self):
        """Content sharing a chunk with the opener must still reach the client.

        Pre-fix the parser returned `normal_text=""` for every chunk once the
        opener was buffered, so the trailing sentence here vanished while the
        tool call streamed normally.
        """
        content = "Sure, let me look that up. One moment please."
        chunks = [
            "Sure, let me ",
            "look that up. ",
            # Tail of the content arrives in the same chunk as the opener.
            "One moment please.\n\n" + _invoke_block(self.OPENER, self.CLOSER),
        ]

        normal_text, calls = _feed(self._detector(), chunks, _make_tools())

        self.assertEqual(normal_text, content)
        name, arguments = _tool_call_summary(calls)
        self.assertEqual(name, "get_current_weather")
        self.assertIn("Beijing", arguments)

    def test_content_emitted_once_when_opener_splits_across_chunks(self):
        """Buffered content is flushed exactly once, not re-sent or dropped.

        The opener is torn between chunks, so the detector holds the content
        for a turn. Re-emitting it on later chunks would duplicate text in the
        stream; never emitting it is the original bug.
        """
        content = "Checking the forecast now."
        body = _invoke_block(self.OPENER, self.CLOSER)
        split_at = len("<｜DSML｜")
        chunks = [content + body[:split_at], body[split_at:]]

        normal_text, calls = _feed(self._detector(), chunks, _make_tools())

        self.assertEqual(normal_text, content)
        name, _ = _tool_call_summary(calls)
        self.assertEqual(name, "get_current_weather")

    def test_content_preceding_bare_invoke_is_streamed(self):
        """Models may emit `<｜DSML｜invoke` with no enclosing section wrapper.

        `has_tool_call` accepts that shape, so the leading-text split has to
        recognise it too; keying only off `bot_token` would drop the content.
        """
        content = "Here you go."
        chunks = [
            content
            + '\n\n<｜DSML｜invoke name="get_current_weather">\n'
            + '<｜DSML｜parameter name="city" string="true">Beijing</｜DSML｜parameter>\n'
            + "</｜DSML｜invoke>"
        ]

        normal_text, calls = _feed(self._detector(), chunks, _make_tools())

        self.assertEqual(normal_text, content)
        name, _ = _tool_call_summary(calls)
        self.assertEqual(name, "get_current_weather")


class TestDeepSeekV32StreamingNormalText(TestDeepSeekV4StreamingNormalText):
    """V3.2 spells the same section `function_calls`.

    Inherits the V4 cases so the split stays keyed off `bot_token`: hardcoding
    either spelling turns one of the two subclasses red.
    """

    OPENER = "<｜DSML｜function_calls>"
    CLOSER = "</｜DSML｜function_calls>"

    def _detector(self):
        return DeepSeekV32Detector()


if __name__ == "__main__":
    unittest.main()
