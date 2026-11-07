"""Unit tests: KimiK2Detector finish() flushes buffered text at stream end.

No server, no model loading. Regression for the same loss pattern as
#41821: text held in _buffer while waiting for a tool-call marker that
can never arrive was silently dropped by the base finish().
"""

import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.kimik2_detector import KimiK2Detector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

SECTION_BEGIN = "<|tool_calls_section_begin|>"
CALL_BEGIN = "<|tool_call_begin|>"
ARG_BEGIN = "<|tool_call_argument_begin|>"
CALL_END = "<|tool_call_end|>"


def _tools():
    return [
        Tool(
            type="function",
            function=Function(
                name="get_weather",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            ),
        )
    ]


class TestKimiK2FinishFlush(unittest.TestCase):
    def test_finish_releases_partial_tool_call_residue(self):
        """Stream ends after the argument-begin marker: residue must not be dropped."""
        d = KimiK2Detector()
        tools = _tools()
        chunk = (
            "x "
            + SECTION_BEGIN
            + CALL_BEGIN
            + "get_weather"
            + ARG_BEGIN
            + '{"location": "par'
        )
        r = d.parse_streaming_increment(chunk, tools)
        self.assertEqual(r.normal_text, "x ")
        self.assertNotEqual(d._buffer, "")

        f = d.finish(tools)
        # residue released, buffer drained
        self.assertEqual(d._buffer, "")

    def test_finish_empty_when_stream_completed(self):
        """A clean stream leaves nothing buffered; finish must not invent output."""
        d = KimiK2Detector()
        tools = _tools()
        r = d.parse_streaming_increment("plain text, no tool calls", tools)
        self.assertEqual(r.normal_text, "plain text, no tool calls")
        self.assertEqual(d._buffer, "")

        f = d.finish(tools)
        self.assertEqual(f.normal_text, "")
        self.assertEqual(f.calls, [])

    def test_finish_after_completed_stream_no_residue(self):
        """A stream that ended cleanly yields an empty finish result."""
        d = KimiK2Detector()
        tools = _tools()
        d.parse_streaming_increment("done", tools)
        f = d.finish(tools)
        self.assertEqual(f.normal_text, "")
        self.assertEqual(f.calls, [])
        self.assertEqual(d._buffer, "")


if __name__ == "__main__":
    unittest.main()
