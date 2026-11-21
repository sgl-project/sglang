"""Unit tests: Lfm2Detector finish() flushes buffered text at stream end.

No server, no model loading. Regression for the same loss pattern as
#41821: text held in _buffer while waiting for a tool-call marker that
can never arrive was silently dropped by the base finish().
"""

import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.lfm2_detector import Lfm2Detector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

BOT = "<|tool_call_start|>"
EOT = "<|tool_call_end|>"


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


class TestLfm2FinishFlush(unittest.TestCase):
    def test_finish_releases_partial_tool_call_residue(self):
        """Stream ends in the middle of a pythonic tool call: residue must not be dropped."""
        d = Lfm2Detector()
        tools = _tools()
        r = d.parse_streaming_increment("hi " + BOT + "get_weather('par", tools)
        self.assertEqual(r.normal_text, "hi ")
        self.assertEqual(d._buffer, BOT + "get_weather('par")

        f = d.finish(tools)
        self.assertEqual(f.normal_text, BOT + "get_weather('par")
        self.assertEqual(f.calls, [])
        self.assertEqual(d._buffer, "")

    def test_finish_empty_when_stream_completed(self):
        """A clean stream leaves nothing buffered; finish must not invent output."""
        d = Lfm2Detector()
        tools = _tools()
        r = d.parse_streaming_increment("plain text, no tool calls", tools)
        self.assertEqual(r.normal_text, "plain text, no tool calls")
        self.assertEqual(d._buffer, "")

        f = d.finish(tools)
        self.assertEqual(f.normal_text, "")
        self.assertEqual(f.calls, [])

    def test_finish_after_completed_stream_no_residue(self):
        """A stream that ended cleanly yields an empty finish result."""
        d = Lfm2Detector()
        tools = _tools()
        d.parse_streaming_increment("done", tools)
        f = d.finish(tools)
        self.assertEqual(f.normal_text, "")
        self.assertEqual(f.calls, [])
        self.assertEqual(d._buffer, "")


if __name__ == "__main__":
    unittest.main()
