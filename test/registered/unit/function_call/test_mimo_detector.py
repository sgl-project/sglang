"""Unit tests: MiMoDetector finish() flushes buffered text at stream end.

No server, no model loading. Regression for the same loss pattern as
#41821: text held in _buffer while waiting for a tool-call marker that
can never arrive was silently dropped by the base finish().
"""

import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.mimo_detector import MiMoDetector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

BOT = "<tool_call>"
EOT = "</tool_call>"


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


class TestMiMoFinishFlush(unittest.TestCase):
    def test_finish_releases_partial_tool_call_residue(self):
        """Stream ends in the middle of a tool call: residue must not be dropped."""
        d = MiMoDetector()
        tools = _tools()
        r = d.parse_streaming_increment("hello " + BOT + "get_weather('par", tools)
        self.assertEqual(r.normal_text, "hello ")
        # residue is still buffered waiting for the closing marker
        self.assertEqual(d._buffer, BOT + "get_weather('par")

        f = d.finish(tools)
        self.assertEqual(f.normal_text, BOT + "get_weather('par")
        self.assertEqual(f.calls, [])
        # buffer drained
        self.assertEqual(d._buffer, "")

    def test_finish_empty_when_stream_completed(self):
        """A clean stream leaves nothing buffered; finish must not invent output."""
        d = MiMoDetector()
        tools = _tools()
        r = d.parse_streaming_increment("plain text, no tool calls", tools)
        self.assertEqual(r.normal_text, "plain text, no tool calls")
        self.assertEqual(d._buffer, "")

        f = d.finish(tools)
        self.assertEqual(f.normal_text, "")
        self.assertEqual(f.calls, [])

    def test_finish_after_complete_tool_call_no_duplication(self):
        """A fully parsed tool call must not be re-emitted by finish()."""
        d = MiMoDetector()
        tools = _tools()
        chunk = BOT + '{"name": "get_weather", "arguments": {"city": "Paris"}}' + EOT
        r = d.parse_streaming_increment(chunk, tools)
        self.assertEqual(r.calls, [])
        self.assertEqual(d._buffer, "")

        f = d.finish(tools)
        self.assertEqual(f.normal_text, "")
        self.assertEqual(f.calls, [])


if __name__ == "__main__":
    unittest.main()
