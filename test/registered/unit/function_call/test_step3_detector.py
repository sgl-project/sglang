"""Unit tests: Step3Detector finish() flushes buffered text at stream end.

No server, no model loading. Regression for the same loss pattern as
#41821: text held in _buffer while waiting for a tool-call marker that
can never arrive was silently dropped by the base finish().
"""

import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.step3_detector import Step3Detector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

BEGIN = "<\uff5ctool_calls_begin\uff5c>"
END = "<\uff5ctool_calls_end\uff5c>"
CALL_BEGIN = "<\uff5ctool_call_begin\uff5c>"


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


class TestStep3FinishFlush(unittest.TestCase):
    def test_finish_releases_partial_tool_call_residue(self):
        """Stream ends inside a tool block with an unfinished call: residue must not be dropped."""
        d = Step3Detector()
        tools = _tools()
        chunk = "hi " + BEGIN + CALL_BEGIN + "get_weather('par"
        r = d.parse_streaming_increment(chunk, tools)
        self.assertEqual(r.normal_text, "hi ")
        self.assertNotEqual(d._buffer, "")

        f = d.finish(tools)
        self.assertEqual(f.normal_text, CALL_BEGIN + "get_weather('par")
        self.assertEqual(f.calls, [])
        # residue was released (buffer now drained)
        self.assertEqual(d._buffer, "")

    def test_finish_empty_when_stream_completed(self):
        """A clean stream leaves nothing buffered; finish must not invent output."""
        d = Step3Detector()
        tools = _tools()
        r = d.parse_streaming_increment("plain text, no tool calls", tools)
        self.assertEqual(r.normal_text, "plain text, no tool calls")
        self.assertEqual(d._buffer, "")

        f = d.finish(tools)
        self.assertEqual(f.normal_text, "")
        self.assertEqual(f.calls, [])

    def test_finish_after_completed_stream_no_residue(self):
        """A stream that ended cleanly yields an empty finish result."""
        d = Step3Detector()
        tools = _tools()
        d.parse_streaming_increment("done", tools)
        f = d.finish(tools)
        self.assertEqual(f.normal_text, "")
        self.assertEqual(f.calls, [])
        self.assertEqual(d._buffer, "")


if __name__ == "__main__":
    unittest.main()
