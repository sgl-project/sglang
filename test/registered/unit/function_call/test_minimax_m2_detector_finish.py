"""Unit tests for MinimaxM2Detector finish() flush - no server, no model loading."""

import unittest

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.minimax_m2 import MinimaxM2Detector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

LT = chr(60)  # <
GT = chr(62)  # >

TOOL = Tool(
    type="function",
    function={
        "name": "get_weather",
        "description": "x",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
)


class TestMinimaxM2DetectorFinishFlush(unittest.TestCase):
    def setUp(self):
        self.d = MinimaxM2Detector()

    def test_partial_tool_call_at_end_is_flushed(self):
        feed = (
            "hi "
            + self.d.tool_call_start_token
            + self.d.tool_call_prefix
            + 'get_weather"'
            + GT
            + LT
            + 'parameter name="city"'
            + GT
            + "P"
        )
        r1 = self.d.parse_streaming_increment(feed, tools=[TOOL])
        self.assertEqual(r1.normal_text, "hi ")
        self.assertNotEqual(self.d._buf, "")
        r2 = self.d.finish([TOOL])
        self.assertNotEqual(r2.normal_text, "")

    def test_clean_stream_finish_is_noop(self):
        self.d.parse_streaming_increment("just some text", tools=[TOOL])
        r = self.d.finish([TOOL])
        self.assertEqual(r.normal_text, "")
        self.assertEqual(len(r.calls), 0)

    def test_complete_call_not_duplicated_at_finish(self):
        full = (
            self.d.tool_call_start_token
            + self.d.tool_call_prefix
            + 'get_weather"'
            + GT
            + LT
            + 'parameter name="city"'
            + GT
            + "Paris"
            + LT
            + "/parameter"
            + GT
            + LT
            + "/invoke"
            + GT
            + self.d.tool_call_end_token
        )
        self.d.parse_streaming_increment(full, tools=[TOOL])
        # The M2 state machine consumes the trailing closing tag on the next
        # parse; if the stream ends first, finish() must not release it as text.
        r = self.d.finish([TOOL])
        self.assertEqual(r.normal_text, "")
        self.assertEqual(self.d._buf, "")


if __name__ == "__main__":
    unittest.main()
