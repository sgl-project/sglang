"""Unit tests for MinimaxM3Detector finish() flush - no server, no model loading."""

import unittest

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.minimax_m3 import MinimaxM3Detector
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


class TestMinimaxM3DetectorFinishFlush(unittest.TestCase):
    def setUp(self):
        self.d = MinimaxM3Detector()

    def test_partial_tool_call_at_end_is_flushed(self):
        feed = (
            "hi "
            + self.d.TOOL_CALL_START
            + self.d.INVOKE_PREFIX
            + 'get_weather"'
            + GT
            + LT
            + 'parameter name="city"'
            + GT
            + "P"
        )
        r1 = self.d.parse_streaming_increment(feed, tools=[TOOL])
        self.assertEqual(r1.normal_text, "hi ")
        self.assertNotEqual(self.d._buffer, "")
        r2 = self.d.finish([TOOL])
        self.assertNotEqual(r2.normal_text, "")

    def test_clean_stream_finish_is_noop(self):
        self.d.parse_streaming_increment("just some text", tools=[TOOL])
        r = self.d.finish([TOOL])
        self.assertEqual(r.normal_text, "")
        self.assertEqual(len(r.calls), 0)

    def test_complete_call_not_duplicated_at_finish(self):
        full = (
            self.d.TOOL_CALL_START
            + self.d.INVOKE_PREFIX
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
            + self.d.TOOL_CALL_END
        )
        self.d.parse_streaming_increment(full, tools=[TOOL])
        self.assertEqual(self.d._buffer, "")
        r = self.d.finish([TOOL])
        self.assertEqual(r.normal_text, "")


if __name__ == "__main__":
    unittest.main()
