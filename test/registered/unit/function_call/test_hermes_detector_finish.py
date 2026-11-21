"""Unit tests for HermesDetector finish() flush - no server, no model loading."""

import unittest

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.hermes_detector import HermesDetector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

TOOL = Tool(
    type="function",
    function={
        "name": "get_weather",
        "description": "x",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
)


class TestHermesDetectorFinishFlush(unittest.TestCase):
    def setUp(self):
        self.d = HermesDetector()

    def test_partial_tool_call_at_end_is_flushed(self):
        feed = (
            "hi "
            + self.d.bot_token
            + '{"name": "get_weather", "arguments": {"city": "P'
        )
        self.d.parse_streaming_increment(feed, tools=[TOOL])
        self.assertNotEqual(self.d._buffer, "")
        r = self.d.finish([TOOL])
        self.assertNotEqual(r.normal_text, "")

    def test_clean_stream_finish_is_noop(self):
        self.d.parse_streaming_increment("just some text", tools=[TOOL])
        r = self.d.finish([TOOL])
        self.assertEqual(r.normal_text, "")
        self.assertEqual(len(r.calls), 0)

    def test_complete_call_not_duplicated_at_finish(self):
        full = (
            self.d.bot_token
            + '{"name": "get_weather", "arguments": {"city": "Paris"}}'
            + self.d.eot_token
        )
        self.d.parse_streaming_increment(full, tools=[TOOL])
        # Fully closed call is not re-emitted at finish().
        r = self.d.finish([TOOL])
        self.assertEqual(r.normal_text, "")
        self.assertEqual(self.d._buffer, "")


if __name__ == "__main__":
    unittest.main()
