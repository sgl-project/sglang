"""Unit tests for GigaChat3Detector finish() flush - no server, no model loading."""

import unittest

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.gigachat3_detector import GigaChat3Detector
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

MARKER = LT + "|function_call|" + GT


class TestGigaChat3DetectorFinishFlush(unittest.TestCase):
    def setUp(self):
        self.d = GigaChat3Detector()

    def test_partial_tool_call_at_end_is_flushed(self):
        feed = "hi " + MARKER + ' {"name": "get_weather", "arguments": {"city": "P'
        self.d.parse_streaming_increment(feed, tools=[TOOL])
        self.assertNotEqual(self.d._buffer, "")
        r = self.d.finish([TOOL])
        self.assertNotEqual(r.normal_text, "")

    def test_plain_text_finish_flushes_not_drops(self):
        self.d.parse_streaming_increment("just some text", tools=[TOOL])
        # GigaChat3 keeps non-tool text in the buffer; finish() must flush it.
        r = self.d.finish([TOOL])
        self.assertIn("just some text", r.normal_text)
        self.assertEqual(self.d._buffer, "")

    def test_complete_call_not_duplicated_at_finish(self):
        full = MARKER + ' {"name": "get_weather", "arguments": {"city": "Paris"}}'
        self.d.parse_streaming_increment(full, tools=[TOOL])
        # Fully closed JSON is not re-emitted at finish().
        r = self.d.finish([TOOL])
        self.assertEqual(r.normal_text, "")
        self.assertEqual(self.d._buffer, "")


if __name__ == "__main__":
    unittest.main()
