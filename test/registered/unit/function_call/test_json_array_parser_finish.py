"""Unit tests for JsonArrayParser finish() flush - no server, no model loading."""

import unittest

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.json_array_parser import JsonArrayParser
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


class TestJsonArrayParserFinishFlush(unittest.TestCase):
    def setUp(self):
        self.d = JsonArrayParser()

    def test_partial_tool_call_at_end_is_flushed(self):
        feed = '[{"name": "get_weather", "arguments": {"city": "P'
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
        full = '[{"name": "get_weather", "arguments": {"city": "Paris"}}]'
        self.d.parse_streaming_increment(full, tools=[TOOL])
        # The base parser keeps the closed JSON in _buffer until the stream
        # ends; finish() must not re-emit it as user text.
        r = self.d.finish([TOOL])
        self.assertEqual(r.normal_text, "")
        self.assertEqual(self.d._buffer, "")


if __name__ == "__main__":
    unittest.main()
