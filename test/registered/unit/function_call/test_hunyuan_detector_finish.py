"""Unit tests for HunyuanDetector finish() flush - no server, no model loading."""

import unittest

from sglang.srt.function_call.hunyuan_detector import HunyuanDetector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestHunyuanDetectorFinishFlush(unittest.TestCase):
    def setUp(self):
        self.d = HunyuanDetector()
        self.bot = self.d.bot_token

    def test_partial_tool_call_at_end_is_flushed(self):
        feed = "hi " + self.bot + "get_weather('par"
        r1 = self.d.parse_streaming_increment(feed, tools=[])
        self.assertEqual(r1.normal_text, "hi ")
        self.assertNotEqual(self.d._buffer, "")
        r2 = self.d.finish([])
        self.assertIn("get_weather", r2.normal_text)

    def test_clean_stream_finish_is_noop(self):
        self.d.parse_streaming_increment("just some text", tools=[])
        r = self.d.finish([])
        self.assertEqual(r.normal_text, "")
        self.assertEqual(len(r.calls), 0)

    def test_complete_call_not_duplicated_at_finish(self):
        eot = self.d.eot_token
        full = (
            self.bot
            + self.d.tool_call_start_token
            + "get_current_date"
            + self.d.tool_sep_token
            + self.d.tool_call_end_token
            + eot
        )
        self.d.parse_streaming_increment(full, tools=[])
        self.assertEqual(self.d._buffer, "")
        r = self.d.finish([])
        self.assertEqual(r.normal_text, "")


if __name__ == "__main__":
    unittest.main()
