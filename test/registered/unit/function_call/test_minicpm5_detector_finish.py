"""Unit tests for MiniCPM5Detector finish() - no server, no model loading."""

import unittest

from sglang.srt.function_call.minicpm5_detector import MiniCPM5Detector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestMiniCPM5DetectorFinish(unittest.TestCase):
    def setUp(self):
        self.d = MiniCPM5Detector()
        self.bot = self.d.bot_token

    def test_truncated_call_at_end_drops_protocol_block(self):
        r1 = self.d.parse_streaming_increment(
            "hi " + self.bot + "get_weather('par", tools=[]
        )
        self.assertEqual(r1.normal_text, "hi ")
        self.assertEqual(self.d._buffer, self.bot + "get_weather('par")
        r2 = self.d.finish([])
        self.assertEqual(r2.normal_text, "")
        self.assertEqual(len(r2.calls), 0)
        self.assertEqual(self.d._buffer, "")

    def test_truncated_marker_prefix_is_released(self):
        r1 = self.d.parse_streaming_increment("hello <fun", tools=[])
        self.assertEqual(r1.normal_text, "hello ")
        r2 = self.d.finish([])
        self.assertEqual(r2.normal_text, "<fun")

    def test_clean_stream_finish_is_noop(self):
        self.d.parse_streaming_increment("just some text", tools=[])
        r = self.d.finish([])
        self.assertEqual(r.normal_text, "")
        self.assertEqual(len(r.calls), 0)

    def test_complete_call_not_duplicated_at_finish(self):
        eot = self.d.eot_token
        full = self.bot + ' name="get_weather"' + eot
        self.d.parse_streaming_increment(full, tools=[])
        self.assertEqual(self.d._buffer, "")
        r = self.d.finish([])
        self.assertEqual(r.normal_text, "")


if __name__ == "__main__":
    unittest.main()
