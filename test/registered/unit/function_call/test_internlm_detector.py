"""Unit tests for InternlmDetector finish() - no server, no model loading."""

import unittest

from sglang.srt.function_call.internlm_detector import InternlmDetector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

BOT = "<|action_start|> <|plugin|>"


class TestInternlmDetectorFinish(unittest.TestCase):
    def setUp(self):
        self.d = InternlmDetector()

    def test_truncated_call_at_end_drops_protocol_block(self):
        r1 = self.d.parse_streaming_increment(
            "hi " + BOT + "get_weather('par", tools=[]
        )
        self.assertEqual(r1.normal_text, "hi ")
        self.assertEqual(self.d._buffer, BOT + "get_weather('par")
        r2 = self.d.finish([])
        self.assertEqual(r2.normal_text, "")
        self.assertEqual(len(r2.calls), 0)
        self.assertEqual(self.d._buffer, "")

    def test_text_before_truncated_marker_is_released(self):
        full = BOT + '{"name": "get_weather", "arguments": {}}<|action_end|>'
        self.d.parse_streaming_increment(full + "done " + BOT + "par", tools=[])
        r = self.d.finish([])
        self.assertEqual(r.normal_text, "done ")
        self.assertNotIn("<|action", r.normal_text)

    def test_truncated_marker_prefix_is_released(self):
        self.d.parse_streaming_increment("hello <|action_sta", tools=[])
        r = self.d.finish([])
        self.assertEqual(r.normal_text, "hello <|action_sta")

    def test_clean_stream_finish_is_noop(self):
        self.d.parse_streaming_increment("just some text", tools=[])
        r = self.d.finish([])
        self.assertEqual(r.normal_text, "")
        self.assertEqual(len(r.calls), 0)

    def test_complete_call_not_duplicated_at_finish(self):
        full = (
            BOT
            + '\n{"name": "get_weather", "arguments": {"city": "Paris"}}<|action_end|>'
        )
        self.d.parse_streaming_increment(full, tools=[])
        self.assertEqual(self.d._buffer, "")
        r = self.d.finish([])
        self.assertEqual(r.normal_text, "")


if __name__ == "__main__":
    unittest.main()
