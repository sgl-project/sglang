"""Unit tests for InternlmDetector finish() flush - no server, no model loading."""

import unittest

from sglang.srt.function_call.internlm_detector import InternlmDetector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PARTIAL = "hi <|action_start|> <|plugin|>get_weather('par"
RESIDUE = "<|action_start|> <|plugin|>get_weather('par"


class TestInternlmDetectorFinishFlush(unittest.TestCase):
    def setUp(self):
        self.d = InternlmDetector()

    def test_partial_tool_call_at_end_is_flushed(self):
        r1 = self.d.parse_streaming_increment(PARTIAL, tools=[])
        self.assertEqual(r1.normal_text, "hi ")
        self.assertEqual(self.d._buffer, RESIDUE)
        r2 = self.d.finish([])
        self.assertEqual(r2.normal_text, RESIDUE)

    def test_clean_stream_finish_is_noop(self):
        self.d.parse_streaming_increment("just some text", tools=[])
        r = self.d.finish([])
        self.assertEqual(r.normal_text, "")
        self.assertEqual(len(r.calls), 0)

    def test_complete_call_not_duplicated_at_finish(self):
        full = '<|action_start|> <|plugin|>\n{"name": "get_weather", "arguments": {"city": "Paris"}}<|action_end|>'
        self.d.parse_streaming_increment(full, tools=[])
        self.assertEqual(self.d._buffer, "")
        r = self.d.finish([])
        self.assertEqual(r.normal_text, "")


if __name__ == "__main__":
    unittest.main()
