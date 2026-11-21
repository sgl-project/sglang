"""Unit tests for DeepSeekV31Detector - no server or model loading."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv31_detector import DeepSeekV31Detector
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDeepSeekV31Detector(unittest.TestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                ),
            )
        ]

    def test_structure_info_round_trips_through_non_stream_parser(self):
        parser = FunctionCallParser(self.tools, "deepseekv31")
        info = parser.detector.structure_info()("get_weather")
        source = info.begin + '{"city": "Paris"}' + info.end

        normal_text, calls = parser.parse_non_stream(source)

        self.assertEqual(normal_text, "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(json.loads(calls[0].parameters), {"city": "Paris"})

    def test_structure_info_includes_tool_calls_section_begin(self):
        info = DeepSeekV31Detector().structure_info()("get_weather")

        self.assertTrue(info.begin.startswith("<｜tool▁calls▁begin｜>"))
        self.assertIn("<｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>", info.begin)


if __name__ == "__main__":
    unittest.main()
