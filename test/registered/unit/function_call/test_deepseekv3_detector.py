"""Unit tests for DeepSeekV3Detector one-shot and streaming parsing."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv3_detector import DeepSeekV3Detector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-b-test-cpu")


def _tool_call_block(name: str, args_str: str) -> str:
    return (
        f"<｜tool▁call▁begin｜>function<｜tool▁sep｜>{name}\n"
        f"```json\n{args_str}\n```<｜tool▁call▁end｜>"
    )


class TestDeepSeekV3DetectorUnit(unittest.TestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="get_tourist_attractions",
                    description="Get tourist attractions",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                ),
            ),
        ]

    def test_detect_and_parse_isolates_malformed_call_in_multi_call_turn(self):
        detector = DeepSeekV3Detector()
        valid_first = _tool_call_block("get_weather", '{"city": "Tokyo"}')
        malformed_second = (
            "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
            '```json\n{"city": \n```<｜tool▁call▁end｜>'
        )
        unmatched_third = "<｜tool▁call▁begin｜>broken_without_sep<｜tool▁call▁end｜>"
        valid_fourth = _tool_call_block("get_tourist_attractions", '{"city": "Paris"}')
        text = (
            "Checking cities.\n"
            "<｜tool▁calls▁begin｜>"
            f"{valid_first}\n{malformed_second}\n{unmatched_third}\n{valid_fourth}"
            "<｜tool▁calls▁end｜>"
        )

        result = detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, "Checking cities.")
        self.assertNotIn("<｜tool▁calls▁begin｜>", result.normal_text)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "Tokyo"})
        self.assertEqual(result.calls[1].name, "get_tourist_attractions")
        self.assertEqual(json.loads(result.calls[1].parameters), {"city": "Paris"})

    def test_parse_streaming_preserves_leading_normal_text_and_multiple_calls_in_one_chunk(
        self,
    ):
        detector = DeepSeekV3Detector()
        call1 = _tool_call_block("get_weather", '{"city": "Tokyo"}')
        call2 = _tool_call_block("get_tourist_attractions", '{"city": "Beijing"}')
        text = (
            "Let me look that up.\n"
            f"<｜tool▁calls▁begin｜>{call1}\n{call2}<｜tool▁calls▁end｜>"
        )

        result = detector.parse_streaming_increment(text, self.tools)

        self.assertEqual(result.normal_text, "Let me look that up.\n")
        names = [c.name for c in result.calls if c.name]
        params = [json.loads(c.parameters) for c in result.calls if c.parameters]
        self.assertEqual(names, ["get_weather", "get_tourist_attractions"])
        self.assertEqual(params, [{"city": "Tokyo"}, {"city": "Beijing"}])

    def test_parse_streaming_fine_grained_chunks(self):
        detector = DeepSeekV3Detector()
        call1 = _tool_call_block("get_weather", '{"city": "Tokyo"}')
        call2 = _tool_call_block("get_tourist_attractions", '{"city": "Beijing"}')
        text = f"Checking.\n<｜tool▁calls▁begin｜>{call1}\n{call2}<｜tool▁calls▁end｜>"

        normal_parts = []
        names = []
        params_by_index = {}
        for i in range(0, len(text), 5):
            res = detector.parse_streaming_increment(text[i : i + 5], self.tools)
            if res.normal_text:
                normal_parts.append(res.normal_text)
            for call in res.calls:
                if call.name:
                    names.append((call.tool_index, call.name))
                if call.parameters:
                    params_by_index.setdefault(call.tool_index, "")
                    params_by_index[call.tool_index] += call.parameters

        self.assertEqual("".join(normal_parts), "Checking.\n")
        self.assertEqual(names, [(0, "get_weather"), (1, "get_tourist_attractions")])
        self.assertEqual(json.loads(params_by_index[0]), {"city": "Tokyo"})
        self.assertEqual(json.loads(params_by_index[1]), {"city": "Beijing"})


if __name__ == "__main__":
    unittest.main()
