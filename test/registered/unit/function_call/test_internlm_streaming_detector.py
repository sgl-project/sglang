import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.internlm_detector import InternlmDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _call(location: str) -> str:
    body = json.dumps(
        {"name": "get_weather", "parameters": {"location": location}},
        ensure_ascii=False,
    )
    return "<|action_start|> <|plugin|>\n" + body + "<|action_end|>"


def _tools() -> list[Tool]:
    return [
        Tool(
            type="function",
            function=Function(
                name="get_weather",
                parameters={
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            ),
        )
    ]


class TestInternlmStreamingSecondCall(CustomTestCase):
    def test_second_call_survives_split_start_marker(self):
        text = _call("Tokyo") + _call("Paris")
        marker = "<|action_start|>"
        second = text.find(marker, 1)
        chunks = [
            text[:second],
            text[second : second + len(marker)],
            text[second + len(marker) :],
        ]
        detector = InternlmDetector()
        names = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, _tools())
            for call in result.calls:
                if call.name:
                    names.append(
                        (call.tool_index, call.name, json.loads(call.parameters))
                    )
        self.assertEqual(
            names,
            [
                (0, "get_weather", {"location": "Tokyo"}),
                (1, "get_weather", {"location": "Paris"}),
            ],
        )

    def test_nonstreaming_still_returns_both_calls(self):
        text = _call("Tokyo") + _call("Paris")
        result = InternlmDetector().detect_and_parse(text, _tools())
        self.assertEqual(
            [call.name for call in result.calls], ["get_weather", "get_weather"]
        )
        self.assertEqual(
            [json.loads(call.parameters)["location"] for call in result.calls],
            ["Tokyo", "Paris"],
        )


if __name__ == "__main__":
    unittest.main()
