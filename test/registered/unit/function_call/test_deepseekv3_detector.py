"""Unit tests for DeepSeekV3Detector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv3_detector import DeepSeekV3Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")

BOT = "<｜tool▁calls▁begin｜>"
CB = "<｜tool▁call▁begin｜>"
SEP = "<｜tool▁sep｜>"
CE = "<｜tool▁call▁end｜>"
EOT = "<｜tool▁calls▁end｜>"


def _make_call(name: str, args_json: str) -> str:
    return f"{CB}function{SEP}{name}\n```json\n{args_json}\n```{CE}"


class TestDeepSeekV3Detector(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                        "required": ["location"],
                    },
                ),
            ),
        ]

    def _run_stream(self, chunks):
        detector = DeepSeekV3Detector()
        names = []
        streamed_args = ""
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            for call in result.calls:
                if call.name:
                    names.append(call.name)
                if call.parameters:
                    streamed_args += call.parameters
        return detector, names, streamed_args

    def test_streaming_whole_call_in_single_increment_preserves_arguments(self):
        """A complete tool call delivered in one streaming increment must still
        emit its arguments. Previously only the name was emitted and the
        arguments were dropped (stored as ``{}``)."""
        text = BOT + _make_call("get_weather", '{"location": "Tokyo"}') + EOT
        detector, names, streamed_args = self._run_stream([text])

        self.assertEqual(names, ["get_weather"])
        self.assertEqual(json.loads(streamed_args), {"location": "Tokyo"})
        self.assertEqual(
            detector.prev_tool_call_arr[0]["arguments"], {"location": "Tokyo"}
        )

    def test_streaming_token_by_token_still_works(self):
        """Regression: token-by-token delivery keeps streaming name then args."""
        chunks = [
            BOT,
            CB,
            "function",
            SEP,
            "get_weather\n```json\n",
            '{"location": ',
            '"Tokyo"}',
            "\n```",
            CE,
            EOT,
        ]
        detector, names, streamed_args = self._run_stream(chunks)

        self.assertEqual(names, ["get_weather"])
        self.assertEqual(json.loads(streamed_args), {"location": "Tokyo"})


if __name__ == "__main__":
    import unittest

    unittest.main()
