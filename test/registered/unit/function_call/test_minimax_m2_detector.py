import json
import re
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.minimax_m2 import MinimaxM2Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")

TOOL_CALL_START = "<minimax:tool_call>"
TOOL_CALL_END = "</minimax:tool_call>"


class TestMinimaxM2DetectorStreaming(CustomTestCase):
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
                            "city": {"type": "string"},
                            "days": {"type": "integer"},
                        },
                        "required": ["city"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="noop",
                    description="Takes no arguments",
                    parameters={"type": "object", "properties": {}},
                ),
            ),
        ]

    def _stream(self, chunks):
        detector = MinimaxM2Detector()
        calls, normal = {}, ""
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            normal += result.normal_text
            for call in result.calls:
                name, args = calls.get(call.tool_index, (None, ""))
                calls[call.tool_index] = (name or call.name, args + call.parameters)
        return [calls[i] for i in sorted(calls)], normal

    def assert_stream_matches_final(self, text):
        reference = MinimaxM2Detector().detect_and_parse(text, self.tools)
        want = [(c.name, json.loads(c.parameters)) for c in reference.calls]
        # The markers are single tokens in the MiniMax-M2 vocabulary.
        tokens = []
        for part in re.split(f"({TOOL_CALL_START}|{TOOL_CALL_END})", text):
            tokens.extend([part] if part in (TOOL_CALL_START, TOOL_CALL_END) else part)

        for label, chunks in {"whole": [text], "per_token": tokens}.items():
            with self.subTest(chunking=label):
                got, normal = self._stream(chunks)
                self.assertEqual([(name, json.loads(args)) for name, args in got], want)
                self.assertEqual(normal.strip(), reference.normal_text.strip())

    def test_closing_brace_inside_string_argument(self):
        self.assert_stream_matches_final(
            f'{TOOL_CALL_START}<invoke name="get_weather">'
            f'<parameter name="city">a}}b</parameter></invoke>{TOOL_CALL_END}'
        )

    def test_two_calls_keep_their_own_arguments(self):
        self.assert_stream_matches_final(
            f'{TOOL_CALL_START}<invoke name="get_weather">'
            f'<parameter name="city">Beijing</parameter></invoke>'
            f'<invoke name="noop"></invoke>'
            f'<invoke name="get_weather">'
            f'<parameter name="days">3</parameter></invoke>{TOOL_CALL_END}'
        )

    def test_normal_text_after_tool_call(self):
        self.assert_stream_matches_final(
            f'{TOOL_CALL_START}<invoke name="get_weather"></invoke>{TOOL_CALL_END}'
            f' Then <invoke name="noop"> {TOOL_CALL_START}<invoke name="noop">'
            f"</invoke>{TOOL_CALL_END} Done."
        )

    def test_unknown_tool_is_dropped_not_leaked(self):
        self.assert_stream_matches_final(
            f'{TOOL_CALL_START}<invoke name="not_registered"></invoke>'
            f'<invoke name="noop"></invoke>{TOOL_CALL_END}'
        )


if __name__ == "__main__":
    unittest.main()
