import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.minimax_m2 import MinimaxM2Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_tools():
    return [
        Tool(
            type="function",
            function=Function(
                name="get_weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer"},
                    },
                },
            ),
        ),
        Tool(
            type="function",
            function=Function(
                name="get_time", parameters={"type": "object", "properties": {}}
            ),
        ),
    ]


def _stream(chunks):
    detector, tools = MinimaxM2Detector(), _make_tools()
    names, args = {}, {}
    for chunk in chunks:
        for call in detector.parse_streaming_increment(chunk, tools).calls:
            if call.name:
                names[call.tool_index] = call.name
            args[call.tool_index] = args.get(call.tool_index, "") + call.parameters
    return [(names[i], json.loads(args[i] or "{}")) for i in sorted(names)]


class TestMinimaxM2StreamingParameterScope(CustomTestCase):
    """A delta that closes one invoke and carries the next invoke's
    parameters must not leak those parameters into the closing call."""

    TEXT = (
        "<minimax:tool_call>\n"
        '<invoke name="get_weather">\n'
        '<parameter name="city">Tokyo</parameter>\n'
        '<parameter name="days">3</parameter>\n'
        "</invoke>\n"
        '<invoke name="get_weather">\n'
        '<parameter name="city">Paris</parameter>\n'
        '<parameter name="days">1</parameter>\n'
        "</invoke>\n"
        "</minimax:tool_call>"
    )

    def test_two_calls_in_one_delta(self):
        self.assertEqual(
            _stream([self.TEXT]),
            [
                ("get_weather", {"city": "Tokyo", "days": 3}),
                ("get_weather", {"city": "Paris", "days": 1}),
            ],
        )

    def test_delta_spanning_invoke_boundary(self):
        split_at = self.TEXT.index('<parameter name="days">3')
        end = self.TEXT.index("</invoke>", self.TEXT.index("Paris"))
        chunks = [self.TEXT[:split_at], self.TEXT[split_at:end], self.TEXT[end:]]
        self.assertEqual(
            _stream(chunks),
            [
                ("get_weather", {"city": "Tokyo", "days": 3}),
                ("get_weather", {"city": "Paris", "days": 1}),
            ],
        )

    def test_no_arg_call_followed_by_call_with_args(self):
        text = (
            "<minimax:tool_call>\n"
            '<invoke name="get_time">\n'
            "</invoke>\n"
            '<invoke name="get_weather">\n'
            '<parameter name="city">Paris</parameter>\n'
            "</invoke>\n"
            "</minimax:tool_call>"
        )
        self.assertEqual(
            _stream([text]),
            [("get_time", {}), ("get_weather", {"city": "Paris"})],
        )


if __name__ == "__main__":
    unittest.main()
