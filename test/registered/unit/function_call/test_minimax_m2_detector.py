import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.minimax_m2 import MinimaxM2Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _tool():
    return Tool(
        type="function",
        function=Function(
            name="search",
            description="Search the web",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "ratio": {"type": "number"},
                },
                "required": ["query"],
            },
        ),
    )


def _wire(ratio):
    return (
        "<minimax:tool_call>"
        '<invoke name="search">'
        '<parameter name="query">hi</parameter>'
        f'<parameter name="ratio">{ratio}</parameter>'
        "</invoke>"
        "</minimax:tool_call>"
    )


class TestMinimaxM2NumberConversion(CustomTestCase):
    def setUp(self):
        self.tools = [_tool()]

    def test_normal_number_parses(self):
        result = MinimaxM2Detector().detect_and_parse(_wire("0.5"), self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(json.loads(result.calls[0].parameters)["ratio"], 0.5)

    def test_overflow_literal_does_not_crash_and_stays_valid_json(self):
        # "1e999" -> inf; int(inf) raised OverflowError and crashed
        # detect_and_parse. It must now degrade to the raw string and the
        # arguments must remain valid JSON (no `Infinity`).
        result = MinimaxM2Detector().detect_and_parse(_wire("1e999"), self.tools)
        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)  # must not raise
        self.assertEqual(args["ratio"], "1e999")

    def test_nan_literal_stays_valid_json(self):
        result = MinimaxM2Detector().detect_and_parse(_wire("nan"), self.tools)
        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)  # must not raise
        self.assertEqual(args["ratio"], "nan")


if __name__ == "__main__":
    unittest.main()
