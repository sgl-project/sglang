"""Unit tests for GigaChat35Detector - no server, no model loading."""

import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.gigachat35_detector import GigaChat35Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_OPEN = "<｜GCML｜tool_calls>"
_CLOSE = "</｜GCML｜tool_calls>"


def _make_tools():
    return [
        Tool(
            type="function",
            function=Function(
                name="search",
                parameters={"type": "object", "properties": {"q": {"type": "string"}}},
            ),
        ),
    ]


def _invoke(name, q):
    return (
        f'<｜GCML｜invoke name="{name}">'
        f'<｜GCML｜parameter name="q" string="true">{q}</｜GCML｜parameter>'
        "</｜GCML｜invoke>"
    )


def _stream(text, tools):
    detector = GigaChat35Detector()
    calls = list(detector.parse_streaming_increment(text, tools).calls)
    calls += detector.finish(tools).calls
    return calls


class TestGigaChat35DetectorStreaming(CustomTestCase):
    def setUp(self):
        self.tools = _make_tools()
        self.text = _OPEN + _invoke("nope", "x") + _invoke("search", "y") + _CLOSE

    def test_skipped_unknown_tool_leaves_no_index_gap(self):
        """Streamed tool_index must stay contiguous after a skipped unknown tool."""
        with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(False):
            calls = _stream(self.text, self.tools)
        self.assertEqual([(c.tool_index, c.name) for c in calls], [(0, "search")])

    def test_forward_unknown_tools_in_streaming(self):
        """SGLANG_FORWARD_UNKNOWN_TOOLS must forward unknown tools when streaming too."""
        with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(True):
            calls = _stream(self.text, self.tools)
        self.assertEqual(
            [(c.tool_index, c.name) for c in calls], [(0, "nope"), (1, "search")]
        )


if __name__ == "__main__":
    unittest.main()
