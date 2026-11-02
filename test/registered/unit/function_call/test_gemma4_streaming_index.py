"""Unit tests for Gemma4Detector streaming tool-call indices - no server."""

import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.gemma4_detector import Gemma4Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _tools():
    return [
        Tool(
            type="function",
            function=Function(name="f", parameters={"type": "object"}),
        ),
        Tool(
            type="function",
            function=Function(name="g", parameters={"type": "object"}),
        ),
    ]


class TestGemma4StreamingIndex(CustomTestCase):
    def test_tool_index_follows_call_order(self):
        # OpenAI streaming deltas index tool calls by their order in the
        # response, not by the tool's position in the request tools array.
        detector = Gemma4Detector()
        result = detector.parse_streaming_increment(
            "<|tool_call>call:g{x: 1}<tool_call|><|tool_call>call:f{y: 2}<tool_call|>",
            _tools(),
        )
        self.assertEqual(
            [(c.name, c.tool_index) for c in result.calls],
            [("g", 0), (None, 0), ("f", 1), (None, 1)],
        )

    def test_single_later_tool_indexes_from_zero(self):
        detector = Gemma4Detector()
        result = detector.parse_streaming_increment(
            "<|tool_call>call:g{x: 1}<tool_call|>", _tools()
        )
        self.assertEqual([c.tool_index for c in result.calls], [0, 0])


if __name__ == "__main__":
    unittest.main()
