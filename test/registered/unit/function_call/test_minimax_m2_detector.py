"""Unit tests for the MiniMax M2 function-call detector."""

import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.minimax_m2 import MinimaxM2Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMinimaxM2Detector(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="search",
                    parameters={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="lookup",
                    parameters={
                        "type": "object",
                        "properties": {"item": {"type": "string"}},
                    },
                ),
            ),
        ]

    def test_streaming_closes_tool_call_before_normal_text(self):
        detector = MinimaxM2Detector()

        first = detector.parse_streaming_increment(
            'before <minimax:tool_call><invoke name="search">'
            '<parameter name="query">cats</parameter></invoke>',
            self.tools,
        )
        self.assertEqual(first.normal_text, "before ")
        self.assertEqual(
            [(call.tool_index, call.name, call.parameters) for call in first.calls],
            [(0, "search", ""), (0, None, '{"query": "cats"'), (0, None, "}")],
        )

        second = detector.parse_streaming_increment(
            "</minimax:tool_call> after", self.tools
        )
        self.assertEqual(second.normal_text, " after")
        self.assertEqual(second.calls, [])
        self.assertEqual(detector._buf, "")
        self.assertFalse(detector._in_tool_call)

    def test_streaming_preserves_multi_invoke_tool_call_block(self):
        detector = MinimaxM2Detector()

        first = detector.parse_streaming_increment(
            'before <minimax:tool_call><invoke name="search">'
            '<parameter name="query">cats</parameter></invoke><in',
            self.tools,
        )
        self.assertEqual(first.normal_text, "before ")
        self.assertEqual(
            [(call.tool_index, call.name, call.parameters) for call in first.calls],
            [(0, "search", ""), (0, None, '{"query": "cats"'), (0, None, "}")],
        )

        second = detector.parse_streaming_increment(
            'voke name="lookup"><parameter name="item">dogs</parameter></invoke>'
            "</minimax:tool_call> after",
            self.tools,
        )
        self.assertEqual(second.normal_text, " after")
        self.assertEqual(
            [(call.tool_index, call.name, call.parameters) for call in second.calls],
            [(1, "lookup", ""), (1, None, '{"item": "dogs"'), (1, None, "}")],
        )
        self.assertEqual(detector._buf, "")
        self.assertFalse(detector._in_tool_call)


if __name__ == "__main__":
    unittest.main()
