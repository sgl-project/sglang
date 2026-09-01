"""Unit tests for MiniMax-M2 tool call indices — no server, no model loading."""

import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.minimax_m2 import MinimaxM2Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


def _make_tools():
    def tool(name, properties):
        return Tool(
            type="function",
            function=Function(
                name=name,
                description=f"{name} tool",
                parameters={"type": "object", "properties": properties},
            ),
        )

    return [
        tool("get_weather", {"city": {"type": "string"}}),
        tool("search_web", {"query": {"type": "string"}}),
        tool("calculate", {"expr": {"type": "string"}}),
    ]


def _build_message(calls):
    parts = ["<minimax:tool_call>"]
    for name, parameters in calls:
        parts.append(f'<invoke name="{name}">')
        for key, value in parameters.items():
            parts.append(f'<parameter name="{key}">{value}</parameter>')
        parts.append("</invoke>")
    parts.append("</minimax:tool_call>")
    return "\n".join(parts)


class TestMinimaxM2ToolIndex(CustomTestCase):
    def setUp(self):
        self.tools = _make_tools()

    def _single_shot_calls(self, text):
        return MinimaxM2Detector().detect_and_parse(text, self.tools).calls

    def _single_shot_indices(self, text):
        return [call.tool_index for call in self._single_shot_calls(text)]

    def _streaming_name_indices(self, text):
        result = MinimaxM2Detector().parse_streaming_increment(text, self.tools)
        return [call.tool_index for call in result.calls if call.name is not None]

    def test_parallel_same_tool_indices_single_shot(self):
        text = _build_message(
            [
                ("get_weather", {"city": "Tokyo"}),
                ("get_weather", {"city": "Paris"}),
            ]
        )

        self.assertEqual(self._single_shot_indices(text), [0, 1])

    def test_index_is_ordinal_not_registry_slot(self):
        text = _build_message([("calculate", {"expr": "2+2"})])

        self.assertEqual(self._single_shot_indices(text), [0])

    def test_indices_are_dense_when_unknown_tool_is_dropped(self):
        text = _build_message(
            [
                ("get_weather", {"city": "Tokyo"}),
                ("unknown_tool", {"city": "Paris"}),
                ("calculate", {"expr": "2+2"}),
            ]
        )

        with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(False):
            calls = self._single_shot_calls(text)

        self.assertEqual([call.name for call in calls], ["get_weather", "calculate"])
        self.assertEqual([call.tool_index for call in calls], [0, 1])

    def test_forwarded_unknown_tool_gets_ordinal_index(self):
        text = _build_message(
            [
                ("get_weather", {"city": "Tokyo"}),
                ("unknown_tool", {"city": "Paris"}),
                ("calculate", {"expr": "2+2"}),
            ]
        )

        with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(True):
            calls = self._single_shot_calls(text)

        self.assertEqual(
            [call.name for call in calls],
            ["get_weather", "unknown_tool", "calculate"],
        )
        self.assertEqual([call.tool_index for call in calls], [0, 1, 2])

    def test_single_shot_matches_streaming_indices(self):
        text = _build_message(
            [
                ("get_weather", {"city": "Tokyo"}),
                ("get_weather", {"city": "Paris"}),
            ]
        )

        self.assertEqual(
            self._single_shot_indices(text), self._streaming_name_indices(text)
        )


if __name__ == "__main__":
    unittest.main()
