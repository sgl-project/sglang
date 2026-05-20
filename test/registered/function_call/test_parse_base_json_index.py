"""
Unit tests for parse_base_json sequential tool_index assignment.

OpenAI's tool_calls[].index must be the sequential position of the call
within the response (0, 1, 2, …), not the tool's position in the tools
definition list.

Previously, parse_base_json used tool_indices.get(name) which returned
the tool's position in the *tools list*, producing wrong indices when the
model called tools in a different order than they were defined, or called
the same tool multiple times.
"""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.hermes_detector import HermesDetector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(5, "base-a-test-cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(name):
    return Tool(
        type="function",
        function=Function(
            name=name,
            description=f"{name} tool",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "number"}},
                "required": ["x"],
            },
        ),
    )


def _tools():
    """Define tools in order: add (index 0), multiply (index 1), subtract (index 2)."""
    return [_make_tool("add"), _make_tool("multiply"), _make_tool("subtract")]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestParseBaseJsonCallIndex(unittest.TestCase):
    """parse_base_json must assign sequential call positions, not tools-list positions."""

    def setUp(self):
        self.detector = HermesDetector()

    def _parse(self, action, tools=None):
        if tools is None:
            tools = _tools()
        return self.detector.parse_base_json(action, tools)

    # -- Single call ---------------------------------------------------------

    def test_single_call_index_is_zero(self):
        action = [{"name": "multiply", "arguments": {"x": 3}}]
        items = self._parse(action)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].tool_index, 0)
        self.assertEqual(items[0].name, "multiply")

    def test_single_first_tool_index_is_zero(self):
        """First tool in list called alone → index 0, not tools-list position."""
        action = [{"name": "add", "arguments": {"x": 1}}]
        items = self._parse(action)
        self.assertEqual(items[0].tool_index, 0)

    # -- Multi-call sequential indexing -------------------------------------

    def test_two_calls_sequential_indices(self):
        """Two calls → indices 0, 1 regardless of their order in the tools list."""
        action = [
            {"name": "multiply", "arguments": {"x": 2}},
            {"name": "add", "arguments": {"x": 5}},
        ]
        items = self._parse(action)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].tool_index, 0)
        self.assertEqual(items[0].name, "multiply")
        self.assertEqual(items[1].tool_index, 1)
        self.assertEqual(items[1].name, "add")

    def test_three_calls_sequential_indices(self):
        action = [
            {"name": "subtract", "arguments": {"x": 10}},
            {"name": "add", "arguments": {"x": 3}},
            {"name": "multiply", "arguments": {"x": 7}},
        ]
        items = self._parse(action)
        self.assertEqual(len(items), 3)
        for expected_idx, item in enumerate(items):
            self.assertEqual(item.tool_index, expected_idx)

    def test_same_tool_called_twice_sequential(self):
        """Calling the same tool twice → indices 0 and 1."""
        action = [
            {"name": "add", "arguments": {"x": 1}},
            {"name": "add", "arguments": {"x": 2}},
        ]
        items = self._parse(action)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].tool_index, 0)
        self.assertEqual(items[1].tool_index, 1)

    def test_index_not_tools_list_position(self):
        """Regression: 'subtract' is at tools-list index 2; as the first call it must be 0."""
        action = [{"name": "subtract", "arguments": {"x": 5}}]
        items = self._parse(action)
        self.assertEqual(len(items), 1)
        # Must be 0 (call position), not 2 (tools-list position).
        self.assertEqual(items[0].tool_index, 0)

    # -- Correct argument parsing --------------------------------------------

    def test_arguments_field_parsed(self):
        action = [{"name": "add", "arguments": {"x": 42}}]
        items = self._parse(action)
        self.assertEqual(json.loads(items[0].parameters), {"x": 42})

    def test_parameters_field_parsed(self):
        action = [{"name": "add", "parameters": {"x": 7}}]
        items = self._parse(action)
        self.assertEqual(json.loads(items[0].parameters), {"x": 7})


if __name__ == "__main__":
    unittest.main()
