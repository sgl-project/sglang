import json

"""
Test case for parallel tool call parsing.

This test verifies that the parser correctly handles parallel tool calls
with array parameters in JSON array format.

Scenario:
- Model outputs two parallel tool calls in JSON array format
- Both tools have array parameters (e.g., "title": ["7.8.9 H-9 ..."])
- First tool completes with closing braces
- Second tool starts with opening brace
- The parser must correctly handle the '[' characters in array parameters
  without confusing them with the JSON array start

Expected behavior: Both tools should be parsed correctly.
"""

import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.json_array_parser import JsonArrayParser
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1.0, "default")


class TestParallelToolCalls(unittest.TestCase):
    """Test case for parallel tool call parsing with array parameters."""

    def setUp(self):
        """Set up test tools and detector."""
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="search_docs",
                    description="Search documents",
                    parameters={
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Document title",
                            }
                        },
                        "required": ["title"],
                    },
                ),
            ),
        ]
        self.detector = JsonArrayParser()

    def _accumulate_tool_calls(self, tool_calls, result):
        """Helper method to accumulate tool call results from parsing output."""
        if not result.calls:
            return
        for call in result.calls:
            if call.tool_index is None:
                continue
            while len(tool_calls) <= call.tool_index:
                tool_calls.append({"name": "", "parameters": ""})
            if call.name:
                tool_calls[call.tool_index]["name"] = call.name
            if call.parameters:
                tool_calls[call.tool_index]["parameters"] += call.parameters

    def test_parallel_tool_calls_with_array_parameters(self):
        """
        Test parsing two parallel tool calls where both have array parameters.

        This test reproduces the specific scenario:
        - Two tool calls separated by comma
        - Both tools have array parameters containing '[' character
        - First tool completes with '}},'
        - Second tool starts with '{"name": ..., "parameters": {"title": ["'

        Expected: Both tools should be parsed correctly without errors.
        """
        # Simulate more realistic streaming chunks where
        # the key issue is the comma separator followed by second tool with array param
        chunks = [
            "[\n",
            '  {"name": "search_docs", "parameters": {"title": ["7.8.9"',
            '], "filename": "doc1"}},\n',
            '  {"name": "search_docs", "parameters": {"title": ',
            '["4.8"], "filename": "doc2"}}',
            "]",
        ]

        tool_calls = []
        errors = []

        for i, chunk in enumerate(chunks):
            try:
                result = self.detector.parse_streaming_increment(chunk, self.tools)
                # Collect tool calls
                self._accumulate_tool_calls(tool_calls, result)

            except Exception as e:
                errors.append(f"Chunk {i} ({repr(chunk)}): {type(e).__name__}: {e}")

        # Verify no errors occurred
        if errors:
            self.fail("Errors occurred during parsing:\n" + "\n".join(errors))

        # Verify both tool calls were parsed
        self.assertEqual(len(tool_calls), 2, "Should have parsed exactly 2 tool calls")

        # Verify first tool call
        self.assertEqual(
            tool_calls[0]["name"],
            "search_docs",
            "First tool name should be search_docs",
        )
        params1 = json.loads(tool_calls[0]["parameters"])
        self.assertEqual(params1["title"], ["7.8.9"], "First tool title should match")
        self.assertEqual(
            params1["filename"], "doc1", "First tool filename should be doc1"
        )

        # Verify second tool call
        self.assertEqual(
            tool_calls[1]["name"],
            "search_docs",
            "Second tool name should be search_docs",
        )
        params2 = json.loads(tool_calls[1]["parameters"])
        self.assertEqual(params2["title"], ["4.8"], "Second tool title should match")
        self.assertEqual(
            params2["filename"], "doc2", "Second tool filename should be doc2"
        )

    def test_simple_parallel_tool_calls(self):
        """
        Test a simpler case of two parallel tool calls with array parameters.

        This is a minimal test case that still tests the core functionality.
        """
        chunks = [
            "[\n",
            '  {"name": "search_docs", "parameters": {"title": ["a"]}},',
            "\n",
            '  {"name": "search_docs", "parameters": {"title": ["b"]}}',
            "]",
        ]

        tool_calls = []

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            self._accumulate_tool_calls(tool_calls, result)

        # Should parse both tools successfully
        self.assertEqual(len(tool_calls), 2, "Should parse 2 tool calls")
        self.assertEqual(tool_calls[0]["name"], "search_docs")
        self.assertEqual(tool_calls[1]["name"], "search_docs")


if __name__ == "__main__":
    unittest.main()
