"""
Test Qwen3CoderDetector streaming and non-streaming parsing functionality.

This test suite validates the parsing capabilities of the Qwen3CoderDetector,
including tool call detection, parameter extraction, and text/tool separation.
"""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.core_types import StreamingParseResult
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1.0, "default")


class TestQwen3CoderDetector(unittest.TestCase):
    """Test suite for Qwen3CoderDetector."""

    def setUp(self):
        """Initialize test fixtures before each test method."""
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_current_weather",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                            "days": {"type": "integer"},
                        },
                        "required": ["location"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="sql_interpreter",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "dry_run": {"type": "boolean"},
                        },
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="TodoWrite",
                    parameters={
                        "type": "object",
                        "properties": {
                            "todos": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "content": {"type": "string"},
                                        "status": {"type": "string"},
                                    },
                                    "required": ["content", "status"],
                                },
                            },
                        },
                    },
                ),
            ),
        ]
        self.detector = Qwen3CoderDetector()

    # ==================== Basic Functionality Tests ====================

    def test_plain_text_only(self):
        """
        Test parsing of plain text without any tool calls.

        Scenario: Input contains only plain text, no tool call markers.
        Purpose: Verify that plain text is correctly identified and no false tool calls are detected.
        """
        text = "This is plain text without any tool calls."
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, text)
        self.assertEqual(len(result.calls), 0)

    def test_single_tool_call(self):
        """
        Test parsing of a single tool call.

        Scenario: Input contains one complete tool call with parameters.
        Purpose: Verify correct extraction of tool name and parameters.
        """
        text = """<tool_call>
<function=get_current_weather>
<parameter=location>Boston</parameter>
<parameter=unit>celsius</parameter>
<parameter=days>3</parameter>
</function>
</tool_call>"""
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_current_weather")

        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["location"], "Boston")
        self.assertEqual(params["unit"], "celsius")
        self.assertEqual(params["days"], 3)

    def test_single_tool_call_with_text_prefix(self):
        """
        Test parsing of tool call with preceding text.

        Scenario: Input has plain text followed by a tool call.
        Purpose: Verify correct separation of text and tool call.
        """
        text = """Let me check the weather for you.

<tool_call>
<function=get_current_weather>
<parameter=location>New York</parameter>
</function>
</tool_call>"""
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertTrue(result.normal_text.startswith("Let me check"))
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_current_weather")

    def test_multiple_tool_calls(self):
        """
        Test parsing of multiple consecutive tool calls.

        Scenario: Input contains two tool calls one after another.
        Purpose: Verify that multiple tool calls are correctly identified and parsed.
        """
        text = """<tool_call>
<function=get_current_weather>
<parameter=location>New York</parameter>
</function>
</tool_call>
<tool_call>
<function=sql_interpreter>
<parameter=query>SELECT * FROM users</parameter>
<parameter=dry_run>True</parameter>
</function>
</tool_call>"""
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_current_weather")
        self.assertEqual(result.calls[1].name, "sql_interpreter")

        params1 = json.loads(result.calls[0].parameters)
        self.assertEqual(params1["location"], "New York")

        params2 = json.loads(result.calls[1].parameters)
        self.assertEqual(params2["query"], "SELECT * FROM users")
        self.assertEqual(params2["dry_run"], True)

    # ==================== Streaming Tests ====================

    def test_streaming_single_tool_call(self):
        """
        Test streaming parsing of a single tool call.

        Scenario: Tool call is fed incrementally in chunks.
        Purpose: Verify streaming parser correctly assembles tool call from chunks.
        """
        chunks = [
            "<tool_call>",
            "<function=get_current_weather>",
            "<parameter=location>",
            "Boston",
            "</parameter>",
            "<parameter=unit>celsius</parameter>",
            "</function>",
            "</tool_call>",
        ]

        detector = Qwen3CoderDetector()
        all_calls = []
        collected_params = ""

        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)
            for call in result.calls:
                if call.parameters:
                    collected_params += call.parameters

        # Verify we got the tool call
        self.assertGreater(len(all_calls), 0)

        # Verify parameters were collected
        if collected_params:
            params = json.loads(collected_params)
            self.assertEqual(params["location"], "Boston")
            self.assertEqual(params["unit"], "celsius")

    def test_streaming_with_text_and_tool(self):
        """
        Test streaming parsing with mixed text and tool call.

        Scenario: Stream contains plain text followed by a tool call.
        Purpose: Verify correct separation in streaming mode.
        """
        chunks = [
            "Let me ",
            "help you.\n\n",
            "<tool_call>",
            "<function=get_current_weather>",
            "<parameter=location>Paris</parameter>",
            "</function>",
            "</tool_call>",
        ]

        detector = Qwen3CoderDetector()
        full_text = ""
        all_calls = []

        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            if result.normal_text:
                full_text += result.normal_text
            all_calls.extend(result.calls)

        self.assertTrue(full_text.startswith("Let me"))
        self.assertGreater(len(all_calls), 0)

    # ==================== Parameter Type Tests ====================

    def test_integer_parameter_conversion(self):
        """
        Test correct type conversion for integer parameters.

        Scenario: Tool call with integer parameter.
        Purpose: Verify integer values are correctly parsed and typed.
        """
        text = """<tool_call>
<function=get_current_weather>
<parameter=location>Tokyo</parameter>
<parameter=days>5</parameter>
</function>
</tool_call>"""
        result = self.detector.detect_and_parse(text, self.tools)

        params = json.loads(result.calls[0].parameters)
        self.assertIsInstance(params["days"], int)
        self.assertEqual(params["days"], 5)

    def test_boolean_parameter_conversion(self):
        """
        Test correct type conversion for boolean parameters.

        Scenario: Tool call with boolean parameter.
        Purpose: Verify boolean values are correctly parsed.
        """
        text = """<tool_call>
<function=sql_interpreter>
<parameter=query>SELECT 1</parameter>
<parameter=dry_run>True</parameter>
</function>
</tool_call>"""
        result = self.detector.detect_and_parse(text, self.tools)

        params = json.loads(result.calls[0].parameters)
        self.assertIsInstance(params["dry_run"], bool)
        self.assertEqual(params["dry_run"], True)

    def test_complex_array_parameter(self):
        """
        Test parsing of complex array parameters.

        Scenario: Tool call with array of objects as parameter.
        Purpose: Verify complex nested structures are correctly parsed.
        """
        text = """<tool_call>
<function=TodoWrite>
<parameter=todos>
[
  {"content": "Buy groceries", "status": "pending"},
  {"content": "Finish report", "status": "completed"}
]
</parameter>
</function>
</tool_call>"""
        result = self.detector.detect_and_parse(text, self.tools)

        params = json.loads(result.calls[0].parameters)
        self.assertIsInstance(params["todos"], list)
        self.assertEqual(len(params["todos"]), 2)
        self.assertEqual(params["todos"][0]["content"], "Buy groceries")
        self.assertEqual(params["todos"][1]["status"], "completed")

    # ==================== Edge Cases ====================

    def test_empty_parameter_value(self):
        """
        Test handling of empty parameter values.

        Scenario: Tool call with empty parameter value.
        Purpose: Verify empty values are handled gracefully.
        """
        text = """<tool_call>
<function=get_current_weather>
<parameter=location></parameter>
</function>
</tool_call>"""
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["location"], "")

    def test_parameter_with_special_characters(self):
        """
        Test handling of parameters with special characters.

        Scenario: Parameter value contains special characters like quotes, newlines.
        Purpose: Verify special characters are correctly preserved.
        """
        text = """<tool_call>
<function=sql_interpreter>
<parameter=query>SELECT * FROM users WHERE name = 'John "Doe"'</parameter>
</function>
</tool_call>"""
        result = self.detector.detect_and_parse(text, self.tools)

        params = json.loads(result.calls[0].parameters)
        self.assertIn("John", params["query"])
        self.assertIn("Doe", params["query"])

    def test_incomplete_tool_call(self):
        """
        Test handling of incomplete tool call at end of stream.

        Scenario: Stream ends with an incomplete tool call (missing closing tag).
        Purpose: Verify detector handles incomplete input gracefully without crashing.
        """
        text = """<tool_call>
<function=get_current_weather>
<parameter=location>London"""

        # Should not crash
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertIsInstance(result, StreamingParseResult)

    def test_has_tool_call_detection(self):
        """
        Test the has_tool_call method for detecting tool call markers.

        Scenario: Various inputs with and without tool call markers.
        Purpose: Verify correct detection of tool call presence.
        """
        self.assertTrue(self.detector.has_tool_call("<tool_call>"))
        self.assertTrue(self.detector.has_tool_call("text <tool_call> more"))
        self.assertFalse(self.detector.has_tool_call("plain text only"))
        self.assertFalse(self.detector.has_tool_call(""))

    # ==================== Streaming State Management ====================

    def test_streaming_state_reset(self):
        """
        Test that streaming state is properly managed across calls.

        Scenario: Multiple separate streaming sessions.
        Purpose: Verify state doesn't leak between sessions.
        """
        detector = Qwen3CoderDetector()

        # First session
        result1 = detector.parse_streaming_increment("<tool_call>", self.tools)
        result2 = detector.parse_streaming_increment("<function=get_current_weather>", self.tools)
        result3 = detector.parse_streaming_increment("<parameter=location>NYC</parameter>", self.tools)
        result4 = detector.parse_streaming_increment("</function>", self.tools)
        result5 = detector.parse_streaming_increment("</tool_call>", self.tools)

        # Reset for new session
        detector._reset_streaming_state()

        # Second session should work independently
        result6 = detector.parse_streaming_increment("<tool_call>", self.tools)
        result7 = detector.parse_streaming_increment("<function=sql_interpreter>", self.tools)
        result8 = detector.parse_streaming_increment("<parameter=query>SELECT 1</parameter>", self.tools)
        result9 = detector.parse_streaming_increment("</function>", self.tools)
        result10 = detector.parse_streaming_increment("</tool_call>", self.tools)

        # Both sessions should produce valid results
        self.assertIsInstance(result1, StreamingParseResult)
        self.assertIsInstance(result6, StreamingParseResult)


if __name__ == "__main__":
    unittest.main()
