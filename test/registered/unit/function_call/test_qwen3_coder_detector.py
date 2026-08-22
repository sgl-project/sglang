"""Unit tests for Qwen3CoderDetector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import (
    Function,
    Tool,
    ToolChoice,
    ToolChoiceFuncName,
)
from sglang.srt.function_call.core_types import StreamingParseResult
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestQwen3CoderDetector(CustomTestCase):
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
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
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
            Tool(
                type="function",
                function=Function(
                    name="get_current_time",
                    parameters={
                        "type": "object",
                        "properties": {
                            "cities": {
                                "anyOf": [
                                    {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    {"type": "null"},
                                ],
                                "default": None,
                            }
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

    def test_anyof_array_parameter_conversion(self):
        """
        Test array parameter conversion for nullable anyOf schemas.

        Scenario: A Pydantic-style nullable list schema is represented by anyOf.
        Purpose: Verify array values are parsed as arrays, not JSON-looking strings.
        """
        text = """<tool_call>
<function=get_current_time>
<parameter=cities>
["NYC"]
</parameter>
</function>
</tool_call>"""
        result = self.detector.detect_and_parse(text, self.tools)

        params = json.loads(result.calls[0].parameters)
        self.assertIsInstance(params["cities"], list)
        self.assertEqual(params["cities"], ["NYC"])

    def test_anyof_array_parameter_conversion_null(self):
        """
        Test 'null' is converted correctly for nullable anyOf schemas.

        Scenario: A Pydantic-style nullable list schema is represented by anyOf.
        Purpose: Verify null values are parsed as 'None', not as strings.
        """
        text = """<tool_call>
<function=get_current_time>
<parameter=cities>
null
</parameter>
</function>
</tool_call>"""
        result = self.detector.detect_and_parse(text, self.tools)

        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["cities"], None)

    def test_streaming_anyof_array_parameter_conversion(self):
        """
        Test streaming array parameter conversion for nullable anyOf schemas.

        Scenario: A Pydantic-style nullable list schema is streamed in Qwen3 Coder format.
        Purpose: Verify the streamed JSON fragments encode an array value, not a string value.
        """
        chunks = [
            "<tool_call>",
            "<function=get_current_time>",
            "<parameter=cities>",
            '["NYC"]',
            "</parameter>",
            "</function>",
            "</tool_call>",
        ]

        detector = Qwen3CoderDetector()
        collected_params = ""

        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            for call in result.calls:
                if call.parameters:
                    collected_params += call.parameters

        params = json.loads(collected_params)
        self.assertIsInstance(params["cities"], list)
        self.assertEqual(params["cities"], ["NYC"])

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

    def test_nested_anyof_array_with_multiple_types_parameter_conversion(self):
        """
        Test several edge cases of parameter conversion for nullable anyOf schemas.
        1) Test nested anyOf 'T | None' extracts 'T' correctly.
        2) Test that order of null and non-null type doesn't affect schema parsing.
        3) Test that list of multiple types (including dict) is parsed correctly.
        """

        tool = Tool(
            type="function",
            function=Function(
                name="process_optional_list",
                parameters={
                    "type": "object",
                    "properties": {
                        "optional_items_to_process": {
                            "anyOf": [
                                {
                                    "anyOf": [
                                        # Note: here "null" is listed before the non-null type.
                                        {"type": "null"},
                                        {
                                            "anyOf": [
                                                {
                                                    "type": "array",
                                                    "items": {},
                                                },
                                                {"type": "null"},
                                            ],
                                        },
                                    ],
                                },
                                {"type": "null"},
                            ],
                            "default": None,
                        }
                    },
                },
            ),
        )

        text = """<tool_call>
<function=process_optional_list>
<parameter=optional_items_to_process>
[true, null, {"enabled": false}]
</parameter>
</function>
</tool_call>"""

        result = self.detector.detect_and_parse(text, [tool])

        params = json.loads(result.calls[0].parameters)
        self.assertIsInstance(params["optional_items_to_process"], list)
        self.assertEqual(
            params["optional_items_to_process"], [True, None, {"enabled": False}]
        )

    # ==================== Structural tag (xgrammar builtin) ====================
    # Qwen3 Coder uses the new builtin structural tag path. supports_structural_tag()
    # is True so required/named tool_choice routes through FunctionCallParser
    # instead of JsonArrayParser.

    def test_supports_structural_tag(self):
        self.assertTrue(self.detector.supports_structural_tag())

    def test_get_model_structural_tag(self):
        import xgrammar as xgr

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=True
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=False
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=True, tool_choice="required"
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=False, tool_choice="required"
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

        tool_choice_name = ToolChoiceFuncName(name="get_current_weather")
        tool_choice = ToolChoice(function=tool_choice_name)
        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=True, tool_choice=tool_choice
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=False, tool_choice=tool_choice
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)


if __name__ == "__main__":
    import unittest

    unittest.main()
