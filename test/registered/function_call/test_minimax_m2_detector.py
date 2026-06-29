import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.minimax_m2 import MinimaxM2Detector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1.0, "default")


class TestMinimaxM2DetectorBugs(unittest.TestCase):
    """
    Test suite to reproduce and fix known bugs in MinimaxM2Detector.

    Related issues:
    - https://github.com/sgl-project/sglang/issues/16057
    """

    def setUp(self):
        """Set up common test fixtures."""
        self.detector = MinimaxM2Detector()

        # Tool with union type (anyOf with null) - related to issue #16057
        self.tools_with_union = [
            Tool(
                type="function",
                function=Function(
                    name="the_tool",
                    description="The Tool. Call this when the user asks.",
                    parameters={
                        "type": "object",
                        "required": ["args"],
                        "properties": {
                            "args": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "description": "Args for The Tool.",
                            }
                        },
                    },
                ),
            )
        ]

        # Tool with simple string parameter
        self.tools_simple = [
            Tool(
                type="function",
                function=Function(
                    name="context_info",
                    description="Get context information",
                    parameters={
                        "type": "object",
                        "properties": {"metadata_only": {"type": "boolean"}},
                    },
                ),
            )
        ]

    # ========== Bug #1: anyOf with null incorrectly returns null ==========

    def test_anyof_with_null_should_preserve_string_value(self):
        """
        Reproduce issue #16057: anyOf [string, null] incorrectly converts "hi" to null.

        Problem: In _convert_param_value_with_types, line 148-149 uses OR logic:
            if "null" in normalized_types or value.lower() in ("null", "none", "nil"):
                return None

        This causes ANY value to return None when "null" is in the type list!

        Expected: "hi" should be preserved as string "hi"
        Actual (buggy): "hi" is converted to null
        """
        text = '<minimax:tool_call><invoke name="the_tool"><parameter name="args">hi</parameter></invoke></minimax:tool_call>'

        result = self.detector.detect_and_parse(text, self.tools_with_union)

        self.assertEqual(len(result.calls), 1, "Should detect one tool call")
        self.assertEqual(result.calls[0].name, "the_tool")

        params = json.loads(result.calls[0].parameters)

        # BUG: This currently fails because params["args"] is None instead of "hi"
        self.assertEqual(
            params["args"],
            "hi",
            "String value 'hi' should be preserved, not converted to null",
        )
        self.assertIsInstance(
            params["args"], str, "Parameter should be string type, not NoneType"
        )

    def test_anyof_with_null_should_allow_explicit_null(self):
        """
        Verify that explicit "null" values are still correctly parsed as None.
        """
        text = '<minimax:tool_call><invoke name="the_tool"><parameter name="args">null</parameter></invoke></minimax:tool_call>'

        result = self.detector.detect_and_parse(text, self.tools_with_union)

        params = json.loads(result.calls[0].parameters)
        self.assertIsNone(params["args"], "Explicit 'null' should be parsed as None")

    # ========== Bug #2: Tool call content leaks into normal_text ==========

    def test_incomplete_tool_call_should_not_leak_to_normal_text(self):
        """
        Reproduce content leak bug: when </minimax:tool_call> is missing,
        the entire tool call content leaks into normal_text.

        Problem: In _extract method, line 470:
            if e == -1:
                normal_parts.append(text[s:])  # BUG: Adds tool call to normal text!

        Expected: Only text before <minimax:tool_call> should be in normal_text
        Actual (buggy): Everything from <minimax:tool_call> onwards leaks into normal_text
        """
        # Incomplete tool call (missing </minimax:tool_call>)
        text = (
            '[Pasted ~4 lines]<minimax:tool_call><invoke name="context_info"><parameter name="metadata_only">false</parameter></invoke>'
            # Missing: </minimax:tool_call>
        )

        result = self.detector.detect_and_parse(text, self.tools_simple)

        # BUG: Currently normal_text contains the tool call markers
        self.assertNotIn(
            "<minimax:tool_call>",
            result.normal_text,
            "Tool call start tag should NOT leak into normal_text",
        )
        self.assertNotIn(
            "<invoke",
            result.normal_text,
            "Tool call invoke tag should NOT leak into normal_text",
        )
        self.assertNotIn(
            "<parameter",
            result.normal_text,
            "Tool call parameter tag should NOT leak into normal_text",
        )

        # Normal text should ONLY contain the text before tool call
        self.assertEqual(
            result.normal_text.strip(),
            "[Pasted ~4 lines]",
            "Normal text should only include text before tool call marker",
        )

    def test_complete_tool_call_should_not_leak_end_marker(self):
        """
        Verify that even complete tool calls don't leak end markers to normal text.
        """
        text = 'Here is some text.<minimax:tool_call><invoke name="context_info"><parameter name="metadata_only">true</parameter></invoke></minimax:tool_call>More text after.'

        result = self.detector.detect_and_parse(text, self.tools_simple)

        # Should properly extract normal text before and after
        self.assertNotIn("</minimax:tool_call>", result.normal_text)
        self.assertIn("Here is some text.", result.normal_text)
        self.assertIn("More text after.", result.normal_text)

    # ========== Bug #3: Special characters in parameter values ==========

    def test_parameter_value_with_closing_tag_substring(self):
        """
        Test parameter values containing substrings that look like closing tags.

        Problem: Regex pattern `(.*?)</parameter>` uses non-greedy match,
        which fails if parameter value contains "</parameter>" substring.
        """
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="test_func",
                    parameters={
                        "type": "object",
                        "properties": {"content": {"type": "string"}},
                    },
                ),
            )
        ]

        text = '<minimax:tool_call><invoke name="test_func"><parameter name="content">This text contains </parameter> in it</parameter></invoke></minimax:tool_call>'

        result = self.detector.detect_and_parse(text, tools)

        # BUG: Currently truncates at first </parameter>
        if result.calls:
            params = json.loads(result.calls[0].parameters)
            self.assertEqual(
                params.get("content"),
                "This text contains </parameter> in it",
                "Parameter value should preserve </parameter> substring",
            )

    def test_parameter_value_with_xml_content(self):
        """
        Test parameter values containing XML/HTML content.
        """
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="process_html",
                    parameters={
                        "type": "object",
                        "properties": {"html": {"type": "string"}},
                    },
                ),
            )
        ]

        text = '<minimax:tool_call><invoke name="process_html"><parameter name="html"><div>Hello</div></parameter></invoke></minimax:tool_call>'

        result = self.detector.detect_and_parse(text, tools)

        if result.calls:
            params = json.loads(result.calls[0].parameters)
            self.assertEqual(
                params.get("html"),
                "<div>Hello</div>",
                "XML/HTML content in parameters should be preserved",
            )

    # ========== Bug #4: Streaming state management ==========

    def test_streaming_incomplete_tool_call_should_buffer(self):
        """
        Test that streaming properly buffers incomplete tool calls.

        When tool call is incomplete (no end marker), streaming should:
        1. NOT emit the incomplete tool call
        2. NOT leak tool call content to normal_text
        3. Buffer the content for next chunk
        """
        chunks = [
            "Normal text before",
            "<minimax:tool_call>",
            '<invoke name="context_info">',
            '<parameter name="metadata_only">false</parameter>',
            # Stream ends here without </invoke> and </minimax:tool_call>
        ]

        detector = MinimaxM2Detector()  # Fresh instance for streaming
        all_normal_text = ""
        all_calls = []

        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools_simple)
            all_normal_text += result.normal_text
            all_calls.extend(result.calls)

        # Should only have normal text from before tool call
        self.assertEqual(
            all_normal_text.strip(),
            "Normal text before",
            "Only text before tool call should be emitted",
        )

        # Should NOT have any markers in normal text
        self.assertNotIn("<minimax:tool_call>", all_normal_text)
        self.assertNotIn("<invoke", all_normal_text)
        self.assertNotIn("<parameter", all_normal_text)

    def test_streaming_complete_tool_call_emits_properly(self):
        """
        Test that complete streaming tool calls are properly emitted.
        """
        chunks = [
            "Text before ",
            "<minimax:tool_call>",
            '<invoke name="context_info">',
            '<parameter name="metadata_only">true</parameter>',
            "</invoke>",
            "</minimax:tool_call>",
            " Text after",
        ]

        detector = MinimaxM2Detector()
        all_normal_text = ""
        all_calls = []

        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools_simple)
            all_normal_text += result.normal_text
            all_calls.extend(result.calls)

        # Should emit complete tool call
        func_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(func_calls), 1)
        self.assertEqual(func_calls[0].name, "context_info")

        # Should have normal text before and after, but NOT tool call markers
        self.assertIn("Text before", all_normal_text)
        self.assertIn("Text after", all_normal_text)
        self.assertNotIn("<minimax:tool_call>", all_normal_text)
        self.assertNotIn("</minimax:tool_call>", all_normal_text)

    # ========== Complex type handling ==========

    def test_complex_anyof_with_object_and_null(self):
        """
        Test complex anyOf schema with object and null types.
        """
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="complex_func",
                    parameters={
                        "type": "object",
                        "properties": {
                            "data": {
                                "anyOf": [
                                    {
                                        "type": "object",
                                        "properties": {"key": {"type": "string"}},
                                    },
                                    {"type": "null"},
                                ]
                            }
                        },
                    },
                ),
            )
        ]

        # Test with object value
        text = '<minimax:tool_call><invoke name="complex_func"><parameter name="data">{"key": "value"}</parameter></invoke></minimax:tool_call>'

        result = self.detector.detect_and_parse(text, tools)
        params = json.loads(result.calls[0].parameters)

        self.assertIsInstance(params["data"], dict)
        self.assertEqual(params["data"]["key"], "value")

    def test_array_type_in_parameters(self):
        """
        Test that array types are properly handled.
        """
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="array_func",
                    parameters={
                        "type": "object",
                        "properties": {"items": {"type": "array"}},
                    },
                ),
            )
        ]

        text = '<minimax:tool_call><invoke name="array_func"><parameter name="items">[1, 2, 3, "four"]</parameter></invoke></minimax:tool_call>'

        result = self.detector.detect_and_parse(text, tools)
        params = json.loads(result.calls[0].parameters)

        self.assertIsInstance(params["items"], list)
        self.assertEqual(params["items"], [1, 2, 3, "four"])


if __name__ == "__main__":
    unittest.main()
