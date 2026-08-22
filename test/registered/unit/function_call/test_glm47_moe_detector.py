"""Unit tests for Glm47MoeDetector — no server, no model loading."""

import json
import warnings

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.glm47_moe_detector import Glm47MoeDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestGlm47MoeDetector(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "date": {"type": "string", "description": "Date"},
                        },
                        "required": ["city", "date"],
                    },
                ),
            ),
        ]
        self.detector = Glm47MoeDetector()

    def test_multiple_tool_calls(self):
        text = (
            "<tool_call>get_weather"
            "<arg_key>city</arg_key><arg_value>Beijing</arg_value>"
            "<arg_key>date</arg_key><arg_value>2024-06-27</arg_value>"
            "</tool_call>"
            "<tool_call>get_weather"
            "<arg_key>city</arg_key><arg_value>Shanghai</arg_value>"
            "<arg_key>date</arg_key><arg_value>2024-06-28</arg_value>"
            "</tool_call>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(
            result.calls[0].parameters, '{"city": "Beijing", "date": "2024-06-27"}'
        )
        self.assertEqual(result.calls[1].name, "get_weather")
        self.assertEqual(
            result.calls[1].parameters, '{"city": "Shanghai", "date": "2024-06-28"}'
        )
        self.assertEqual(result.normal_text, "")

    def test_streaming_multiple_tool_calls(self):
        """Test streaming incremental parsing of multiple tool calls."""
        chunks = [
            "<tool_call>get_weather",
            "<arg_key>city</arg_key><arg_value>Beijing</arg_value>",
            "<arg_key>date</arg_key><arg_value>2024-06-27</arg_value>",
            "</tool_call><tool_call>get_weather",
            "<arg_key>city</arg_key><arg_value>Shanghai</arg_value>",
            "<arg_key>date</arg_key><arg_value>2024-06-28</arg_value>",
            "</tool_call>",
        ]
        tool_calls = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            for tool_call_chunk in result.calls:
                if (
                    hasattr(tool_call_chunk, "tool_index")
                    and tool_call_chunk.tool_index is not None
                ):
                    while len(tool_calls) <= tool_call_chunk.tool_index:
                        tool_calls.append({"name": "", "parameters": ""})
                    tc = tool_calls[tool_call_chunk.tool_index]
                    if tool_call_chunk.name:
                        tc["name"] = tool_call_chunk.name
                    if tool_call_chunk.parameters:
                        tc["parameters"] += tool_call_chunk.parameters
        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(tool_calls[0]["name"], "get_weather")
        self.assertEqual(
            tool_calls[0]["parameters"], '{"city": "Beijing", "date": "2024-06-27"}'
        )
        self.assertEqual(tool_calls[1]["name"], "get_weather")
        self.assertEqual(
            tool_calls[1]["parameters"], '{"city": "Shanghai", "date": "2024-06-28"}'
        )

    def test_tool_call_id(self):
        """Test that the buffer and state are reset after a tool call is completed."""
        chunks = [
            "<tool_call>get_weather",
            "<arg_key>city</arg_key><arg_value>Beijing</arg_value>",
            "<arg_key>date</arg_key><arg_value>2024-06-27</arg_value>",
            "</tool_call>",
        ]
        for chunk in chunks:
            self.detector.parse_streaming_increment(chunk, self.tools)
        self.assertEqual(self.detector.current_tool_id, 1)

    def test_invalid_tool_call(self):
        """Test that invalid tool calls are handled correctly."""
        text = "<tool_call>invalid_func<arg_key>city</arg_key><arg_value>Beijing</arg_value></tool_call>"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)

    def test_partial_tool_call(self):
        """Test parsing a partial tool call that spans multiple chunks."""
        chunks = [
            "<tool_call>get_weather",
            "<arg_key>city</arg_key><arg_value>Beijing</arg_value>",
            "<arg_key>date</arg_key><arg_value>2024-06-27</arg_value></tool_call>",
        ]

        tool_calls = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            for tool_call_chunk in result.calls:
                if (
                    hasattr(tool_call_chunk, "tool_index")
                    and tool_call_chunk.tool_index is not None
                ):
                    while len(tool_calls) <= tool_call_chunk.tool_index:
                        tool_calls.append({"name": "", "parameters": ""})
                    tc = tool_calls[tool_call_chunk.tool_index]
                    if tool_call_chunk.name:
                        tc["name"] = tool_call_chunk.name
                    if tool_call_chunk.parameters:
                        tc["parameters"] += tool_call_chunk.parameters

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["name"], "get_weather")
        self.assertEqual(
            tool_calls[0]["parameters"], '{"city": "Beijing", "date": "2024-06-27"}'
        )

    def test_array_argument_with_escaped_json(self):
        """Test that array arguments with escaped JSON are properly handled without double-escaping."""
        # Add a tool with array parameter
        tools_with_array = [
            Tool(
                type="function",
                function=Function(
                    name="todo_write",
                    description="Write todos",
                    parameters={
                        "type": "object",
                        "properties": {
                            "todos": {
                                "type": "array",
                                "description": "The updated todo list",
                            }
                        },
                        "required": ["todos"],
                    },
                ),
            ),
        ]

        def check_params(result):
            self.assertEqual(1, len(result.calls))
            self.assertEqual("todo_write", result.calls[0].name)
            params = json.loads(result.calls[0].parameters)
            self.assertIsInstance(params["todos"], list)
            self.assertEqual(4, len(params["todos"]))
            self.assertEqual("1", params["todos"][0]["id"])
            self.assertEqual(
                "Check for hard-coded issues in the backend code",
                params["todos"][0]["task"],
            )
            self.assertEqual("in_progress", params["todos"][0]["status"])
            self.assertEqual("2", params["todos"][1]["id"])
            self.assertEqual(
                "Check for hard-coded issues in the frontend code",
                params["todos"][1]["task"],
            )
            self.assertEqual("pending", params["todos"][1]["status"])
            self.assertEqual("3", params["todos"][2]["id"])
            self.assertEqual(
                "Check for code violating the Single Responsibility Principle",
                params["todos"][2]["task"],
            )
            self.assertEqual("pending", params["todos"][2]["status"])
            self.assertEqual("4", params["todos"][3]["id"])
            self.assertEqual(
                "Generate a rectification proposal report", params["todos"][3]["task"]
            )
            self.assertEqual("pending", params["todos"][3]["status"])

        # Simulate the raw response from GLM-4.6 model with normal and escaped JSON in XML
        result = self.detector.detect_and_parse(
            """<tool_call>todo_write<arg_key>todos</arg_key><arg_value>[{\"id\": \"1\", \"task\": \"Check for hard-coded issues in the backend code\", \"status\": \"in_progress\"}, {\"id\": \"2\", \"task\": \"Check for hard-coded issues in the frontend code\", \"status\": \"pending\"}, {\"id\": \"3\", \"task\": \"Check for code violating the Single Responsibility Principle\", \"status\": \"pending\"}, {\"id\": \"4\", \"task\": \"Generate a rectification proposal report\", \"status\": \"pending\"}]</arg_value>
</tool_call>""",
            tools_with_array,
        )
        check_params(result)
        result = self.detector.detect_and_parse(
            r"""<tool_call>todo_write<arg_key>todos</arg_key><arg_value>[{\"id\": \"1\", \"task\": \"Check for hard-coded issues in the backend code\", \"status\": \"in_progress\"}, {\"id\": \"2\", \"task\": \"Check for hard-coded issues in the frontend code\", \"status\": \"pending\"}, {\"id\": \"3\", \"task\": \"Check for code violating the Single Responsibility Principle\", \"status\": \"pending\"}, {\"id\": \"4\", \"task\": \"Generate a rectification proposal report\", \"status\": \"pending\"}]</arg_value>
</tool_call>""",
            tools_with_array,
        )
        check_params(result)

        def check_single_todos(tool_result, expected):
            self.assertEqual(1, len(tool_result.calls))
            self.assertEqual("todo_write", tool_result.calls[0].name)
            params = json.loads(tool_result.calls[0].parameters)
            self.assertIsInstance(params["todos"], list)
            self.assertEqual(1, len(params["todos"]))
            self.assertEqual("1", params["todos"][0]["id"])
            self.assertEqual(expected, params["todos"][0]["task"])
            self.assertEqual("pending", params["todos"][0]["status"])

        # Test with escaped JSON containing backslashes in content (e.g., Windows paths)
        expected_path = r"Check file at C:\Users\test.txt"
        result = self.detector.detect_and_parse(
            """<tool_call>todo_write<arg_key>todos</arg_key><arg_value>[{\"id\": \"1\", \"task\": \"Check file at C:\\\\Users\\\\test.txt\", \"status\": \"pending\"}]</arg_value></tool_call>""",
            tools_with_array,
        )
        check_single_todos(result, expected_path)
        result = self.detector.detect_and_parse(
            r"""<tool_call>todo_write<arg_key>todos</arg_key><arg_value>[{\"id\": \"1\", \"task\": \"Check file at C:\\\\Users\\\\test.txt\", \"status\": \"pending\"}]</arg_value></tool_call>""",
            tools_with_array,
        )
        check_single_todos(result, expected_path)

        # Should contain literal \n, not actual newline
        expected_output = r"Print \n to see newline"
        result = self.detector.detect_and_parse(
            """<tool_call>todo_write<arg_key>todos</arg_key><arg_value>[{\"id\": \"1\", \"task\": \"Print \\\\n to see newline\",\"status\": \"pending\"}]</arg_value></tool_call>""",
            tools_with_array,
        )
        check_single_todos(result, expected_output)
        result = self.detector.detect_and_parse(
            r"""<tool_call>todo_write<arg_key>todos</arg_key><arg_value>[{\"id\": \"1\", \"task\": \"Print \\\\n to see newline\",\"status\": \"pending\"}]</arg_value></tool_call>""",
            tools_with_array,
        )
        check_single_todos(result, expected_output)

    def test_whitespace_preserved_in_arg_values(self):
        """Test that leading/trailing whitespace in arg values is not stripped."""
        tools_with_string = [
            Tool(
                type="function",
                function=Function(
                    name="apply_diff",
                    description="Apply a diff",
                    parameters={
                        "type": "object",
                        "properties": {
                            "old_string": {"type": "string"},
                            "new_string": {"type": "string"},
                        },
                        "required": ["old_string", "new_string"],
                    },
                ),
            )
        ]
        text = (
            "<tool_call>apply_diff"
            "<arg_key>old_string</arg_key>"
            "<arg_value>    indented code</arg_value>"
            "<arg_key>new_string</arg_key>"
            "<arg_value>        also indented</arg_value>"
            "</tool_call>"
        )
        result = self.detector.detect_and_parse(text, tools_with_string)
        self.assertEqual(len(result.calls), 1)
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["old_string"], "    indented code")
        self.assertEqual(params["new_string"], "        also indented")

    def test_quoted_string_invalid_python_escape_no_warning(self):
        text = (
            '<tool_call>get_weather<arg_key>city</arg_key><arg_value>"\\C|\\."</arg_value>'
            "<arg_key>date</arg_key><arg_value>2024-06-27</arg_value></tool_call>"
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", SyntaxWarning)
            result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["city"], r"\C|\.")
        self.assertFalse(
            any(isinstance(w.message, SyntaxWarning) for w in caught),
            [str(w.message) for w in caught],
        )

    def test_parse_arguments_preserves_underscore_in_string_args(self):
        """Same PEP 515 guard as the GLM-4 detector, on the GLM-4.7 parser."""
        from sglang.srt.function_call.glm47_moe_detector import parse_arguments

        value, is_good = parse_arguments("123_456", arg_type="string")
        self.assertTrue(is_good)
        self.assertIsInstance(value, str)
        self.assertEqual(value, "123_456")

        value, is_good = parse_arguments("1_000.5", arg_type="string")
        self.assertTrue(is_good)
        self.assertIsInstance(value, str)
        self.assertEqual(value, "1_000.5")

        value, is_good = parse_arguments("123_456")
        self.assertTrue(is_good)
        self.assertIsInstance(value, int)
        self.assertEqual(value, 123456)

    def test_parse_arguments_object_with_invalid_escape(self):
        """Same object-arg escape guard as the GLM-4 detector."""
        from sglang.srt.function_call.glm47_moe_detector import parse_arguments

        value, is_good = parse_arguments("{'pattern': '\\d+'}", arg_type="object")
        self.assertTrue(is_good)
        self.assertIsInstance(value, dict)
        self.assertEqual(value, {"pattern": "\\d+"})

    def test_get_model_structural_tag(self):
        """GLM-4.7/GLM-5 use xgrammar's native "glm_4_7" structural tag."""
        import xgrammar as xgr

        self.assertTrue(self.detector.supports_structural_tag())
        self.assertEqual(self.detector.get_structural_tag_name(), "glm_4_7")

        # thinking_mode=True keeps the </think> reasoning prefix.
        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=True
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        self.assertIsInstance(
            xgr.Grammar.from_structural_tag(structural_tag), xgr.Grammar
        )
        serialized = structural_tag.model_dump_json()
        self.assertIn("glm_xml", serialized)
        self.assertIn("<tool_call>", serialized)
        self.assertIn("</think>", serialized)

        # thinking_mode=False drops the reasoning prefix (ReasonerGrammarBackend
        # owns </think> when --reasoning-parser is configured).
        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=False
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        self.assertIsInstance(
            xgr.Grammar.from_structural_tag(structural_tag), xgr.Grammar
        )
        self.assertNotEqual("sequence", structural_tag.model_dump()["format"]["type"])

        # tool_choice="required" must still compile to a grammar.
        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=True, tool_choice="required"
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        self.assertIsInstance(
            xgr.Grammar.from_structural_tag(structural_tag), xgr.Grammar
        )

    def test_required_tool_choice_falls_back_when_native_tag_is_unavailable(self):
        from unittest.mock import patch

        from sglang.srt.function_call.function_call_parser import FunctionCallParser
        from sglang.srt.function_call.glm47_moe_detector import (
            _glm47_native_structural_tag_available,
        )

        with patch(
            "sglang.srt.function_call.glm47_moe_detector.get_model_structural_tag",
            None,
        ):
            _glm47_native_structural_tag_available.cache_clear()
            self.assertFalse(self.detector.supports_structural_tag())
            self.assertIsNone(self.detector.get_structural_tag(self.tools))

            parser = FunctionCallParser(self.tools, "glm47")
            constraint = parser.get_structure_constraint("required")

            self.assertIsNotNone(constraint)
            self.assertEqual("json_schema", constraint[0])
            _glm47_native_structural_tag_available.cache_clear()


if __name__ == "__main__":
    import unittest

    unittest.main()
