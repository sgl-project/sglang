# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for OpenAI API utility functions"""

import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.entrypoints.openai.protocol import (
    Function,
    ResponseTool,
    Tool,
)
from sglang.srt.entrypoints.openai.utils import (
    convert_response_tools_to_chat_tools,
    parse_tool_calls_from_content,
)


class TestConvertResponseToolsToChatTools(unittest.TestCase):
    """Test convert_response_tools_to_chat_tools utility function"""

    def test_convert_function_tool(self):
        """Test converting a function ResponseTool to Chat Tool"""
        function = Function(
            name="get_weather",
            description="Get weather information",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                },
                "required": ["city"],
            },
            strict=True,
        )
        response_tools = [ResponseTool(type="function", function=function)]

        result = convert_response_tools_to_chat_tools(response_tools)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], Tool)
        self.assertEqual(result[0].type, "function")
        self.assertEqual(result[0].function.name, "get_weather")
        self.assertEqual(result[0].function.description, "Get weather information")
        self.assertTrue(result[0].function.strict)

    def test_convert_multiple_function_tools(self):
        """Test converting multiple function ResponseTools"""
        func1 = Function(name="add", description="Add numbers")
        func2 = Function(name="subtract", description="Subtract numbers")
        response_tools = [
            ResponseTool(type="function", function=func1),
            ResponseTool(type="function", function=func2),
        ]

        result = convert_response_tools_to_chat_tools(response_tools)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].function.name, "add")
        self.assertEqual(result[1].function.name, "subtract")

    def test_skip_builtin_tools(self):
        """Test that built-in tools are skipped during conversion"""
        function = Function(name="custom_func", description="Custom function")
        response_tools = [
            ResponseTool(type="web_search_preview"),
            ResponseTool(type="function", function=function),
            ResponseTool(type="code_interpreter"),
        ]

        result = convert_response_tools_to_chat_tools(response_tools)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].function.name, "custom_func")

    def test_skip_function_without_definition(self):
        """Test that function tools without function definition are skipped"""
        response_tools = [
            ResponseTool(type="function"),  # No function definition
            ResponseTool(type="function", function=Function(name="valid", description="Valid")),
        ]

        result = convert_response_tools_to_chat_tools(response_tools)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].function.name, "valid")

    def test_skip_function_without_name(self):
        """Test that function tools without name are skipped"""
        # Function with empty name
        func_no_name = Function(name="", description="No name function")
        response_tools = [
            ResponseTool(type="function", function=func_no_name),
        ]

        result = convert_response_tools_to_chat_tools(response_tools)

        # Empty name is falsy, so it should be skipped
        self.assertIsNone(result)

    def test_empty_list(self):
        """Test with empty tools list"""
        result = convert_response_tools_to_chat_tools([])

        self.assertIsNone(result)

    def test_none_input(self):
        """Test with None input"""
        result = convert_response_tools_to_chat_tools(None)

        self.assertIsNone(result)

    def test_only_builtin_tools(self):
        """Test with only built-in tools (no function tools)"""
        response_tools = [
            ResponseTool(type="web_search_preview"),
            ResponseTool(type="code_interpreter"),
            ResponseTool(type="mcp"),
        ]

        result = convert_response_tools_to_chat_tools(response_tools)

        self.assertIsNone(result)

    def test_preserves_function_parameters(self):
        """Test that function parameters are preserved during conversion"""
        params = {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
            },
            "required": ["x", "y"],
        }
        function = Function(
            name="calculate",
            description="Calculate result",
            parameters=params,
            strict=False,
        )
        response_tools = [ResponseTool(type="function", function=function)]

        result = convert_response_tools_to_chat_tools(response_tools)

        self.assertIsNotNone(result)
        self.assertEqual(result[0].function.parameters, params)
        self.assertFalse(result[0].function.strict)


class TestParseToolCallsFromContent(unittest.TestCase):
    """Test parse_tool_calls_from_content utility function"""

    def test_no_tool_calls_in_content(self):
        """Test parsing content with no tool calls"""
        tools = [
            Tool(
                type="function",
                function=Function(name="get_weather", description="Get weather"),
            )
        ]

        # Mock the FunctionCallParser to return no tool calls
        with patch(
            "sglang.srt.entrypoints.openai.utils.FunctionCallParser"
        ) as MockParser:
            mock_parser_instance = MagicMock()
            mock_parser_instance.has_tool_call.return_value = False
            MockParser.return_value = mock_parser_instance

            remaining_text, tool_calls = parse_tool_calls_from_content(
                content="Hello, how are you?",
                tools=tools,
                tool_call_parser="llama3",
                generate_tool_call_id=lambda call_info, cnt: f"call_{cnt}",
            )

        self.assertEqual(remaining_text, "Hello, how are you?")
        self.assertEqual(len(tool_calls), 0)

    def test_single_tool_call_parsing(self):
        """Test parsing content with a single tool call"""
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather",
                    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
                ),
            )
        ]

        # Create mock call info
        mock_call_info = MagicMock()
        mock_call_info.name = "get_weather"
        mock_call_info.parameters = '{"city": "Tokyo"}'

        with patch(
            "sglang.srt.entrypoints.openai.utils.FunctionCallParser"
        ) as MockParser:
            mock_parser_instance = MagicMock()
            mock_parser_instance.has_tool_call.return_value = True
            mock_parser_instance.parse_non_stream.return_value = (
                "",  # remaining text
                [mock_call_info],  # call info list
            )
            MockParser.return_value = mock_parser_instance

            remaining_text, tool_calls = parse_tool_calls_from_content(
                content='<tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tool_call>',
                tools=tools,
                tool_call_parser="llama3",
                generate_tool_call_id=lambda call_info, cnt: f"call_{cnt}",
            )

        self.assertEqual(remaining_text, "")
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "get_weather")
        self.assertEqual(tool_calls[0].arguments, '{"city": "Tokyo"}')
        self.assertEqual(tool_calls[0].call_id, "call_0")

    def test_multiple_tool_calls_parsing(self):
        """Test parsing content with multiple tool calls"""
        tools = [
            Tool(
                type="function",
                function=Function(name="get_weather", description="Get weather"),
            ),
            Tool(
                type="function",
                function=Function(name="get_time", description="Get time"),
            ),
        ]

        # Create mock call infos
        mock_call_info1 = MagicMock()
        mock_call_info1.name = "get_weather"
        mock_call_info1.parameters = '{"city": "Tokyo"}'

        mock_call_info2 = MagicMock()
        mock_call_info2.name = "get_time"
        mock_call_info2.parameters = '{"timezone": "JST"}'

        with patch(
            "sglang.srt.entrypoints.openai.utils.FunctionCallParser"
        ) as MockParser:
            mock_parser_instance = MagicMock()
            mock_parser_instance.has_tool_call.return_value = True
            mock_parser_instance.parse_non_stream.return_value = (
                "Some remaining text",
                [mock_call_info1, mock_call_info2],
            )
            MockParser.return_value = mock_parser_instance

            remaining_text, tool_calls = parse_tool_calls_from_content(
                content="Tool call content here",
                tools=tools,
                tool_call_parser="llama3",
                generate_tool_call_id=lambda call_info, cnt: f"call_{cnt}",
            )

        self.assertEqual(remaining_text, "Some remaining text")
        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(tool_calls[0].name, "get_weather")
        self.assertEqual(tool_calls[0].call_id, "call_0")
        self.assertEqual(tool_calls[1].name, "get_time")
        self.assertEqual(tool_calls[1].call_id, "call_1")

    def test_parsing_error_fallback(self):
        """Test that parsing errors fall back to returning original content"""
        tools = [
            Tool(
                type="function",
                function=Function(name="test_func", description="Test"),
            )
        ]

        with patch(
            "sglang.srt.entrypoints.openai.utils.FunctionCallParser"
        ) as MockParser:
            mock_parser_instance = MagicMock()
            mock_parser_instance.has_tool_call.return_value = True
            mock_parser_instance.parse_non_stream.side_effect = Exception(
                "Parsing error"
            )
            MockParser.return_value = mock_parser_instance

            remaining_text, tool_calls = parse_tool_calls_from_content(
                content="Invalid tool call content",
                tools=tools,
                tool_call_parser="llama3",
                generate_tool_call_id=lambda call_info, cnt: f"call_{cnt}",
            )

        self.assertEqual(remaining_text, "Invalid tool call content")
        self.assertEqual(len(tool_calls), 0)

    def test_tool_call_id_generation(self):
        """Test that tool call IDs are generated correctly using the provided function"""
        tools = [
            Tool(
                type="function",
                function=Function(name="func1", description="Function 1"),
            )
        ]

        mock_call_info = MagicMock()
        mock_call_info.name = "func1"
        mock_call_info.parameters = "{}"

        with patch(
            "sglang.srt.entrypoints.openai.utils.FunctionCallParser"
        ) as MockParser:
            mock_parser_instance = MagicMock()
            mock_parser_instance.has_tool_call.return_value = True
            mock_parser_instance.parse_non_stream.return_value = ("", [mock_call_info])
            MockParser.return_value = mock_parser_instance

            # Custom ID generator
            def custom_id_generator(call_info, cnt):
                return f"custom_{call_info.name}_{cnt}"

            remaining_text, tool_calls = parse_tool_calls_from_content(
                content="Tool call",
                tools=tools,
                tool_call_parser="llama3",
                generate_tool_call_id=custom_id_generator,
            )

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].call_id, "custom_func1_0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
