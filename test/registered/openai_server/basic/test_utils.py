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
    Tool,
)
from sglang.srt.entrypoints.openai.utils import (
    parse_tool_calls_from_content,
)


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
                history_tool_calls_cnt=0,
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
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
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
                history_tool_calls_cnt=0,
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
                history_tool_calls_cnt=0,
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
                history_tool_calls_cnt=0,
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
                history_tool_calls_cnt=0,
            )

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].call_id, "custom_func1_0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
