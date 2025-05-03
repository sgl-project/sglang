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

import logging
from typing import Any, List, Optional, Tuple, Union

from sglang.srt.openai_api.protocol import FunctionResponse, ToolCall
from sglang.srt.parser.code_completion_parser import (
    generate_completion_prompt,
    is_completion_template_defined,
)
from sglang.srt.parser.function_call_parser import FunctionCallParser, ToolCallItem
from sglang.srt.parser.reasoning_parser import ReasoningParser

logger = logging.getLogger(__name__)


class ParserManager:
    """Parser manager for managing different parsers and parsing operations"""

    def __init__(self, support_stream: bool = False):
        self.support_stream = support_stream
        if support_stream:
            self.function_call_parser_dict = {}
            self.reasoning_parser_dict = {}

    def handle_completion_request(self, prompt: str, suffix: str) -> str:
        """Handle completion request by applying the appropriate template"""
        if is_completion_template_defined():
            if suffix == "":
                return prompt
            prompt = generate_completion_prompt(prompt, suffix)
        return prompt

    def handle_tool_calls(
        self,
        text: str,
        tools: List[Any],
        tool_call_parser: Optional[str] = None,
        use_tool_call: bool = False,
        stream: bool = False,
        index: Optional[int] = None,
        finish_reason_type: Optional[str] = None,
    ) -> Tuple[
        Optional[str],
        Optional[Union[List[ToolCall], List[ToolCallItem]]],
        Optional[str],
    ]:
        """Handle tool calls in the response"""
        if not use_tool_call or not tools:
            return text, None, finish_reason_type

        if stream:
            assert self.support_stream, "Stream is not supported"
            if index not in self.function_call_parser_dict:
                self.function_call_parser_dict[index] = FunctionCallParser(
                    tools=tools,
                    tool_call_parser=tool_call_parser,
                )
            parser = self.function_call_parser_dict[index]
            # parse_increment => returns (normal_text, calls)
            normal_text, calls = parser.parse_stream_chunk(text)
            return normal_text, calls, finish_reason_type

        else:
            parser = FunctionCallParser(tools, tool_call_parser)
            if parser.has_tool_call(text):
                if finish_reason_type == "stop":
                    finish_reason_type = "tool_calls"
                text, call_info_list = parser.parse_non_stream(text)
                tool_calls = [
                    ToolCall(
                        id=str(call_info.tool_index),
                        function=FunctionResponse(
                            name=call_info.name, arguments=call_info.parameters
                        ),
                    )
                    for call_info in call_info_list
                ]

                return text, tool_calls, finish_reason_type
            return text, None, finish_reason_type

    def has_tool_call(self, text: str, tools: List[Any], tool_call_parser: str) -> bool:
        """Check if the text contains tool calls"""
        parser = FunctionCallParser(
            tools=tools,
            tool_call_parser=tool_call_parser,
        )
        return parser.has_tool_call(text)

    def transform_call_item(
        self,
        call_item: ToolCallItem,
        index: int,
        finish_reason_type: Optional[str] = None,
    ) -> Tuple[ToolCall, str]:
        """Transform call item -> FunctionResponse + ToolCall"""
        if finish_reason_type == "stop":
            parser = self.function_call_parser_dict[index]
            latest_delta_len = 0
            if isinstance(call_item.parameters, str):
                latest_delta_len = len(call_item.parameters)

            expected_call = (
                parser.multi_format_parser.detectors[0]
                .prev_tool_call_arr[index]
                .get("arguments", {})
            )
            actual_call = parser.multi_format_parser.detectors[
                0
            ].streamed_args_for_tool[index]
            if latest_delta_len > 0:
                actual_call = actual_call[:-latest_delta_len]
            remaining_call = expected_call.replace(actual_call, "", 1)
            call_item.parameters = remaining_call

            finish_reason_type = "tool_calls"

        tool_call = ToolCall(
            id=str(call_item.tool_index),
            index=call_item.tool_index,
            function=FunctionResponse(
                name=call_item.name,
                arguments=call_item.parameters,
            ),
        )
        return tool_call, finish_reason_type

    def handle_reasoning(
        self,
        text: str,
        reasoning_parser: Optional[str] = None,
        separate_reasoning: bool = False,
        stream: bool = False,
        index: Optional[int] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Handle reasoning in the response"""
        if not separate_reasoning or not reasoning_parser:
            return None, text

        if stream:
            assert self.support_stream, "Stream is not supported"
            if index not in self.reasoning_parser_dict:
                self.reasoning_parser_dict[index] = ReasoningParser(
                    reasoning_parser,
                    stream_reasoning=True,
                )
            parser = self.reasoning_parser_dict[index]
            reasoning_text, text = parser.parse_stream_chunk(text)
            return reasoning_text, text
        else:
            parser = ReasoningParser(
                model_type=reasoning_parser, stream_reasoning=False
            )
            reasoning_text, text = parser.parse_non_stream(text)
            return reasoning_text, text
