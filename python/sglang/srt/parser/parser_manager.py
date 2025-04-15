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
"""Parser manager for managing different parsers and parsing operations"""

import logging
from typing import Any, List, Optional, Tuple

from sglang.srt.openai_api.protocol import FunctionResponse, ToolCall
from sglang.srt.parser.code_completion_parser import (
    generate_completion_prompt,
    is_completion_template_defined,
)
from sglang.srt.parser.function_call_parser import FunctionCallParser, ToolCallItem
from sglang.srt.parser.reasoning_parser import ReasoningParser

logger = logging.getLogger(__name__)


class ParserManager:

    def __init__(self, support_stream: bool = False):
        self.support_stream = support_stream
        if support_stream:
            self.parser_dict = {}
            self.reasoning_parser_dict = {}

    def handle_completion_request(
        self, prompt: str, suffix: str, template_name: str
    ) -> str:
        """Handle completion request by applying the appropriate template"""
        if is_completion_template_defined():
            prompt = generate_completion_prompt(prompt, suffix, template_name)
        return prompt

    def handle_tool_calls(
        self,
        text: str,
        tools: List[Any],
        tool_call_parser: Optional[str] = None,
        use_tool_call: bool = True,
        stream: bool = False,
        delta: Optional[str] = None,
        index: Optional[int] = None,
        finish_reason_type: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[List[ToolCall]], Optional[str]]:
        """Handle tool calls in the response"""
        if not use_tool_call or not tools:
            return text, None, finish_reason_type

        if stream:
            assert self.support_stream, "Stream is not supported"
            if index not in self.parser_dict:
                self.parser_dict[index] = FunctionCallParser(
                    tools=tools,
                    tool_call_parser=tool_call_parser,
                )
            parser = self.parser_dict[index]

            # parse_increment => returns (normal_text, calls)
            normal_text, calls = parser.parse_stream_chunk(delta)

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

    def transform_call_item(
        self,
        call_item: ToolCallItem,
        index: int,
        finish_reason_type: Optional[str] = None,
    ) -> Tuple[ToolCall, str]:
        """Transform call item -> FunctionResponse + ToolCall"""
        parser = self.parser_dict[index]
        if finish_reason_type == "stop":
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
        delta: Optional[str] = None,
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
            reasoning_text, delta = parser.parse_stream_chunk(delta)
            return reasoning_text, delta
        else:
            parser = ReasoningParser(
                model_type=reasoning_parser, stream_reasoning=False
            )
            reasoning_text, text = parser.parse_non_stream(text)
            return reasoning_text, text
