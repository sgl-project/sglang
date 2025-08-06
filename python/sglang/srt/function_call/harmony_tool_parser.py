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
"""Harmony tool call parser for processing tool calls in harmony models."""

import uuid
from typing import List, Optional, Tuple

from sglang.srt.entrypoints.openai.protocol import (
    ChatMessage,
    FunctionResponse,
    ToolCall,
)


class HarmonyToolCallParser:
    """Parser for extracting tool calls from harmony model outputs."""

    def extract_tool_calls_from_message(self, msg) -> Optional[ToolCall]:
        """
        Extract tool call from a single message if it's a tool call.

        Args:
            msg: The harmony message

        Returns:
            ToolCall if the message is a tool call, None otherwise
        """
        if (
            msg.channel == "commentary"
            and msg.recipient
            and msg.recipient.startswith("functions.")
        ):
            function_name = msg.recipient.split(".")[-1]
            arguments = msg.content[0].text if msg.content else "{}"

            return ToolCall(
                id=f"call_{uuid.uuid4().hex[:24]}",
                function=FunctionResponse(
                    name=function_name,
                    arguments=arguments,
                ),
            )
        return None

    def process_streaming_chunk(
        self,
        harmony_parser,
        index: int,
        tool_call_trackers: dict,
        stream_buffers: dict,
    ) -> Tuple[Optional[dict], bool, Optional[str]]:
        """
        Process a streaming chunk for tool calls.

        Args:
            harmony_parser: The harmony parser instance
            index: The choice index
            tool_call_trackers: Dict tracking tool calls per choice
            stream_buffers: Dict for buffering content

        Returns:
            Tuple of (tool_call_data, is_tool_call, delta)
        """
        # Check if we're in a tool call
        is_tool_call = (
            harmony_parser.current_channel == "commentary"
            and harmony_parser.current_recipient
            and harmony_parser.current_recipient.startswith("functions.")
        )

        delta = harmony_parser.last_content_delta or ""
        tool_call_data = None

        if is_tool_call:
            # Handle tool call streaming
            function_name = harmony_parser.current_recipient.split(".")[-1]

            # Track tool call indices per choice
            if index not in tool_call_trackers:
                tool_call_trackers[index] = {"count": 0, "current_function": None}

            # Check if we just started a new tool call
            tool_call_tracker = tool_call_trackers[index]
            if tool_call_tracker["current_function"] != function_name:
                # New tool call started
                tool_call_tracker["current_function"] = function_name
                tool_call_index = tool_call_tracker["count"]
                tool_call_tracker["count"] += 1

                # Store the tool call index for this function
                tool_call_key = f"{index}_{function_name}"
                stream_buffers[tool_call_key] = {
                    "index": tool_call_index,
                    "content": "",
                }

                tool_call_data = {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "index": tool_call_index,
                    "function_name": function_name,
                    "arguments": delta,
                    "is_first_chunk": True,
                }
            else:
                # Subsequent chunks for the same tool call
                tool_call_key = f"{index}_{function_name}"
                tool_call_index = stream_buffers[tool_call_key]["index"]

                tool_call_data = {
                    "id": None,
                    "index": tool_call_index,
                    "function_name": None,
                    "arguments": delta,
                    "is_first_chunk": False,
                }

            stream_buffers[tool_call_key]["content"] += delta

        return tool_call_data, is_tool_call, delta
