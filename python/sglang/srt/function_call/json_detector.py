import json
import logging
from typing import Any, Dict, List

from partial_json_parser.core.exceptions import MalformedJSON
from partial_json_parser.core.options import Allow

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    StructureInfo,
    _GetInfoFunc,
)
from sglang.srt.function_call.json_schema_composer import JSONSchemaComposer
from sglang.srt.function_call.utils import (
    _find_common_prefix,
    _is_complete_json,
    _partial_json_loads,
)

logger = logging.getLogger(__name__)


class JSONDetector:
    """
    Universal detector for JSON Schema constrained tool calls.
    
    This detector handles the standardized JSON format that all models output
    when using JSON Schema constraints, eliminating the need for model-specific
    JSON parsing logic.
    
    Format Structure:
    ```
    [
      {
        "name": "tool_name",
        "parameters": {...}
      }
    ]
    ```
    """

    def __init__(self):
        # Streaming state management
        # Buffer for accumulating incomplete patterns that arrive across multiple streaming chunks
        self._buffer = ""
        # Stores complete tool call info (name and arguments) for each tool being parsed.
        # Used by serving layer for completion handling when streaming ends.
        # Format: [{"name": str, "arguments": dict}, ...]
        self.prev_tool_call_arr: List[Dict] = []
        # Index of currently streaming tool call. Starts at -1 (no active tool),
        # increments as each tool completes. Tracks which tool's arguments are streaming.
        self.current_tool_id: int = -1
        # Flag for whether current tool's name has been sent to client.
        # Tool names sent first with empty parameters, then arguments stream incrementally.
        self.current_tool_name_sent: bool = False
        # Tracks raw JSON string content streamed to client for each tool's arguments.
        # Critical for serving layer to calculate remaining content when streaming ends.
        # Each index corresponds to a tool_id. Example: ['{"location": "San Francisco"', '{"temp": 72']
        self.streamed_args_for_tool: List[str] = []

        # Token configuration (not needed for JSON Schema, but kept for compatibility)
        self.bot_token = ""
        self.eot_token = ""
        # Multiple tool calls are separated by array elements
        self.tool_call_separator = ", "

    def _get_tool_indices(self, tools: List[Tool]) -> Dict[str, int]:
        """
        Get a mapping of tool names to their indices in the tools list.

        This utility method creates a dictionary mapping function names to their
        indices in the tools list, which is commonly needed for tool validation
        and ToolCallItem creation.

        Args:
            tools: List of available tools

        Returns:
            Dictionary mapping tool names to their indices
        """
        return {
            tool.function.name: i for i, tool in enumerate(tools) if tool.function.name
        }

    def parse_base_json(self, action: Any, tools: List[Tool]) -> List[ToolCallItem]:
        tool_indices = self._get_tool_indices(tools)
        if not isinstance(action, list):
            action = [action]

        results = []
        for act in action:
            name = act.get("name")
            if name and name in tool_indices:
                results.append(
                    ToolCallItem(
                        tool_index=-1,  # Caller should update this based on the actual tools array called
                        name=name,
                        parameters=json.dumps(
                            act.get("parameters") or act.get("arguments", {}),
                            ensure_ascii=False,
                        ),
                    )
                )
            else:
                logger.warning(f"Model attempted to call undefined function: {name}")

        return results

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains JSON format tool calls."""
        stripped_text = text.strip()
        return stripped_text.startswith("[") or stripped_text.startswith("{")

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """Parse function calls from text, handling JSON arrays and objects."""
        stripped_text = text.strip()
        
        # Handle both array and single object formats
        if not (stripped_text.startswith("[") or stripped_text.startswith("{")):
            return StreamingParseResult(normal_text=text, calls=[])

        try:
            # Try to parse as complete JSON first
            parsed_content = json.loads(stripped_text)
            
            if isinstance(parsed_content, list):
                # Array format: [{"name": "tool", "parameters": {...}}]
                all_actions = parsed_content
            else:
                # Single object format: {"name": "tool", "parameters": {...}}
                all_actions = [parsed_content]
            
            # Parse the actions into tool calls
            calls = self.parse_base_json(all_actions, tools) if all_actions else []
            
            return StreamingParseResult(normal_text="", calls=calls)
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {stripped_text}, error: {str(e)}")
            # If JSON parsing fails, return the text as normal text
            return StreamingParseResult(normal_text=text, calls=[])

    def _ends_with_partial_token(self, buffer: str, bot_token: str) -> int:
        """
        Check if buffer ends with a partial bot_token.
        Return the length of the partial bot_token.
        """
        for i in range(1, min(len(buffer) + 1, len(bot_token))):
            if bot_token.startswith(buffer[-i:]):
                return i
        return 0

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing with tool validation for JSON format.
        
        This implementation handles JSON Schema constrained tool calls where:
        1. Tool calls are in JSON array format
        2. JSON can be parsed incrementally using partial_json_loads
        3. Multiple tool calls are separated by array elements
        """
        # Append new text to buffer
        self._buffer += new_text
        current_text = self._buffer

        # Check if we have a tool call (JSON array or object)
        if not self.has_tool_call(current_text):
            # Only clear buffer if we're sure no tool call is starting
            if not self._ends_with_partial_token(self._buffer, "["):
                normal_text = self._buffer
                self._buffer = ""
                return StreamingParseResult(normal_text=normal_text)
            else:
                # Might be partial JSON start, keep buffering
                return StreamingParseResult()

        # Build tool indices if not already built
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR

        try:
            # For JSON Schema format, we expect the text to start with [ or {
            if current_text.startswith("["):
                start_idx = 1  # Skip the opening bracket
            elif current_text.startswith("{"):
                start_idx = 0  # Single object
            else:
                start_idx = 0

            if start_idx >= len(current_text):
                return StreamingParseResult()

            # Try to parse JSON incrementally
            try:
                (obj, end_idx) = _partial_json_loads(current_text[start_idx:], flags)
            except MalformedJSON:
                # Incomplete JSON, keep buffering
                return StreamingParseResult()

            is_current_complete = _is_complete_json(
                current_text[start_idx : start_idx + end_idx]
            )

            # Validate tool name if present
            if "name" in obj and obj["name"] not in self._tool_indices:
                # Invalid tool name - reset state
                self._buffer = ""
                self.current_tool_id = -1
                self.current_tool_name_sent = False
                if self.streamed_args_for_tool:
                    self.streamed_args_for_tool.pop()
                return StreamingParseResult()

            # Handle parameters/arguments consistency
            if "parameters" in obj:
                assert (
                    "arguments" not in obj
                ), "model generated both parameters and arguments"
                obj["arguments"] = obj["parameters"]

            current_tool_call = obj

            if not current_tool_call:
                return StreamingParseResult()

            # Case 1: Handle tool name streaming
            if not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")

                if function_name and function_name in self._tool_indices:
                    # Initialize tool call tracking
                    if self.current_tool_id == -1:
                        self.current_tool_id = 0
                        self.streamed_args_for_tool.append("")
                    elif self.current_tool_id >= len(self.streamed_args_for_tool):
                        while len(self.streamed_args_for_tool) <= self.current_tool_id:
                            self.streamed_args_for_tool.append("")

                    # Send the tool name with empty parameters
                    res = StreamingParseResult(
                        calls=[
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                name=function_name,
                                parameters="",
                            )
                        ],
                    )
                    self.current_tool_name_sent = True
                else:
                    res = StreamingParseResult()

            # Case 2: Handle streaming arguments
            else:
                cur_arguments = current_tool_call.get("arguments")
                res = StreamingParseResult()

                if cur_arguments:
                    # Calculate how much of the arguments we've already streamed
                    sent = len(self.streamed_args_for_tool[self.current_tool_id])
                    cur_args_json = json.dumps(cur_arguments)
                    prev_arguments = None
                    if self.current_tool_id < len(self.prev_tool_call_arr):
                        prev_arguments = self.prev_tool_call_arr[
                            self.current_tool_id
                        ].get("arguments")

                    argument_diff = None

                    # If the current tool's JSON is complete, send all remaining arguments
                    if is_current_complete:
                        argument_diff = cur_args_json[sent:]
                        completing_tool_id = self.current_tool_id

                        # Only remove the processed portion, keep unprocessed content
                        self._buffer = current_text[start_idx + end_idx :]

                        if self.current_tool_id < len(self.prev_tool_call_arr):
                            self.prev_tool_call_arr[self.current_tool_id].clear()
                        self.current_tool_name_sent = False
                        self.streamed_args_for_tool[self.current_tool_id] = ""
                        self.current_tool_id += 1

                    # If the tool is still being parsed, send incremental changes
                    elif prev_arguments:
                        prev_args_json = json.dumps(prev_arguments)
                        if cur_args_json != prev_args_json:
                            prefix = _find_common_prefix(prev_args_json, cur_args_json)
                            argument_diff = prefix[sent:]

                    # Send the argument diff if there's something new
                    if argument_diff is not None:
                        tool_index_to_use = (
                            completing_tool_id
                            if is_current_complete
                            else self.current_tool_id
                        )
                        res = StreamingParseResult(
                            calls=[
                                ToolCallItem(
                                    tool_index=tool_index_to_use,
                                    parameters=argument_diff,
                                )
                            ],
                        )
                        if not is_current_complete:
                            self.streamed_args_for_tool[
                                self.current_tool_id
                            ] += argument_diff

            # Update prev_tool_call_arr with current state
            if self.current_tool_id >= 0:
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                self.prev_tool_call_arr[self.current_tool_id] = current_tool_call

            return res

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult()

    def supports_structural_tag(self) -> bool:
        """Return True if this detector supports structural tag format."""
        return True

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin='{"name":"' + name + '", "parameters":',
            end="}",
            trigger="",
        )

    def build_json_schema(self, tools: List[Tool], tool_choice: str = "auto"):
        return JSONSchemaComposer.build_json_schema(
            tools,
            tool_choice=tool_choice,
        )
    
    def build_ebnf(self, tools: List[Tool]):
        # Keep for backward compatibility, but JSON Schema is preferred
        from sglang.srt.function_call.ebnf_composer import EBNFComposer
        return EBNFComposer.build_ebnf(
            tools,
            function_format="json",
            tool_call_separator=self.tool_call_separator,
        )
