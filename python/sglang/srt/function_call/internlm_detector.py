# modified from https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/serve/openai/tool_parser/internlm2_parser.py

import json
import logging
import re
from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)


class InternlmDetector(BaseFormatDetector):
    """
    Detector for InternLM2/Intern-S1 model function call format.

    The InternLM format uses special tokens to delimit function calls
    with JSON for arguments.

    Format Structure:
    ```
    text<|action_start|> <|plugin|>
    {json}<|action_end|>
    ```

    Examples:
    ```
    What's the weather like?<|action_start|> <|plugin|>
    {"name": "get_weather", "parameters": {"location": "Tokyo"}}<|action_end|>
    ```

    Key Components:
    - Tool Call Start: `<|action_start|> <|plugin|>`
    - Tool Call End: `<|action_end|>`
    - Arguments: JSON object with `name` and `parameters`/`arguments`
    - Supports multiple sequential tool calls in both streaming and non-streaming modes

    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<|action_start|> <|plugin|>"
        self.eot_token = "<|action_end|>"
        self.position = 0

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains an InternLM format tool call."""
        has_call = self.bot_token in text
        return has_call

    def get_arguments(self, obj):
        """Extract arguments from object, supporting both 'parameters' and 'arguments' keys."""
        if "parameters" in obj:
            return obj.get("parameters")
        elif "arguments" in obj:
            return obj.get("arguments")
        return None

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.
        Supports multiple tool calls in the format:
        <|action_start|> <|plugin|>\n{JSON}<|action_end|>

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: StreamingParseResult with normal text and parsed tool calls.
        """

        # Find the first occurrence of tool call marker to extract normal text
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text

        if self.bot_token not in text:
            logger.warning("[InternLM Tool Call] No tool call markers found in text")
            return StreamingParseResult(normal_text=normal_text, calls=[])

        # Use regex to find all tool call blocks
        # Pattern matches: {self.bot_token}{...}{self.eot_token}
        tool_call_pattern = (
            rf"{re.escape(self.bot_token)}\s*(.*?){re.escape(self.eot_token)}"
        )
        matches = re.findall(tool_call_pattern, text, re.DOTALL)

        if not matches:
            logger.warning("[InternLM Tool Call] No complete tool call blocks found")
            return StreamingParseResult(normal_text=text, calls=[])

        logger.info(f"[InternLM Tool Call] Found {len(matches)} tool call(s)")

        calls = []
        tool_indices = self._get_tool_indices(tools)

        try:
            for idx, action_json in enumerate(matches):
                action_json = action_json.strip()

                try:
                    # Parse the JSON
                    action_dict = json.loads(action_json)
                    name = action_dict.get("name")
                    parameters = self.get_arguments(action_dict)

                    if not parameters:
                        parameters = {}

                    logger.info(
                        f"[InternLM Tool Call] Parsed tool call #{idx+1}: name={name}, "
                        f"parameters={json.dumps(parameters, ensure_ascii=False)}"
                    )

                    # Validate tool name
                    if not (name and name in tool_indices):
                        logger.warning(
                            f"[InternLM Tool Call] Model attempted to call undefined function: {name}, "
                            f"available_tools={list(tool_indices.keys())}"
                        )
                        if not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get():
                            continue  # Skip this tool call

                    # Create tool call item and add to list
                    tool_call = ToolCallItem(
                        tool_index=tool_indices[name],
                        name=name,
                        parameters=json.dumps(parameters, ensure_ascii=False),
                    )
                    calls.append(tool_call)

                except json.JSONDecodeError as e:
                    logger.error(
                        f"[InternLM Tool Call] Failed to parse JSON for tool call #{idx+1}: {e}"
                    )
                    continue

            logger.info(
                f"[InternLM Tool Call] Successfully parsed {len(calls)} tool call(s), "
                f"normal_text_length={len(normal_text)}"
            )
            return StreamingParseResult(normal_text=normal_text, calls=calls)

        except Exception as e:
            logger.error(
                f"[InternLM Tool Call] Error in detect_and_parse: {e}", exc_info=True
            )
            return StreamingParseResult(normal_text=text, calls=[])

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing for InternLM format.

        Supports a single tool call in streaming mode.
        """
        self._buffer += new_text
        current_text = self._buffer

        # Check if we don't have a tool call start marker
        start = current_text.find(self.bot_token)
        if start == -1:
            # No tool call marker found
            # If we've already processed tool calls, don't return text again
            if self.current_tool_id > 0:
                self._buffer = ""
                return StreamingParseResult(normal_text="")

            # Check if buffer could be partial start of bot_token
            if not self._ends_with_partial_token(current_text, self.bot_token):
                # Not a partial match, return as normal text
                normal_text = current_text
                self._buffer = ""
                # Clean up any stray end tokens
                if self.eot_token in normal_text:
                    normal_text = normal_text.replace(self.eot_token, "")
                return StreamingParseResult(normal_text=normal_text)
            else:
                # Might be partial start token, keep buffering
                return StreamingParseResult()

        # Check if we have a complete tool call (with end marker)
        end = current_text.find(self.eot_token)
        if end != -1:
            # We have a complete tool call
            # Initialize state if this is the first tool call
            if self.current_tool_id == -1:
                self.current_tool_id = 0
                self.prev_tool_call_arr = []
                self.streamed_args_for_tool = [""]

            # Ensure we have enough entries in our tracking arrays
            while len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})
            while len(self.streamed_args_for_tool) <= self.current_tool_id:
                self.streamed_args_for_tool.append("")

            # Use detect_and_parse on the complete tool call
            complete_section = current_text[: end + len(self.eot_token)]
            result = self.detect_and_parse(complete_section, tools=tools)

            if result.calls:
                # Update the tool call index
                result.calls[0].tool_index = self.current_tool_id
                # Store the parsed tool call for reference
                self.prev_tool_call_arr[self.current_tool_id] = {
                    "name": result.calls[0].name,
                    "arguments": json.loads(result.calls[0].parameters),
                }
                self.streamed_args_for_tool[self.current_tool_id] = result.calls[
                    0
                ].parameters
                # Increment tool ID for next tool call
                self.current_tool_id += 1

            # Remove the completed tool call from buffer
            self._buffer = current_text[end + len(self.eot_token) :]
            return result

        # We have bot_token but no eot_token yet - handle partial tool call streaming
        # Extract normal text before the tool call
        normal_text = current_text[:start]
        # Keep the tool call part in buffer
        self._buffer = current_text[start:]
        return StreamingParseResult(normal_text=normal_text)

    def structure_info(self) -> _GetInfoFunc:
        """
        Return structure information for constrained generation.

        For InternLM format, the structure is:
        - begin: <|action_start|> <|plugin|>\n
        - end: <|action_end|>
        - trigger: the begin token
        """
        return lambda name: StructureInfo(
            begin='<|action_start|> <|plugin|>\n{"name": "'
            + name
            + '", "parameters": ',
            end="}<|action_end|>",
            trigger="<|action_start|> <|plugin|>",
        )
