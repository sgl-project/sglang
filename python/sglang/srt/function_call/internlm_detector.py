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
                        f"[InternLM Tool Call] Parsed tool call #{idx + 1}: name={name}, "
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
                        f"[InternLM Tool Call] Failed to parse JSON for tool call #{idx + 1}: {e}"
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

        Every complete call in the buffer is emitted, so a delta carrying two
        calls yields both of them instead of only the first one. Text outside
        tool-call blocks is streamed as normal text, and a trailing partial
        start marker is held back for the next delta.
        """
        self._buffer += new_text
        current_text = self._buffer

        normal_text = ""
        calls: list[ToolCallItem] = []

        while True:
            start = current_text.find(self.bot_token)
            if start == -1:
                # No start marker: emit the text, holding back a possible
                # partial marker that a later delta could complete.
                held_back = self._pending_start_length(current_text)
                if held_back:
                    normal_text += current_text[:-held_back]
                    current_text = current_text[-held_back:]
                else:
                    normal_text += current_text
                    current_text = ""
                break

            normal_text += current_text[:start]

            # Check if we have a complete tool call (with end marker)
            end = current_text.find(self.eot_token, start + len(self.bot_token))
            if end == -1:
                # We have bot_token but no eot_token yet - keep the tool call
                # part in the buffer, the text before it has been emitted.
                current_text = current_text[start:]
                break

            # Use detect_and_parse on the complete tool call
            section_end = end + len(self.eot_token)
            result = self.detect_and_parse(current_text[start:section_end], tools=tools)
            calls.extend(self._commit_streamed_calls(result.calls))

            # Remove the completed tool call from buffer and keep scanning so
            # a second call in the same delta is not delayed by one round-trip.
            current_text = current_text[section_end:]

        self._buffer = current_text
        # Clean up any stray end tokens
        if self.eot_token in normal_text:
            normal_text = normal_text.replace(self.eot_token, "")
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def _commit_streamed_calls(
        self, parsed_calls: list[ToolCallItem]
    ) -> list[ToolCallItem]:
        """Assign sequential indices and record per-call streaming state."""
        committed = []
        for call in parsed_calls:
            if self.current_tool_id == -1:
                self.current_tool_id = 0
                self.prev_tool_call_arr = []
                self.streamed_args_for_tool = []

            # Ensure we have enough entries in our tracking arrays
            while len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})
            while len(self.streamed_args_for_tool) <= self.current_tool_id:
                self.streamed_args_for_tool.append("")

            call.tool_index = self.current_tool_id
            self.prev_tool_call_arr[self.current_tool_id] = {
                "name": call.name,
                "arguments": json.loads(call.parameters),
            }
            self.streamed_args_for_tool[self.current_tool_id] = call.parameters
            committed.append(call)
            # Increment tool ID for next tool call
            self.current_tool_id += 1
        return committed

    def _pending_start_length(self, text: str) -> int:
        """Length of the longest buffer suffix that is a partial start marker.

        ``_ends_with_partial_token`` returns the shortest matching suffix, which
        would leak the earlier characters of a start marker that is still
        arriving one token at a time.
        """
        max_length = min(len(text), len(self.bot_token) - 1)
        for length in range(max_length, 0, -1):
            if self.bot_token.startswith(text[-length:]):
                return length
        return 0

    def finish(self, tools: list[Tool]) -> StreamingParseResult:
        """Release buffered prose; an incomplete call is dropped at stream end."""
        buffered, self._buffer = self._buffer, ""
        start = buffered.find(self.bot_token)
        if start != -1:
            buffered = buffered[:start]
        else:
            pending = self._pending_start_length(buffered)
            if pending:
                buffered = buffered[:-pending]
        return StreamingParseResult(normal_text=buffered)

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
