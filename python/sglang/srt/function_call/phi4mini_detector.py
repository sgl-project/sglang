"""
Phi-4 Mini Tool Call Detector

This module implements a tool call parser for Phi-4 Mini models.
Phi-4 Mini uses the `functools[...]` format for function calling.

Format Structure:
```
functools[{"name": "function_name", "arguments": {...}}, {"name": "function_name2", "arguments": {...}}]
```

Reference: vLLM phi4mini_tool_parser.py
"""

import json
import logging
from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)


class Phi4MiniDetector(BaseFormatDetector):
    """
    Detector for Phi-4 Mini models with functools tool call format.

    Format Structure:
    ```
    functools[{"name": "xxx", "arguments": {...}}, ...]
    ```

    The format uses `functools[...]` to wrap an array of tool calls,
    where each tool call is a JSON object with "name" and "arguments"
    (or "parameters") fields.
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "functools["
        self.eot_token = "]"
        self.tool_call_separator = ", "

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a Phi-4 Mini format tool call."""
        return "functools[" in text

    def _find_functools_end(self, text: str, start: int) -> int:
        """
        Find the matching closing bracket for functools[...], handling nested brackets
        and JSON strings properly.

        Args:
            text: The text to search in
            start: Position right after 'functools['

        Returns:
            Position of the matching ']', or -1 if not found
        """
        bracket_count = 1
        in_string = False
        escape_next = False

        for i in range(start, len(text)):
            char = text[i]

            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char == '"':
                in_string = not in_string
                continue

            # Only count brackets outside of strings
            if not in_string:
                if char == "[":
                    bracket_count += 1
                elif char == "]":
                    bracket_count -= 1
                    if bracket_count == 0:
                        return i

        return -1

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        Parse function calls from text in Phi-4 Mini functools format.

        Args:
            text: The complete text to parse
            tools: List of available tools

        Returns:
            StreamingParseResult with parsed tool calls and remaining text
        """
        if not self.has_tool_call(text):
            return StreamingParseResult(normal_text=text, calls=[])

        functools_start = text.find("functools[")
        if functools_start == -1:
            return StreamingParseResult(normal_text=text, calls=[])

        try:
            # Extract the content before the functools block
            normal_text_before = text[:functools_start] if functools_start > 0 else ""

            # Find the matching closing bracket using proper bracket counting
            content_start = functools_start + len("functools[")
            functools_end = self._find_functools_end(text, content_start)

            if functools_end == -1:
                # Incomplete functools block
                return StreamingParseResult(normal_text=text, calls=[])

            # Extract the JSON array content inside functools[...]
            json_content = "[" + text[content_start:functools_end] + "]"

            # Parse the JSON array
            function_call_arr = json.loads(json_content)

            if not isinstance(function_call_arr, list):
                function_call_arr = [function_call_arr]

            # Parse tool calls using base class method
            calls = self.parse_base_json(function_call_arr, tools)

            # Get any text after the functools block
            normal_text_after = (
                text[functools_end + 1 :] if functools_end + 1 < len(text) else ""
            )

            return StreamingParseResult(
                normal_text=normal_text_before + normal_text_after,
                calls=calls,
            )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse functools JSON: {e}")
            return StreamingParseResult(normal_text=text, calls=[])
        except Exception as e:
            logger.error(f"Error parsing Phi-4 Mini tool call: {e}")
            return StreamingParseResult(normal_text=text, calls=[])

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing for Phi-4 Mini functools format.

        Buffers input until a complete functools[...] block is found,
        then parses and emits any detected calls.

        Args:
            new_text: New chunk of text to parse
            tools: List of available tools

        Returns:
            StreamingParseResult with parsed tool calls and normal text
        """
        self._buffer += new_text
        current_text = self._buffer

        # Check if we have the start of a functools block
        functools_start = current_text.find("functools[")

        if functools_start == -1:
            # No functools block started yet
            # Check for partial "functools[" at the end
            partial_len = self._ends_with_partial_token(current_text, self.bot_token)
            if partial_len > 0:
                # Hold back the partial token
                safe_text = current_text[:-partial_len]
                self._buffer = current_text[-partial_len:]
                if safe_text:
                    return StreamingParseResult(normal_text=safe_text)
                return StreamingParseResult()
            else:
                # No partial token, output all as normal text
                self._buffer = ""
                return StreamingParseResult(normal_text=current_text)

        # We have the start of a functools block
        # Output any text before it as normal text
        normal_text_before = (
            current_text[:functools_start] if functools_start > 0 else ""
        )

        # Find the matching closing bracket
        # Need to handle JSON properly - ignore brackets inside strings
        bracket_start = functools_start + len("functools[")
        bracket_count = 1
        bracket_end = -1
        in_string = False
        escape_next = False

        for i in range(bracket_start, len(current_text)):
            char = current_text[i]

            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char == '"':
                in_string = not in_string
                continue

            # Only count brackets outside of strings
            if not in_string:
                if char == "[":
                    bracket_count += 1
                elif char == "]":
                    bracket_count -= 1
                    if bracket_count == 0:
                        bracket_end = i
                        break

        if bracket_end == -1:
            # Incomplete functools block, keep buffering
            if normal_text_before:
                self._buffer = current_text[functools_start:]
                return StreamingParseResult(normal_text=normal_text_before)
            return StreamingParseResult()

        # We have a complete functools block
        functools_content = current_text[functools_start : bracket_end + 1]

        # Parse the complete block
        result = self.detect_and_parse(functools_content, tools)

        # Update buffer with remaining text after the functools block
        remaining_text = current_text[bracket_end + 1 :]
        self._buffer = remaining_text

        # Add any text before the functools block to the result
        if normal_text_before:
            result.normal_text = normal_text_before + (result.normal_text or "")

        return result

    def structure_info(self) -> _GetInfoFunc:
        """
        Return a function that creates StructureInfo for constrained generation.

        Returns:
            A function that takes a tool name and returns StructureInfo
        """
        return lambda name: StructureInfo(
            begin='functools[{"name": "' + name + '", "arguments": ',
            end="}]",
            trigger="functools[",
        )
