import json
import logging
from typing import Any, List, Optional, Tuple

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import _is_complete_json

logger = logging.getLogger(__name__)


class MistralDetector(BaseFormatDetector):
    """
    Detector for Mistral tool/function call formats.

    Supported formats:

    1) JSON-array format:
       `[TOOL_CALLS] [{"name": "...", "arguments": {...}}, ...]`

    2) Compact format (common in newer templates/models, especially in streaming):
       `[TOOL_CALLS]tool_name[ARGS]{...}`
       (also tolerates missing delimiters like `]` after `[TOOL_CALLS` and/or `[ARGS]` while streaming)

    Reference: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3?chat_template=default
    """

    def __init__(self):
        """Initialize tokens and streaming state."""
        super().__init__()
        # Canonical Mistral prefix for JSON-array tool calls.
        self.bot_token = "[TOOL_CALLS] ["
        # Common marker shared by both JSON-array and compact formats.
        self._tool_calls_marker = "[TOOL_CALLS"
        self.eot_token = "]"
        self.tool_call_separator = ", "

    def has_tool_call(self, text: str) -> bool:
        """Return True if the text contains either supported tool-call marker."""
        return self._tool_calls_marker in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        marker_idx = text.find(self._tool_calls_marker)
        if marker_idx == -1:
            return StreamingParseResult(normal_text=text, calls=[])

        normal_text = text[:marker_idx].strip()
        tool_part = text[marker_idx:]

        # Canonical: `[TOOL_CALLS] [{...}, ...]`
        if self.bot_token in tool_part:
            json_array_str = self._extract_json_array(tool_part)
            if not json_array_str:
                return StreamingParseResult(normal_text=normal_text, calls=[])

            calls: list = []
            try:
                function_call_arr = json.loads(json_array_str)
                if not isinstance(function_call_arr, list):
                    function_call_arr = [function_call_arr]
                calls = self.parse_base_json(function_call_arr, tools)
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Failed to parse JSON part: {json_array_str}, JSON parse error: {str(e)}"
                )
            json_pos = tool_part.find(json_array_str) if json_array_str else -1
            trailing_text = (
                tool_part[json_pos + len(json_array_str) :].strip()
                if json_pos != -1
                else ""
            )
            combined_normal = (
                (normal_text + " " + trailing_text).strip()
                if trailing_text
                else normal_text
            )
            return StreamingParseResult(normal_text=combined_normal, calls=calls)

        # Compact: `[TOOL_CALLS]tool_name[ARGS]{...}`
        parsed = self._try_parse_compact_args_format(tool_part)
        if not parsed:
            return StreamingParseResult(normal_text=normal_text, calls=[])
        func_name, args_obj, consumed = parsed

        calls = self.parse_base_json({"name": func_name, "arguments": args_obj}, tools)
        trailing_text = tool_part[consumed:].strip()
        combined_normal = (
            (normal_text + " " + trailing_text).strip()
            if trailing_text
            else normal_text
        )
        return StreamingParseResult(normal_text=combined_normal, calls=calls)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming parsing for both JSON-array and compact formats.

        For the compact format, this buffers until the JSON arguments payload is complete,
        then emits two items: tool name (with empty parameters) and a full arguments JSON
        chunk (OpenAI streaming semantics).
        """
        self._buffer += new_text
        current_text = self._buffer

        # No marker: either flush as normal text or keep buffering a partial marker.
        if self._tool_calls_marker not in current_text:
            if not self._ends_with_partial_token(self._buffer, self._tool_calls_marker):
                normal_text = self._buffer
                self._buffer = ""
                if self.eot_token in normal_text:
                    normal_text = normal_text.replace(self.eot_token, "")
                return StreamingParseResult(normal_text=normal_text)
            return StreamingParseResult()

        # If there's leading normal text before the marker, stream it out first.
        marker_pos = current_text.find(self._tool_calls_marker)
        if marker_pos > 0:
            normal_text = current_text[:marker_pos]
            self._buffer = current_text[marker_pos:]
            return StreamingParseResult(normal_text=normal_text)

        # Build tool indices if not already built.
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        # Try compact first; JSON-array requires `] [` and often arrives later in streaming.
        compact = self._try_parse_compact_args_format(current_text)
        if compact:
            func_name, args_obj, consumed = compact
            if func_name not in self._tool_indices:
                # Unknown tool: treat as normal text and reset state.
                normal_text = self._buffer
                self._buffer = ""
                return StreamingParseResult(normal_text=normal_text)

            # Initialize state if this is the first tool call.
            if self.current_tool_id == -1:
                self.current_tool_id = 0
                self.prev_tool_call_arr = []
                self.streamed_args_for_tool = []

            args_json = json.dumps(args_obj, ensure_ascii=False)
            tool_id = self.current_tool_id

            # Ensure arrays are large enough.
            while len(self.prev_tool_call_arr) <= tool_id:
                self.prev_tool_call_arr.append({})
            while len(self.streamed_args_for_tool) <= tool_id:
                self.streamed_args_for_tool.append("")

            self.prev_tool_call_arr[tool_id] = {
                "name": func_name,
                "arguments": args_obj,
            }
            self.streamed_args_for_tool[tool_id] = args_json

            calls: List[ToolCallItem] = [
                ToolCallItem(tool_index=tool_id, name=func_name, parameters=""),
                ToolCallItem(tool_index=tool_id, name=None, parameters=args_json),
            ]

            # Consume parsed content from buffer.
            self._buffer = current_text[consumed:]
            self.current_tool_id += 1
            self.current_tool_name_sent = False
            return StreamingParseResult(normal_text="", calls=calls)

        # Canonical format delegates to the BaseFormatDetector JSON streaming logic.
        if self.bot_token in current_text:
            return super().parse_streaming_increment(new_text="", tools=tools)

        # Otherwise, keep buffering.
        return StreamingParseResult()

    def _try_parse_compact_args_format(
        self, text: str
    ) -> Optional[Tuple[str, Any, int]]:
        """
        Parse the compact tool call format:
            `[TOOL_CALLS]tool_name[ARGS]{...}`

        Tolerates common streaming variants where delimiters are missing:
            `[TOOL_CALLStool_name[ARGS{...}`

        Returns:
            (tool_name, arguments_obj, consumed_end_index) if a complete JSON arguments
            payload is present; otherwise None.
        """
        start = text.find(self._tool_calls_marker)
        if start == -1:
            return None

        i = start + len(self._tool_calls_marker)  # position after "[TOOL_CALLS"
        if i < len(text) and text[i] == "]":
            i += 1
        while i < len(text) and text[i].isspace():
            i += 1

        args_marker = "[ARGS"
        args_pos = text.find(args_marker, i)
        if args_pos == -1:
            return None

        func_name = text[i:args_pos].strip()
        if not func_name:
            return None

        j = args_pos + len(args_marker)
        if j < len(text) and text[j] == "]":
            j += 1
        while j < len(text) and text[j].isspace():
            j += 1

        if j >= len(text) or text[j] not in "{[":
            return None

        json_str, end_idx = self._extract_json_value(text, j)
        if not json_str:
            return None
        if not _is_complete_json(json_str):
            return None

        try:
            args_obj = json.loads(json_str)
        except json.JSONDecodeError:
            return None

        return func_name, args_obj, end_idx

    def _extract_json_value(
        self, text: str, json_start: int
    ) -> Tuple[Optional[str], int]:
        """
        Extract a JSON value (object or array) starting at json_start using bracket counting,
        robust to nested braces/brackets inside strings.

        Returns:
            (json_str_or_None, end_index_exclusive)
        """
        if json_start >= len(text) or text[json_start] not in "{[":
            return None, json_start

        opening = text[json_start]
        closing = "}" if opening == "{" else "]"
        depth = 0
        in_string = False
        escape_next = False

        for k in range(json_start, len(text)):
            ch = text[k]
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == opening:
                depth += 1
            elif ch == closing:
                depth -= 1
                if depth == 0:
                    return text[json_start : k + 1], k + 1

        return None, json_start

    def _extract_json_array(self, text: str) -> str:
        """
        Extract the JSON array part using bracket counting to handle nested brackets.

        :param text: The complete text containing [TOOL_CALLS] [...]
        :return: The JSON array string or None if not found
        """
        start_idx = text.find(self.bot_token)
        if start_idx == -1:
            return None

        # Start from the opening bracket after [TOOL_CALLS]
        json_start = (
            start_idx + len(self.bot_token) - 1
        )  # -1 to include the opening bracket
        bracket_count = 0
        in_string = False
        escape_next = False

        for i in range(json_start, len(text)):
            char = text[i]

            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if not in_string:
                if char == "[":
                    bracket_count += 1
                elif char == "]":
                    bracket_count -= 1
                    if bracket_count == 0:
                        return text[json_start : i + 1]

        return None

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin='[TOOL_CALLS] [{"name":"' + name + '", "arguments":',
            end="}]",
            trigger="[TOOL_CALLS]",
        )
