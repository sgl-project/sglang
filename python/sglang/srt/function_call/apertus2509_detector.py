import json
import logging
from typing import Any, List, Optional, Tuple

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


class Apertus2509Detector(BaseFormatDetector):
    """
    Detector for Apertus 2509 tool/function call format
    ```
    <|tools_prefix|>[{"tool1": {...}}, {"tool2": {...}}]<|tools_suffix|>
    ```

    Notes:
    - Each list element is a single-key object: {"<tool_name>": <arguments_object>}
    - The list can contain multiple tool calls separated by ", "
    - This is distinct from the OpenAI-style {"name": "...", "arguments": {...}} objects
    """

    def __init__(self):
        super().__init__()
        self.bot = "<|tools_prefix|>["
        self.suffix = "<|tools_suffix|>"
        self._in_tools_block: bool = False

    def has_tool_call(self, text: str) -> bool:
        return self.bot in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Extract all Apertus tools blocks and parse their JSON payloads.
        """
        if not self.has_tool_call(text):
            return StreamingParseResult(normal_text=text, calls=[])

        calls: List[ToolCallItem] = []
        normal_parts: List[str] = []
        cursor = 0

        while True:
            if (start := text.find(self.bot, cursor)) == -1:
                normal_parts.append(text[cursor:])
                break

            normal_parts.append(text[cursor:start])
            tool_part = text[start:]
            parsed_arr, json_end = self._try_parse_json_array(tool_part)
            if parsed_arr is None:
                normal_parts.append(tool_part)
                break

            if (suffix_pos := tool_part.find(self.suffix, json_end)) == -1:
                normal_parts.append(tool_part)
                break

            calls.extend(
                self._parse_apertus_call_list(
                    parsed_arr, tools, tool_index_offset=len(calls)
                )
            )

            cursor = start + suffix_pos + len(self.suffix)

        return StreamingParseResult(
            normal_text="".join(normal_parts).strip(), calls=calls
        )

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing for Apertus tool calls.

        - Streams any normal text before `<|tools_prefix|>[` immediately.
        - Buffers tool calls until we have a complete tools block, then emits:
          - Tool name (empty args), then
          - Full JSON arguments string
        """
        self._buffer += new_text
        out_normal = ""
        out_calls: List[ToolCallItem] = []

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        while True:
            if not self._in_tools_block:
                if (pos := self._buffer.find(self.bot)) > 0:
                    out_normal += self._buffer[:pos]
                    self._buffer = self._buffer[pos:]
                elif pos == -1:
                    if partial_bot := self._ends_with_partial_token(
                        self._buffer, self.bot
                    ):
                        out_normal += self._buffer[:-partial_bot]
                        self._buffer = self._buffer[-partial_bot:]
                    else:
                        out_normal += self._buffer
                        self._buffer = ""
                    return StreamingParseResult(normal_text=out_normal, calls=out_calls)

                self._in_tools_block = True

            if not self._buffer.startswith(self.bot):
                if (marker_pos := self._buffer.find(self.bot)) == -1:
                    out_normal += self._buffer
                    self._buffer = ""
                    self._in_tools_block = False
                    return StreamingParseResult(normal_text=out_normal, calls=out_calls)
                out_normal += self._buffer[:marker_pos]
                self._buffer = self._buffer[marker_pos:]
                continue

            parsed_arr, suffix_pos = self._try_parse_json_array(self._buffer)
            if parsed_arr is None:
                if self.suffix in self._buffer:
                    out_normal += self._buffer
                    self._buffer = ""
                    self._in_tools_block = False
                    return StreamingParseResult(normal_text=out_normal, calls=out_calls)
                return StreamingParseResult(normal_text=out_normal, calls=out_calls)

            while suffix_pos < len(self._buffer) and self._buffer[suffix_pos].isspace():
                suffix_pos += 1

            if not self._buffer.startswith(self.suffix, suffix_pos):
                return StreamingParseResult(normal_text=out_normal, calls=out_calls)

            if self.current_tool_id == -1:
                self.current_tool_id = 0

            for item in parsed_arr:
                name, args = self._apertus_obj_to_call(item)
                if name is None:
                    continue
                if args is None:
                    args = {}

                if (
                    name not in self._tool_indices
                    and not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get()
                ):
                    logger.warning(
                        f"Model attempted to call undefined function: {name}"
                    )
                    continue

                tool_id = self.current_tool_id
                self.current_tool_id += 1

                args_json = json.dumps(args, ensure_ascii=False)

                while len(self.prev_tool_call_arr) <= tool_id:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= tool_id:
                    self.streamed_args_for_tool.append("")

                self.prev_tool_call_arr[tool_id] = {"name": name, "arguments": args}
                self.streamed_args_for_tool[tool_id] = args_json

                # Emit tool name first, then full args (OpenAI streaming semantics)
                out_calls.append(
                    ToolCallItem(tool_index=tool_id, name=name, parameters="")
                )
                out_calls.append(
                    ToolCallItem(tool_index=tool_id, name=None, parameters=args_json)
                )

            # Consume the parsed tools block and reset state
            self._buffer = self._buffer[suffix_pos + len(self.suffix) :]
            self._in_tools_block = False

            if out_calls:
                # Flush normal text after the tools block, but keep a tool marker or its partial prefix in the buffer for the next stream
                if (marker_pos := self._buffer.find(self.bot)) > 0:
                    out_normal += self._buffer[:marker_pos]
                    self._buffer = self._buffer[marker_pos:]
                elif marker_pos == -1:
                    if partial_bot := self._ends_with_partial_token(
                        self._buffer, self.bot
                    ):
                        out_normal += self._buffer[:-partial_bot]
                        self._buffer = self._buffer[-partial_bot:]
                    else:
                        out_normal += self._buffer
                        self._buffer = ""
                return StreamingParseResult(normal_text=out_normal, calls=out_calls)

            continue

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin='<|tools_prefix|>[{"' + name + '": ',
            end="}]<|tools_suffix|>",
            trigger="<|tools_prefix|>",
        )

    def _apertus_obj_to_call(self, obj: Any) -> Tuple[Optional[str], Optional[Any]]:
        """
        Convert a single Apertus tool-call object to (name, arguments).

        Expected shape: {"tool_name": {...}}.
        """
        if not isinstance(obj, dict) or not obj:
            return None, None
        name = next(iter(obj.keys()))
        return name, obj.get(name)

    def _parse_apertus_call_list(
        self, arr: Any, tools: List[Tool], tool_index_offset: int = 0
    ) -> List[ToolCallItem]:

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        calls: List[ToolCallItem] = []
        for item in arr:
            name, args = self._apertus_obj_to_call(item)
            if name is None:
                continue
            if args is None:
                args = {}

            if (
                name not in self._tool_indices
                and not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get()
            ):
                logger.warning(f"Model attempted to call undefined function: {name}")
                continue

            calls.append(
                ToolCallItem(
                    tool_index=tool_index_offset + len(calls),
                    name=name,
                    parameters=json.dumps(args, ensure_ascii=False),
                )
            )

        return calls

    def _try_parse_json_array(self, text: str) -> Tuple[Optional[Any], int]:
        """
        Returns: (parsed_array_or_None, end_index_exclusive_in_text)
        """
        if (start_idx := text.find(self.bot)) == -1:
            return None, 0

        json_start = start_idx + len(self.bot) - 1  # points to '['
        try:
            parsed, end_idx = json.JSONDecoder().raw_decode(text, json_start)
        except json.JSONDecodeError:
            return None, 0

        if isinstance(parsed, list):
            return parsed, end_idx
        return [parsed], end_idx
