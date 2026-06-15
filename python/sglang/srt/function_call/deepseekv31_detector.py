import json
import logging
import re
from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.compatibility import CompatibilityEvent
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import _is_complete_json

logger = logging.getLogger(__name__)


class DeepSeekV31Detector(BaseFormatDetector):
    """
    Detector for DeepSeek V3 model function call format.

    The DeepSeek V3 format uses special Unicode tokens to delimit function calls
    with JSON code blocks for arguments.

    Format Structure:
    ```
    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>{function_name}<｜tool▁sep｜>{json_arguments}<｜tool▁calls▁end｜><｜end▁of▁sentence｜>
    ```
    Examples:
    ```
    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "Tokyo"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "Paris"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>
    ```

    Key Components:
    - Tool Calls Section: Wrapped between `<｜tool▁calls▁begin｜>` and `<｜tool▁calls▁end｜>`
    - Individual Tool Call: Wrapped between `<｜tool▁call▁begin｜>` and `<｜tool▁call▁end｜>`
    - Function Declaration: `<｜tool▁call▁begin｜>{function_name}<｜tool▁sep｜>`
    - Arguments: JSON code block between `<｜tool▁sep｜>` and `<｜tool▁call▁end｜>`
    - Supports multiple tool calls

    Reference: https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V3.1
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<｜tool▁calls▁begin｜>"
        self.eot_token = "<｜tool▁calls▁end｜>"
        self.func_call_regex = r"<｜tool▁call▁begin｜>.*?<｜tool▁call▁end｜>"
        self.func_detail_regex = (
            r"<｜tool▁call▁begin｜>(.*)<｜tool▁sep｜>(.*)<｜tool▁call▁end｜>"
        )
        self._last_arguments = ""
        self.current_tool_id = -1

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a deepseek format tool call."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])
        eot_idx = text.find(self.eot_token, idx + len(self.bot_token))
        if eot_idx != -1:
            normal_text = (text[:idx] + text[eot_idx + len(self.eot_token) :]).strip()
        match_result_list = re.findall(self.func_call_regex, text, re.DOTALL)
        if not match_result_list:
            self.compatibility.note(
                CompatibilityEvent.TRUNCATED_CALL_DROPPED,
                detail=text[idx : idx + 80],
            )
            return StreamingParseResult(normal_text=text, calls=[])
        calls = []
        for match_result in match_result_list:
            # Get function name
            func_detail = re.search(self.func_detail_regex, match_result, re.DOTALL)
            if func_detail is None:
                self.compatibility.note(
                    CompatibilityEvent.MALFORMED_JSON_DROPPED,
                    detail=match_result[:80],
                )
                continue
            func_name = func_detail.group(1)
            func_args_raw = func_detail.group(2)
            with self.compatibility.absorb(
                CompatibilityEvent.MALFORMED_JSON_DROPPED,
                json.JSONDecodeError,
                detail=f"{func_name}: {func_args_raw[:80]}",
            ) as absorbed:
                func_args = json.loads(func_args_raw)
            if absorbed.fired:
                continue
            # construct match_result for parse_base_json
            match_result = {"name": func_name, "parameters": func_args}
            calls.extend(self.parse_base_json(match_result, tools))
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing tool calls for DeepSeekV3 format.
        """
        self._buffer += new_text
        current_text = self._buffer

        # Check if we have a tool call (either the start token or individual tool call)
        has_tool_call = (
            self.bot_token in current_text or "<｜tool▁call▁begin｜>" in current_text
        )

        if not has_tool_call:
            normal_text, self._buffer = self._hold_back_partial_tokens(
                current_text,
                (
                    self.bot_token,
                    self.eot_token,
                    "<｜tool▁call▁begin｜>",
                    "<｜tool▁call▁end｜>",
                ),
            )
            for e_token in [self.eot_token, "<｜tool▁call▁end｜>"]:
                if e_token in normal_text:
                    normal_text = normal_text.replace(e_token, "")
            return StreamingParseResult(normal_text=normal_text)

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        calls: list[ToolCallItem] = []
        partial_match = re.search(
            pattern=r"<｜tool▁call▁begin｜>(.*?)<｜tool▁sep｜>(.*?)(<｜tool▁call▁end｜>|$)",
            string=current_text,
            flags=re.DOTALL,
        )
        if partial_match:
            func_name = partial_match.group(1).strip()
            func_args_raw = partial_match.group(2).strip()
            is_tool_end = partial_match.group(3)
            if is_tool_end and not _is_complete_json(func_args_raw):
                self.compatibility.note(
                    CompatibilityEvent.MALFORMED_JSON_DROPPED,
                    detail=f"{func_name}: {func_args_raw[:80]}",
                )
                calls.extend(self._close_started_malformed_call())
                self._buffer = current_text[partial_match.end(3) :]
                self.current_tool_name_sent = False
                self._last_arguments = ""
                return self._retry_streaming_tail(current_text, calls, tools)
            if func_name not in self._tool_indices and self._skip_unknown_tool(
                func_name
            ):
                if _is_complete_json(func_args_raw) and is_tool_end:
                    self._buffer = current_text[partial_match.end(3) :]
                    return self._retry_streaming_tail(current_text, calls, tools)
                return StreamingParseResult(normal_text="", calls=calls)

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

            if not self.current_tool_name_sent:
                calls.append(
                    ToolCallItem(
                        tool_index=self.current_tool_id,
                        name=func_name,
                        parameters="",
                    )
                )
                self.current_tool_name_sent = True
                # Store the tool call info for serving layer completions endpoint
                self.prev_tool_call_arr[self.current_tool_id] = {
                    "name": func_name,
                    "arguments": {},
                }
            else:
                argument_diff = (
                    func_args_raw[len(self._last_arguments) :]
                    if func_args_raw.startswith(self._last_arguments)
                    else func_args_raw
                )

                if argument_diff:
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=None,
                            parameters=argument_diff,
                        )
                    )
                    self._last_arguments += argument_diff
                    self.streamed_args_for_tool[self.current_tool_id] += argument_diff

                if _is_complete_json(func_args_raw):
                    # Update the stored arguments
                    try:
                        parsed_args = json.loads(func_args_raw)
                        self.prev_tool_call_arr[self.current_tool_id][
                            "arguments"
                        ] = parsed_args
                    except json.JSONDecodeError:
                        pass

                    # Find the end of the current tool call and remove only that part from buffer
                    if is_tool_end:
                        # Remove the completed tool call from buffer, keep any remaining content
                        self._buffer = current_text[partial_match.end(3) :]
                    else:
                        self._buffer = ""

                    result = StreamingParseResult(normal_text="", calls=calls)
                    self.current_tool_id += 1
                    self._last_arguments = ""
                    self.current_tool_name_sent = False
                    return result

        return StreamingParseResult(normal_text="", calls=calls)

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin="<｜tool▁call▁begin｜>" + name + "<｜tool▁sep｜>",
            end="<｜tool▁call▁end｜>",
            trigger="<｜tool▁call▁begin｜>",
        )
