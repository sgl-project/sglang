import json
import logging
import re
from typing import List

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
        match_result_list = re.findall(self.func_call_regex, text, re.DOTALL)
        calls = []
        try:
            for match_result in match_result_list:
                # Get function name
                func_detail = re.search(self.func_detail_regex, match_result, re.DOTALL)
                func_name = func_detail.group(1)
                func_args = func_detail.group(2)
                func_args = json.loads(func_args)
                # construct match_result for parse_base_json
                match_result = {"name": func_name, "parameters": func_args}
                calls.extend(self.parse_base_json(match_result, tools))
            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            # return the normal text if parsing fails
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing tool calls for DeepSeekV3 format.
        """
        self._buffer += new_text

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        CB = "<｜tool▁call▁begin｜>"
        CE = "<｜tool▁call▁end｜>"
        SEP = "<｜tool▁sep｜>"

        calls: list[ToolCallItem] = []
        normal_text = ""

        try:
            # Process as many complete/partial tool calls as possible.
            while True:
                if not self.current_tool_name_sent:
                    first_marker = self._buffer.find(CB)
                    if first_marker == -1:
                        # No tool call marker yet. If a bot token is present,
                        # keep buffering; otherwise flush as normal text.
                        if self.bot_token not in self._buffer and CB not in self._buffer:
                            text = self._buffer
                            self._buffer = ""
                            for e_token in [self.eot_token, CE]:
                                text = text.replace(e_token, "")
                            if text or calls or normal_text:
                                if text:
                                    normal_text += text
                                break
                        return StreamingParseResult(
                            normal_text=normal_text, calls=calls
                        ) if (calls or normal_text) else StreamingParseResult(
                            normal_text="", calls=[]
                        )

                    if first_marker > 0:
                        normal_text += self._buffer[:first_marker]
                        self._buffer = self._buffer[first_marker:]

                    m = re.search(CB + r"(.*?)" + SEP, self._buffer, flags=re.DOTALL)
                    if not m:
                        break
                    func_name = m.group(1).strip()

                    if self.current_tool_id == -1:
                        self.current_tool_id = 0
                        self.prev_tool_call_arr = []
                        self.streamed_args_for_tool = [""]

                    while len(self.prev_tool_call_arr) <= self.current_tool_id:
                        self.prev_tool_call_arr.append({})
                    while len(self.streamed_args_for_tool) <= self.current_tool_id:
                        self.streamed_args_for_tool.append("")

                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=func_name,
                            parameters="",
                        )
                    )
                    self.current_tool_name_sent = True
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": func_name,
                        "arguments": {},
                    }
                    self._last_arguments = ""
                    self._buffer = self._buffer[m.end():]
                    continue

                # Current tool call already named: stream its arguments.
                ce_idx = self._buffer.find(CE)
                if ce_idx >= 0:
                    args_raw = self._buffer[:ce_idx].strip()
                else:
                    # Avoid consuming a following tool call's marker as args.
                    next_cb = self._buffer.find(CB)
                    if next_cb >= 0:
                        args_raw = self._buffer[:next_cb].strip()
                    else:
                        args_raw = self._buffer.strip()

                argument_diff = (
                    args_raw[len(self._last_arguments):]
                    if args_raw.startswith(self._last_arguments)
                    else args_raw
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
                    if self.current_tool_id < len(self.streamed_args_for_tool):
                        self.streamed_args_for_tool[self.current_tool_id] += (
                            argument_diff
                        )

                if ce_idx >= 0:
                    try:
                        parsed_args = json.loads(args_raw)
                        self.prev_tool_call_arr[self.current_tool_id][
                            "arguments"
                        ] = parsed_args
                    except json.JSONDecodeError:
                        pass

                    self._buffer = self._buffer[ce_idx + len(CE):]
                    self.current_tool_id += 1
                    self.current_tool_name_sent = False
                    self._last_arguments = ""
                    continue

                # No end marker yet: keep the buffer for future deltas.
                break

            if calls or normal_text:
                return StreamingParseResult(normal_text=normal_text, calls=calls)
            return StreamingParseResult(normal_text="", calls=[])

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult(normal_text=self._buffer)

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin="<｜tool▁call▁begin｜>" + name + "<｜tool▁sep｜>",
            end="<｜tool▁call▁end｜>",
            trigger="<｜tool▁call▁begin｜>",
        )
