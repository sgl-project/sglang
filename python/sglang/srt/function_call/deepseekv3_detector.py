import json
import logging
from typing import List, Optional, Tuple

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


_CALL_BEGIN = "<｜tool▁call▁begin｜>"
_CALL_END = "<｜tool▁call▁end｜>"
_SEP = "<｜tool▁sep｜>"
_JSON_OPEN = "\n```json\n"
_JSON_CLOSE = "\n```"


def _greedy_chain_search(
    text: str, a: str, b: str, c: str, d: str
) -> Optional[Tuple[str, str, str, int]]:
    # Linear equivalent of re.search(a(.*)b(.*)c(.*)d, DOTALL) for literal tokens
    # -> (g1, g2, g3, match end). The regex backtracks quadratically over every
    # a/b occurrence; greedy groups pick the last feasible d, then c, then b.
    start = text.find(a)
    d_pos = text.rfind(d)
    if start == -1 or d_pos == -1:
        return None
    c_pos = text.rfind(c, 0, d_pos)
    if c_pos == -1:
        return None
    b_pos = text.rfind(b, start + len(a), c_pos)
    if b_pos == -1:
        return None
    return (
        text[start + len(a) : b_pos],
        text[b_pos + len(b) : c_pos],
        text[c_pos + len(c) : d_pos],
        d_pos + len(d),
    )


def _iter_call_blocks(text: str):
    # Linear scan for `begin.*?end` blocks; a regex rescans to the end from every
    # begin token when the end token never arrives.
    start = text.find(_CALL_BEGIN)
    while start != -1:
        end = text.find(_CALL_END, start + len(_CALL_BEGIN))
        if end == -1:
            return
        end += len(_CALL_END)
        yield start, end
        start = text.find(_CALL_BEGIN, end)


class DeepSeekV3Detector(BaseFormatDetector):
    """
    Detector for DeepSeek V3 model function call format.

    The DeepSeek V3 format uses special Unicode tokens to delimit function calls
    with JSON code blocks for arguments.

    Format Structure:
    ```
    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>{function_name}\n```json\n{json_arguments}\n```<｜tool▁calls▁end｜><｜end▁of▁sentence｜>
    ```
    Examples:
    ```
    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_current_weather\n```json\n{"location": "Tokyo"}\n```<｜tool▁call▁end｜>\n<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_current_weather\n```json\n{"location": "Paris"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>
    ```

    Key Components:
    - Tool Calls Section: Wrapped between `<｜tool▁calls▁begin｜>` and `<｜tool▁calls▁end｜>`
    - Individual Tool Call: Wrapped between `<｜tool▁call▁begin｜>` and `<｜tool▁call▁end｜>`
    - Function Declaration: `function<｜tool▁sep｜>{function_name}`
    - Arguments: JSON code block between ````json` and ````
    - Supports multiple tool calls

    Reference: https://huggingface.co/deepseek-ai/DeepSeek-V3-0324?chat_template=default
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<｜tool▁calls▁begin｜>"
        self.eot_token = "<｜tool▁calls▁end｜>"
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
        match_result_list = [text[s:e] for s, e in _iter_call_blocks(text)]
        calls = []
        try:
            for match_result in match_result_list:
                # Get function name
                func_detail = _greedy_chain_search(
                    match_result, _CALL_BEGIN, _SEP, _JSON_OPEN, _JSON_CLOSE + _CALL_END
                )
                func_name = func_detail[1]
                func_args = func_detail[2]
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
        current_text = self._buffer

        # Check if we have a tool call (either the start token or individual tool call)
        has_tool_call = (
            self.bot_token in current_text or "<｜tool▁call▁begin｜>" in current_text
        )

        if not has_tool_call:
            self._buffer = ""
            for e_token in [self.eot_token, "```", "<｜tool▁call▁end｜>"]:
                if e_token in new_text:
                    new_text = new_text.replace(e_token, "")
            return StreamingParseResult(normal_text=new_text)

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        calls: list[ToolCallItem] = []
        try:
            partial_match = _greedy_chain_search(
                current_text, _CALL_BEGIN, _SEP, _JSON_OPEN, _JSON_CLOSE
            )
            if partial_match:
                func_name = partial_match[1].strip()
                func_args_raw = partial_match[2].strip()

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
                        self.streamed_args_for_tool[self.current_tool_id] += (
                            argument_diff
                        )

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
                        block = next(_iter_call_blocks(current_text), None)
                        if block is not None:
                            # Remove the completed tool call from buffer, keep any remaining content
                            self._buffer = current_text[block[1] :]
                        else:
                            self._buffer = ""

                        result = StreamingParseResult(normal_text="", calls=calls)
                        self.current_tool_id += 1
                        self._last_arguments = ""
                        self.current_tool_name_sent = False
                        return result

            return StreamingParseResult(normal_text="", calls=calls)

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult(normal_text=current_text)

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin="<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>"
            + name
            + "\n```json\n",
            end="\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            trigger="<｜tool▁calls▁begin｜>",
        )
