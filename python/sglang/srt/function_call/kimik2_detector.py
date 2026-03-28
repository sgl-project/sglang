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

_KIMI_K2_SPECIAL_TOKENS = [
    "<|tool_calls_section_begin|>",
    "<|tool_calls_section_end|>",
    "<|tool_call_begin|>",
    "<|tool_call_end|>",
    "<|tool_call_argument_begin|>",
]


def _strip_special_tokens(text: str) -> str:
    """Remove all Kimi-K2 tool-call special tokens from text."""
    for token in _KIMI_K2_SPECIAL_TOKENS:
        text = text.replace(token, "")
    return text


class KimiK2Detector(BaseFormatDetector):
    """
    Detector for Kimi K2 / K2.5 model function call format.

    Format Structure:
    ```
    <|tool_calls_section_begin|>
    <|tool_call_begin|>functions.{func_name}:{index}<|tool_call_argument_begin|>{json_args}<|tool_call_end|>
    <|tool_calls_section_end|>
    ```

    Reference: https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/docs/tool_call_guidance.md
    """

    def __init__(self):
        super().__init__()

        self.bot_token: str = "<|tool_calls_section_begin|>"
        self.eot_token: str = "<|tool_calls_section_end|>"

        self.tool_call_start_token: str = "<|tool_call_begin|>"
        self.tool_call_end_token: str = "<|tool_call_end|>"
        self.tool_call_argument_begin_token: str = "<|tool_call_argument_begin|>"

        # Support hyphenated function names (common in MCP tools, e.g. mcp__portal__search-documents)
        self.tool_call_regex = re.compile(
            r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[\w.\-]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>\{.*?\})\s*<\|tool_call_end\|>",
            re.DOTALL,
        )

        self.stream_tool_call_portion_regex = re.compile(
            r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[\w.\-]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>\{.*)",
            re.DOTALL,
        )

        self._last_arguments = ""

        # Robust parser for ids like "functions.search:0", "functions.mcp__search-docs:0", or fallback "search:0"
        self.tool_call_id_regex = re.compile(
            r"^(?:functions\.)?(?P<name>[\w.\-]+):(?P<index>\d+)$"
        )

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a KimiK2 format tool call."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=text, calls=[])
        try:
            # there are two possible captures - between tags, or between a
            # tag and end-of-string so the result of
            # findall is an array of tuples where one is a function call and
            # the other is None
            function_call_tuples = self.tool_call_regex.findall(text)

            logger.debug("function_call_tuples: %s", function_call_tuples)

            tool_calls = []
            for match in function_call_tuples:
                function_id, function_args = match
                m = self.tool_call_id_regex.match(function_id)
                if not m:
                    logger.warning("Unexpected tool_call_id format: %s", function_id)
                    continue
                function_name = m.group("name")
                function_idx = int(m.group("index"))

                logger.debug(f"function_name {function_name}")

                tool_calls.append(
                    ToolCallItem(
                        tool_index=function_idx,
                        name=function_name,
                        parameters=function_args,
                    )
                )

            content = text[: text.find(self.bot_token)]
            return StreamingParseResult(normal_text=content, calls=tool_calls)

        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            # return the normal text if parsing fails
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing tool calls for KimiK2 format.
        """
        self._buffer += new_text
        current_text = self._buffer

        # Check if we have a tool call (either the start token or individual tool call)
        has_tool_call = (
            self.bot_token in current_text or self.tool_call_start_token in current_text
        )

        if not has_tool_call:
            self._buffer = ""
            normal_text = _strip_special_tokens(new_text)
            return StreamingParseResult(normal_text=normal_text)

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        calls: list[ToolCallItem] = []
        try:
            match = self.stream_tool_call_portion_regex.search(current_text)
            if match:
                function_id = match.group("tool_call_id")
                function_args = match.group("function_arguments")

                m = self.tool_call_id_regex.match(function_id)
                if not m:
                    logger.warning("Unexpected tool_call_id format: %s", function_id)
                    return StreamingParseResult(normal_text="", calls=calls)
                function_name = m.group("name")

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
                            name=function_name,
                            parameters="",
                        )
                    )
                    self.current_tool_name_sent = True
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": function_name,
                        "arguments": {},
                    }
                else:
                    argument_diff = (
                        function_args[len(self._last_arguments) :]
                        if function_args.startswith(self._last_arguments)
                        else function_args
                    )

                    parsed_args_diff = argument_diff.split(self.tool_call_end_token, 1)[
                        0
                    ]

                    if parsed_args_diff:
                        calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                name=None,
                                parameters=parsed_args_diff,
                            )
                        )
                        self._last_arguments += parsed_args_diff
                        self.streamed_args_for_tool[
                            self.current_tool_id
                        ] += parsed_args_diff

                    parsed_args = function_args.split(self.tool_call_end_token, 1)[0]
                    if _is_complete_json(parsed_args):
                        try:
                            parsed_args = json.loads(parsed_args)
                            self.prev_tool_call_arr[self.current_tool_id][
                                "arguments"
                            ] = parsed_args
                        except json.JSONDecodeError:
                            pass

                        # Find the end of the current tool call and remove only that part from buffer
                        tool_call_end_pattern = (
                            r"<\|tool_call_begin\|>.*?<\|tool_call_end\|>"
                        )
                        end_match = re.search(
                            tool_call_end_pattern, current_text, re.DOTALL
                        )
                        if end_match:
                            self._buffer = current_text[end_match.end() :]
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
            return StreamingParseResult(normal_text=_strip_special_tokens(current_text))

    def structure_info(self) -> _GetInfoFunc:
        """Return function that creates StructureInfo for guided generation."""

        def get_info(name: str) -> StructureInfo:
            return StructureInfo(
                begin=f"<|tool_calls_section_begin|><|tool_call_begin|>functions.{name}:0<|tool_call_argument_begin|>",
                end="<|tool_call_end|><|tool_calls_section_end|>",
                trigger="<|tool_calls_section_begin|>",
            )

        return get_info
