import json
import logging
import re
from typing import List

from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.ebnf_composer import EBNFComposer
from sglang.srt.function_call.utils import _is_complete_json
from sglang.srt.openai_api.protocol import Tool

logger = logging.getLogger(__name__)


class DeepSeekV3Detector(BaseFormatDetector):
    """
    Detector for DeepSeek models.
    Assumes function call format:
      '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_current_weather\n```json\n{"location": "Tokyo"}\n```<｜tool▁call▁end｜>\n<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_current_weather\n```json\n{"location": "Paris"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<｜tool▁calls▁begin｜>"
        self.eot_token = "<｜tool▁calls▁end｜>"
        self.func_call_regex = r"<｜tool▁call▁begin｜>.*?<｜tool▁call▁end｜>"
        self.func_detail_regex = r"<｜tool▁call▁begin｜>(.*)<｜tool▁sep｜>(.*)\n```json\n(.*)\n```<｜tool▁call▁end｜>"
        self._last_arguments = ""

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
                func_name = func_detail.group(2)
                func_args = func_detail.group(3)
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

        if self.bot_token not in current_text:
            self._buffer = ""
            for e_token in [self.eot_token, "```", "<｜tool▁call▁end｜>"]:
                if e_token in new_text:
                    new_text = new_text.replace(e_token, "")
            return StreamingParseResult(normal_text=new_text)

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = {
                tool.function.name: i
                for i, tool in enumerate(tools)
                if tool.function and tool.function.name
            }

        calls: list[ToolCallItem] = []
        try:
            partial_match = re.search(
                pattern=r"<｜tool▁call▁begin｜>(.*)<｜tool▁sep｜>(.*)\n```json\n(.*)",
                string=current_text,
                flags=re.DOTALL,
            )
            if partial_match:
                func_name = partial_match.group(2).strip()
                func_args_raw = partial_match.group(3).strip()

                if not self.current_tool_name_sent:
                    calls.append(
                        ToolCallItem(
                            tool_index=self._tool_indices.get(func_name, 0),
                            name=func_name,
                            parameters="",
                        )
                    )
                    self.current_tool_name_sent = True
                else:
                    argument_diff = (
                        func_args_raw[len(self._last_arguments) :]
                        if func_args_raw.startswith(self._last_arguments)
                        else func_args_raw
                    )

                    if argument_diff:
                        calls.append(
                            ToolCallItem(
                                tool_index=self._tool_indices.get(func_name, 0),
                                name=None,
                                parameters=argument_diff,
                            )
                        )
                        self._last_arguments += argument_diff

                    if _is_complete_json(func_args_raw):
                        result = StreamingParseResult(normal_text="", calls=calls)
                        self._buffer = ""
                        self._last_arguments = ""
                        self.current_tool_name_sent = False
                        return result

            return StreamingParseResult(normal_text="", calls=calls)

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult(normal_text=current_text)

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=">" + name + "\n```json\n",
            end="\n```<",
            trigger=">" + name + "\n```json\n",
        )

    def build_ebnf(self, tools: List[Tool]):
        return EBNFComposer.build_ebnf(
            tools,
            sequence_start_token=self.bot_token,
            sequence_end_token=self.eot_token,
            tool_call_separator="",
            call_rule_fmt='"<｜tool▁call▁begin｜>function<｜tool▁sep｜>{name}\\n```json\\n" {arguments_rule} "\\n```<｜tool▁call▁end｜>"',
            function_format="json",
        )
