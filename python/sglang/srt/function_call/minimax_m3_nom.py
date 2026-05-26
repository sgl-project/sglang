import json
import logging
from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)

from ._llm_nom.m3_text import M3TextParser

logger = logging.getLogger(__name__)


class MinimaxM3NomDetector(BaseFormatDetector):
    def __init__(self):
        super().__init__()
        self._parser = None
        self._with_reasoning = False
        self._reasoning_prefix = ""
        self._error = None

    def _init_parser_with_tools(
        self,
        *,
        tools: List[Tool],
    ):
        if self._parser is None:
            params = {
                "with_reasoning": self._with_reasoning,
                "reasoning_prefix": self._reasoning_prefix,
                "functions": (
                    {
                        tool.function.name: {
                            "parameters": tool.function.parameters,
                        }
                        for tool in tools
                    }
                    if tools
                    else None
                ),
            }
            self._parser = M3TextParser(**params)
        return self._parser

    def has_tool_call(self, text: str) -> bool:
        return ("]<]minimax[>[<tool_call>") in text

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        if self._error is not None:
            return StreamingParseResult()
        self._init_parser_with_tools(tools=tools)
        try:
            self._parser.update(new_text)
        except Exception as e:
            self._error = e
        delta = self._parser.get_delta()
        if delta is None:
            return StreamingParseResult()

        normal_text = delta.get("content", "")
        calls: List[ToolCallItem] = []

        for tc in delta.get("tool_calls", []):
            func = tc.get("function", {})
            idx = tc.get("index", 0)
            name = func.get("name")
            arguments = func.get("arguments")

            if name is not None:
                if self.current_tool_id == -1:
                    self.current_tool_id = 0
                while len(self.prev_tool_call_arr) <= idx:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= idx:
                    self.streamed_args_for_tool.append("")
                self.prev_tool_call_arr[idx] = {"name": name, "arguments": {}}
                calls.append(ToolCallItem(tool_index=idx, name=name, parameters=""))

            if arguments is not None:
                while len(self.streamed_args_for_tool) <= idx:
                    self.streamed_args_for_tool.append("")
                self.streamed_args_for_tool[idx] += arguments
                if idx < len(self.prev_tool_call_arr):
                    try:
                        self.prev_tool_call_arr[idx]["arguments"] = json.loads(
                            self.streamed_args_for_tool[idx]
                        )
                    except json.JSONDecodeError:
                        pass
                self.current_tool_id = idx
                calls.append(
                    ToolCallItem(tool_index=idx, name=None, parameters=arguments)
                )

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        self._init_parser_with_tools(tools=tools)
        try:
            self._parser.update(text)
        except Exception as e:
            self._error = e
        delta = self._parser.get_delta()
        if delta is None:
            return StreamingParseResult(normal_text=text, calls=[])

        normal_text = delta.get("content", "")
        calls: List[ToolCallItem] = []
        for tc in delta.get("tool_calls", []):
            func = tc.get("function", {})
            name = func.get("name", "")
            arguments = func.get("arguments", "{}")
            try:
                args_dict = json.loads(arguments)
            except json.JSONDecodeError:
                args_dict = {}
            try:
                calls.extend(
                    self.parse_base_json({"name": name, "arguments": args_dict}, tools)
                )
            except Exception:
                logger.warning("invalid tool call for %s dropped", name)

        return StreamingParseResult(normal_text=normal_text, calls=calls)
