import json
import logging
import re
from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)

REGEX_FUNCTION_CALL = re.compile(
    r"(?:function call<\|role_sep\|>\n|<\|function_call\|>)(.*)",
    re.DOTALL,
)

REGEX_CONTENT_PATTERN = re.compile(
    r"^(.*?)(?:<\|message_sep\|>|<\|function_call\|>)",
    re.DOTALL,
)

NAME_REGEX = re.compile(
    r'"name"\s*:\s*"([^"]*)"',
    re.DOTALL,
)

ARGS_REGEX = re.compile(
    r'"arguments"\s*:\s*(.*)',
    re.DOTALL,
)


class GigaChat3Detector(BaseFormatDetector):
    def __init__(self) -> None:
        super().__init__()
        self.tool_started: bool = False
        self.tool_name_sent: bool = False
        self.end_content: bool = False
        self._buffer: str = ""
        self.prev_tool_call_arr: list[dict] = []
        self._func_json_start: int = -1
        self._cached_func_name: str | None = None
        self._args_start: int = -1
        self._args_scan_pos: int = -1
        self._args_brace_depth: int = 0
        self._args_in_string: bool = False
        self._args_escape: bool = False
        self._args_close_idx: int | None = None
        self._streamed_args_end: int = -1
        self._parsed_args_dict: dict | None = None

    def _advance_args_scan(self, text: str, end: int) -> None:
        if self._args_close_idx is not None and self._args_close_idx <= end:
            return
        i = self._args_scan_pos
        while i < end:
            ch = text[i]
            if self._args_in_string:
                if self._args_escape:
                    self._args_escape = False
                elif ch == "\\":
                    self._args_escape = True
                elif ch == '"':
                    self._args_in_string = False
            else:
                if ch == '"':
                    self._args_in_string = True
                elif ch == "{":
                    self._args_brace_depth += 1
                elif ch == "}":
                    self._args_brace_depth -= 1
                    if self._args_brace_depth == 0:
                        self._args_close_idx = i + 1
                        i += 1
                        break
            i += 1
        self._args_scan_pos = i

    def has_tool_call(self, text: str) -> bool:
        """Check if text contains a tool call marker"""
        return "function call<|role_sep|>\n" in text or "<|function_call|>" in text

    def detect_and_parse(
        self,
        text: str,
        tools: List[Tool],
    ) -> StreamingParseResult:
        """
        Non-streaming parsing of complete model output.
        Extracts tool calls and content from the full text.
        """
        logger.debug(f"[GigaChat3] detect_and_parse: {text}")
        model_output = text
        function_call = None
        content = None
        if model_output.rstrip().endswith("</s>"):
            model_output = model_output[: model_output.rfind("</s>")]
        m_func = REGEX_FUNCTION_CALL.search(model_output)
        if m_func:
            try:
                function_call = json.loads(m_func.group(1), strict=False)
                if not (
                    isinstance(function_call, dict)
                    and "name" in function_call
                    and "arguments" in function_call
                ):
                    function_call = None
                elif not isinstance(function_call["arguments"], dict):
                    function_call = None
            except json.JSONDecodeError as e:
                logger.warning(f"[GigaChat3] JSON decode error: {e}")
                return StreamingParseResult(
                    normal_text=model_output,
                    calls=[],
                )
        m_content = REGEX_CONTENT_PATTERN.search(model_output)
        if m_content:
            content = m_content.group(1)
        else:
            content = model_output
        if not function_call:
            return StreamingParseResult(normal_text=content, calls=[])
        name = function_call["name"]
        args = function_call["arguments"]
        match_result = {"name": name, "arguments": args}
        calls = self.parse_base_json(match_result, tools)
        return StreamingParseResult(normal_text=content, calls=calls)

    def parse_streaming_increment(
        self,
        new_text: str,
        tools: List[Tool],
    ) -> StreamingParseResult:
        """
        Streaming parser for incremental text chunks.
        Maintains state across calls to build complete tool calls.
        """
        if not new_text:
            return StreamingParseResult()
        logger.debug(f"[GigaChat3] parse_streaming_increment: '{new_text}'")
        self._buffer += new_text
        current_text = self._buffer
        delta_text = new_text
        content = None

        if self._func_json_start == -1:
            m_func = REGEX_FUNCTION_CALL.search(current_text)
            if m_func:
                self._func_json_start = m_func.start(1)

        if not self.tool_started:
            m_content = REGEX_CONTENT_PATTERN.search(delta_text)
            if m_content:
                content = m_content.group(1)
                self.end_content = True
            else:
                if not self.end_content:
                    content = delta_text
            if self._func_json_start != -1:
                self.tool_started = True
                logger.debug("[GigaChat3] Tool call started")
            if content:
                return StreamingParseResult(normal_text=content)
        if self._func_json_start == -1:
            return StreamingParseResult()

        if self._cached_func_name is None:
            name_match = NAME_REGEX.search(current_text, self._func_json_start)
            if name_match:
                self._cached_func_name = name_match.group(1)
        func_name = self._cached_func_name

        if self._args_start == -1:
            args_match = ARGS_REGEX.search(current_text, self._func_json_start)
            if args_match:
                pos = args_match.start(1)
                n = len(current_text)
                while pos < n and current_text[pos].isspace():
                    pos += 1
                if pos < n:
                    self._args_start = pos
                    self._args_scan_pos = pos
                    self._streamed_args_end = pos

        args_end = -1
        if self._args_start != -1:
            end = len(current_text)
            while end > self._args_start and current_text[end - 1].isspace():
                end -= 1
            if current_text.endswith("</s>", self._args_start, end):
                end -= 4
                while end > self._args_start and current_text[end - 1].isspace():
                    end -= 1
            self._advance_args_scan(current_text, end)
            if (
                self._args_close_idx is not None
                and end > self._args_close_idx
                and current_text[end - 1] == "}"
            ):
                cand_end = end - 1
                while (
                    cand_end > self._args_start and current_text[cand_end - 1].isspace()
                ):
                    cand_end -= 1
                if cand_end == self._args_close_idx:
                    if self._parsed_args_dict is None:
                        candidate = current_text[self._args_start : cand_end]
                        try:
                            self._parsed_args_dict = json.loads(candidate, strict=False)
                        except json.JSONDecodeError:
                            self._parsed_args_dict = None
                    if self._parsed_args_dict is not None:
                        end = cand_end
                else:
                    candidate = current_text[self._args_start : cand_end]
                    try:
                        self._parsed_args_dict = json.loads(candidate, strict=False)
                        end = cand_end
                    except json.JSONDecodeError:
                        pass
            if end > self._args_start:
                args_end = end

        calls: List[ToolCallItem] = []
        if not self.prev_tool_call_arr:
            self.prev_tool_call_arr.append({})
        if not self.tool_name_sent:
            if not func_name:
                return StreamingParseResult()
            self.tool_name_sent = True
            self.prev_tool_call_arr[0]["name"] = func_name
            logger.debug(f"[GigaChat3] Sending tool name: {func_name}")
            calls.append(
                ToolCallItem(
                    tool_index=0,
                    name=func_name,
                    parameters="",
                )
            )
            return StreamingParseResult(calls=calls)
        if args_end == -1:
            return StreamingParseResult()
        if args_end < self._streamed_args_end:
            prev_args = self.prev_tool_call_arr[0].get("arguments_str", "")
            cur_args = current_text[self._args_start : args_end]
            logger.warning(
                f"[GigaChat3] Arguments overlap mismatch. "
                f"prev='{prev_args[:50]}...' cur='{cur_args[:50]}...'"
            )
            return StreamingParseResult()
        delta_args = current_text[self._streamed_args_end : args_end]
        if not delta_args:
            return StreamingParseResult()
        cur_args = current_text[self._args_start : args_end]
        self._streamed_args_end = args_end
        self.prev_tool_call_arr[0]["arguments_str"] = cur_args
        if self._args_close_idx is not None and args_end == self._args_close_idx:
            if self._parsed_args_dict is None:
                try:
                    self._parsed_args_dict = json.loads(cur_args, strict=False)
                except json.JSONDecodeError:
                    self._parsed_args_dict = {}
            self.prev_tool_call_arr[0]["arguments"] = self._parsed_args_dict
        else:
            self.prev_tool_call_arr[0]["arguments"] = {}
        logger.debug(f"[GigaChat3] Sending args delta: '{delta_args[:100]}...'")
        calls.append(
            ToolCallItem(
                tool_index=0,
                name=None,
                parameters=delta_args,
            )
        )
        return StreamingParseResult(calls=calls)

    def supports_structural_tag(self) -> bool:
        """GigaChat3 does not use structural tags"""
        return False

    def structure_info(self) -> _GetInfoFunc:
        """Not applicable for GigaChat3"""
        raise NotImplementedError(
            "GigaChat3Detector does not support structural_tag format."
        )
