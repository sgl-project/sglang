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
    r"function call<\|role_sep\|>\n(.*)",
    re.DOTALL,
)

REGEX_CONTENT_PATTERN = re.compile(
    r"^(.*?)<\|message_sep\|>",
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

    def has_tool_call(self, text: str) -> bool:
        """Check if text contains a tool call marker"""
        return "function call<|role_sep|>\n" in text

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
            if "<|message_sep|>" in model_output:
                content = model_output.split("<|message_sep|>")[0]
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
        func_name = None
        cur_args = None
        m_func = REGEX_FUNCTION_CALL.search(current_text)
        if not self.tool_started:
            m_content = REGEX_CONTENT_PATTERN.search(delta_text)
            if m_content:
                content = m_content.group(1)
                self.end_content = True
            else:
                if "<|message_sep|>" in delta_text:
                    content = delta_text.split("<|message_sep|>")[0]
                    self.end_content = True
                else:
                    if not self.end_content:
                        content = delta_text
            if m_func:
                self.tool_started = True
                logger.debug("[GigaChat3] Tool call started")
            if content:
                return StreamingParseResult(normal_text=content)
        if not m_func:
            return StreamingParseResult()
        json_tail = m_func.group(1).strip()
        name_match = NAME_REGEX.search(json_tail)
        if name_match:
            func_name = name_match.group(1)
        args_match = ARGS_REGEX.search(json_tail)
        if args_match:
            cur_args = args_match.group(1).strip()
            if cur_args.endswith("</s>"):
                cur_args = cur_args[: -len("</s>")]
            if cur_args.endswith("}"):
                try:
                    candidate = cur_args[:-1].strip()
                    json.loads(candidate, strict=False)
                    cur_args = candidate
                except json.JSONDecodeError:
                    pass
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
        if cur_args is None:
            return StreamingParseResult()
        prev_args = self.prev_tool_call_arr[0].get("arguments_str", "")
        if not prev_args:
            delta_args = cur_args
        elif cur_args.startswith(prev_args):
            delta_args = cur_args[len(prev_args) :]
        else:
            logger.warning(
                f"[GigaChat3] Arguments overlap mismatch. "
                f"prev='{prev_args[:50]}...' cur='{cur_args[:50]}...'"
            )
            return StreamingParseResult()
        if not delta_args:
            return StreamingParseResult()
        self.prev_tool_call_arr[0]["arguments_str"] = cur_args
        try:
            args_dict = json.loads(cur_args, strict=False)
            self.prev_tool_call_arr[0]["arguments"] = args_dict
        except json.JSONDecodeError:
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
