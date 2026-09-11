import json
import logging
import re
from typing import List, Literal, Optional, Union

from xgrammar import StructuralTag

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.kimik3_format import (
    MESSAGE_CLOSE,
    RESPONSE_CLOSE,
    RESPONSE_OPEN,
    TOOLS_CLOSE,
    TOOLS_OPEN,
    partial_suffix_len,
    strip_partial_marker_suffix,
    strip_response_wrappers,
)
from sglang.srt.function_call.kimik3_structural_tag import (
    get_kimik3_auto_tool_call_structural_tag,
    get_kimik3_structural_tag,
)

logger = logging.getLogger(__name__)

_CALL_RE = re.compile(
    r"<\|open\|>call\s+(?P<attrs>(?:(?!<\|sep\|>).)*?)<\|sep\|>"
    r"(?P<body>.*?)<\|close\|>call<\|sep\|>",
    re.DOTALL,
)
_ARG_RE = re.compile(
    r"<\|open\|>argument\s+(?P<attrs>(?:(?!<\|sep\|>).)*?)<\|sep\|>"
    r"(?P<val>.*?)<\|close\|>argument<\|sep\|>",
    re.DOTALL,
)
_ATTR_RE = re.compile(r'(?P<k>\w+)="(?P<v>[^"]*)"')


def _unescape_attr(value: str) -> str:
    return value.replace("&quot;", '"').replace("&amp;", "&")


def _parse_attrs(attrs: str) -> dict:
    return {m["k"]: _unescape_attr(m["v"]) for m in _ATTR_RE.finditer(attrs)}


class KimiK3Detector(BaseFormatDetector):
    """Detector for the Kimi K3 XTML tool-call format.

    K3 emits tool calls in a ``tools`` channel built from dedicated special
    tokens; the plain reply lives in a preceding ``response`` channel:

    ```
    <|open|>response<|sep|>text<|close|>response<|sep|>
    <|open|>tools<|sep|>
      <|open|>call tool="name" index="1"<|sep|>
        <|open|>argument key="k" type="string"<|sep|>raw text<|close|>argument<|sep|>
      <|close|>call<|sep|>
    <|close|>tools<|sep|>
    ```

    ``type="string"`` argument values are raw text; other types are
    JSON-decoded. Attribute values reverse the template's ``&amp;``/``&quot;``
    escaping.
    """

    def __init__(self):
        super().__init__()
        self.bot_token = TOOLS_OPEN
        self.eot_token = TOOLS_CLOSE
        self._sent_normal_idx = 0

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def supports_structural_tag(self) -> bool:
        return True

    def parses_required_natively(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError(
            "Kimi K3 uses its model-native structural tag implementation"
        )

    def get_auto_tool_call_structural_tag(
        self,
        tools: Union[List[Tool], None] = None,
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
    ) -> Optional[StructuralTag]:
        return get_kimik3_auto_tool_call_structural_tag(
            tools or [],
            thinking_mode=thinking_mode,
            parallel_tool_calls=parallel_tool_calls,
        )

    def get_structural_tag(
        self,
        tools: Union[List[Tool], None] = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required"]] = "auto",
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
    ) -> StructuralTag:
        return get_kimik3_structural_tag(
            tools=tools or [],
            tool_choice=tool_choice,
            thinking_mode=thinking_mode,
            parallel_tool_calls=parallel_tool_calls,
        )

    def _decode_call(self, attrs: str, body: str) -> dict | None:
        call_attrs = _parse_attrs(attrs)
        tool_name = call_attrs.get("tool", "")
        if not tool_name:
            return None
        arguments = {}
        for arg in _ARG_RE.finditer(body):
            arg_attrs = _parse_attrs(arg["attrs"])
            key = arg_attrs.get("key", "")
            arg_type = arg_attrs.get("type", "string")
            raw_value = arg["val"]
            if arg_type == "string":
                arguments[key] = raw_value
            else:
                try:
                    arguments[key] = json.loads(raw_value)
                except json.JSONDecodeError:
                    arguments[key] = raw_value
        return {
            "name": tool_name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        }

    def _parse_calls(self, section: str) -> List[dict]:
        return [
            call
            for m in _CALL_RE.finditer(section)
            if (call := self._decode_call(m["attrs"], m["body"])) is not None
        ]

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        open_idx = text.find(self.bot_token)
        if open_idx == -1:
            return StreamingParseResult(normal_text=strip_response_wrappers(text))
        # Computed outside the try so the error path can reuse it instead of
        # falling back to raw text, which would ship the XTML tools markup to
        # the client.
        before = strip_response_wrappers(text[:open_idx])
        try:
            section_start = open_idx + len(self.bot_token)
            close_idx = text.find(self.eot_token, section_start)
            section = (
                text[section_start:]
                if close_idx == -1
                else text[section_start:close_idx]
            )
            calls = [
                ToolCallItem(
                    tool_index=i,
                    name=call["name"],
                    parameters=call["arguments"],
                )
                for i, call in enumerate(self._parse_calls(section))
            ]
            return StreamingParseResult(normal_text=before, calls=calls)
        except Exception as e:
            logger.error("Error in Kimi K3 detect_and_parse: %s", e, exc_info=True)
            return StreamingParseResult(normal_text=before)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._buffer += new_text
        try:
            open_idx = self._buffer.find(self.bot_token)
            if open_idx == -1:
                return StreamingParseResult(normal_text=self._emit_normal_text())

            normal_text = self._emit_normal_text(limit=open_idx)
            section = self._buffer[open_idx + len(self.bot_token) :]
            calls = []
            parsed = self._parse_calls(section)
            for call in parsed[self.current_tool_id + 1 :]:
                self.current_tool_id += 1
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")
                self.prev_tool_call_arr[self.current_tool_id] = {
                    "name": call["name"],
                    "arguments": json.loads(call["arguments"]),
                }
                self.streamed_args_for_tool[self.current_tool_id] = call["arguments"]
                calls.append(
                    ToolCallItem(
                        tool_index=self.current_tool_id,
                        name=call["name"],
                        parameters=call["arguments"],
                    )
                )
            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(
                "Error in Kimi K3 parse_streaming_increment: %s", e, exc_info=True
            )
            # _sent_normal_idx indexes into _buffer, so it must be reset with it;
            # otherwise every later _emit_normal_text sees limit <= _sent_normal_idx
            # and silently drops the rest of the response.
            self._buffer = ""
            self._sent_normal_idx = 0
            return StreamingParseResult()

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        open_idx = self._buffer.find(self.bot_token)
        if open_idx != -1:
            section = self._buffer[open_idx + len(self.bot_token) :]
            if not self._parse_calls(section):
                logger.warning(
                    "Kimi K3 tools section ended with no complete tool call; "
                    "dropping %d buffered chars",
                    len(section),
                )
            return StreamingParseResult()
        pending = self._emit_normal_text(limit=len(self._buffer))
        return StreamingParseResult(normal_text=strip_partial_marker_suffix(pending))

    def _emit_normal_text(self, limit: int | None = None) -> str:
        if limit is None:
            holdback = partial_suffix_len(
                self._buffer,
                [self.bot_token, RESPONSE_OPEN, RESPONSE_CLOSE, MESSAGE_CLOSE],
            )
            limit = len(self._buffer) - holdback
        if limit <= self._sent_normal_idx:
            return ""
        pending = self._buffer[self._sent_normal_idx : limit]
        for marker in (RESPONSE_OPEN, RESPONSE_CLOSE, MESSAGE_CLOSE):
            if marker in pending:
                pending = pending.replace(marker, "")
        self._sent_normal_idx = limit
        return pending
