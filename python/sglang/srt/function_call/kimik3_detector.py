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
    ARGUMENT_CLOSE,
    CALL_CLOSE,
    CALL_OPEN,
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


_SEP = "<|sep|>"
_ARGUMENT_OPEN = "<|open|>argument"

# Streaming states, mirroring the nesting _CALL_RE / _ARG_RE match.
_NORMAL = "normal"  # before the tools channel
_TOOLS = "tools"  # inside the tools channel, between calls
_CALL = "call"  # inside a call, between arguments
_VALUE = "value"  # inside an argument value
_SKIP_CALL = "skip_call"  # inside a call with no tool name
_DONE = "done"  # tools channel closed; the rest is ignored

_NO_MATCH = object()


def _json_str_body(text: str) -> str:
    # json.dumps escapes per character, so escaped fragments concatenate to the
    # escaped whole.
    return json.dumps(text, ensure_ascii=False)[1:-1]


def _first_marker(text: str, pos: int, markers: List[str]) -> tuple[int, str | None]:
    best_idx, best_marker = -1, None
    for marker in markers:
        idx = text.find(marker, pos)
        if idx != -1 and (best_idx == -1 or idx < best_idx):
            best_idx, best_marker = idx, marker
    return best_idx, best_marker


def _hold(buf: str, pos: int, markers: List[str]) -> int:
    """End of the text that cannot be the start of one of ``markers``."""
    return max(pos, len(buf) - partial_suffix_len(buf, markers))


def _read_open_tag(buf: str, idx: int, prefix: str, fences: List[str]):
    """Parse ``prefix\\s+attrs<|sep|>`` at ``idx`` like _CALL_RE / _ARG_RE do.

    Returns (attrs, end) on a match, None if more text is needed, or _NO_MATCH
    if the regex would not match here (no whitespace, or a fence comes first).
    """
    start = idx + len(prefix)
    if start >= len(buf):
        return None
    if not buf[start].isspace():
        return _NO_MATCH
    sep = buf.find(_SEP, start)
    if sep == -1:
        return None
    for fence in fences:
        fence_idx = buf.find(fence, start)
        if fence_idx != -1 and fence_idx < sep:
            return _NO_MATCH
    return buf[start:sep], sep + len(_SEP)


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
        # Streaming state; once the tools channel opens, _buffer only holds
        # text not yet consumed.
        self._state = _NORMAL
        self._call_name = ""
        self._call_args: dict = {}
        self._call_arg_count = 0
        self._arg_key = ""
        self._arg_type = "string"
        self._arg_raw = ""

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
        """Stream tool calls as they are generated.

        A call's name is emitted as soon as its open tag is complete, and its
        arguments stream as JSON fragments whose concatenation is byte-identical
        to the ``parameters`` produced by ``detect_and_parse``. String values
        stream as they arrive; non-string values are emitted when they close.
        """
        self._buffer += new_text
        try:
            normal_text = ""
            if self._state == _NORMAL:
                open_idx = self._buffer.find(self.bot_token)
                if open_idx == -1:
                    return StreamingParseResult(normal_text=self._emit_normal_text())
                normal_text = self._emit_normal_text(limit=open_idx)
                self._buffer = self._buffer[open_idx + len(self.bot_token) :]
                self._state = _TOOLS
            return StreamingParseResult(normal_text=normal_text, calls=self._advance())
        except Exception as e:
            logger.error(
                "Error in Kimi K3 parse_streaming_increment: %s", e, exc_info=True
            )
            # _sent_normal_idx indexes into _buffer, so it must be reset with it;
            # otherwise every later _emit_normal_text sees limit <= _sent_normal_idx
            # and silently drops the rest of the response.
            self._buffer = ""
            self._sent_normal_idx = 0
            if self._state != _NORMAL:
                # Never fall back to normal text inside the tools channel, which
                # would ship XTML markup to the client.
                self._state = _DONE
            return StreamingParseResult()

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        if self._state == _NORMAL:
            pending = self._emit_normal_text(limit=len(self._buffer))
            return StreamingParseResult(
                normal_text=strip_partial_marker_suffix(pending)
            )
        calls: List[ToolCallItem] = []
        if self._state in (_CALL, _VALUE):
            logger.warning(
                "Kimi K3 stream ended inside tool call %d; closing its arguments",
                self.current_tool_id,
            )
            self._abort_call(out=calls)
        elif self.current_tool_id == -1:
            logger.warning(
                "Kimi K3 tools section ended with no complete tool call; "
                "dropping %d buffered chars",
                len(self._buffer),
            )
        self._state = _DONE
        self._buffer = ""
        return StreamingParseResult(calls=calls)

    # ===== Streaming state machine (tools channel) =====

    def _advance(self) -> List[ToolCallItem]:
        out: List[ToolCallItem] = []
        buf, pos, progressed = self._buffer, 0, True
        while progressed and self._state != _DONE:
            pos, progressed = self._step(buf=buf, pos=pos, out=out)
        self._buffer = "" if self._state == _DONE else buf[pos:]
        return out

    def _step(self, buf: str, pos: int, out: List[ToolCallItem]) -> tuple[int, bool]:
        """Consume one marker or a run of text; return (new_pos, progressed)."""
        if self._state == _TOOLS:
            return self._step_tools(buf=buf, pos=pos, out=out)
        if self._state == _CALL:
            return self._step_call(buf=buf, pos=pos, out=out)
        if self._state == _VALUE:
            return self._step_value(buf=buf, pos=pos, out=out)
        return self._step_skip_call(buf=buf, pos=pos)

    def _step_tools(
        self, buf: str, pos: int, out: List[ToolCallItem]
    ) -> tuple[int, bool]:
        markers = [CALL_OPEN, TOOLS_CLOSE]
        idx, marker = _first_marker(buf, pos, markers)
        if marker is None:
            return _hold(buf=buf, pos=pos, markers=markers), False
        if marker == TOOLS_CLOSE:
            self._state = _DONE
            return len(buf), False
        tag = _read_open_tag(buf=buf, idx=idx, prefix=CALL_OPEN, fences=[TOOLS_CLOSE])
        if tag is None:
            return idx, False
        if tag is _NO_MATCH:
            return idx + 1, True
        attrs, end = tag
        name = _parse_attrs(attrs).get("tool", "")
        if name:
            self._start_call(name=name, out=out)
            self._state = _CALL
        else:
            # detect_and_parse drops nameless calls without consuming an index.
            self._state = _SKIP_CALL
        return end, True

    def _step_call(
        self, buf: str, pos: int, out: List[ToolCallItem]
    ) -> tuple[int, bool]:
        markers = [_ARGUMENT_OPEN, CALL_CLOSE, TOOLS_CLOSE]
        idx, marker = _first_marker(buf, pos, markers)
        if marker is None:
            return _hold(buf=buf, pos=pos, markers=markers), False
        if marker == TOOLS_CLOSE:
            logger.warning(
                "Kimi K3 tools channel closed inside tool call %d; "
                "closing its arguments",
                self.current_tool_id,
            )
            self._abort_call(out=out)
            self._state = _DONE
            return len(buf), False
        if marker == CALL_CLOSE:
            self._finish_call(out=out)
            self._state = _TOOLS
            return idx + len(CALL_CLOSE), True
        tag = _read_open_tag(
            buf=buf, idx=idx, prefix=_ARGUMENT_OPEN, fences=[CALL_CLOSE, TOOLS_CLOSE]
        )
        if tag is None:
            return idx, False
        if tag is _NO_MATCH:
            return idx + 1, True
        attrs, end = tag
        self._start_argument(attrs=attrs, out=out)
        self._state = _VALUE
        return end, True

    def _step_value(
        self, buf: str, pos: int, out: List[ToolCallItem]
    ) -> tuple[int, bool]:
        markers = [ARGUMENT_CLOSE, CALL_CLOSE, TOOLS_CLOSE]
        idx, marker = _first_marker(buf, pos, markers)
        if marker is None:
            safe = _hold(buf=buf, pos=pos, markers=markers)
            self._append_value(text=buf[pos:safe], out=out)
            return safe, False
        self._append_value(text=buf[pos:idx], out=out)
        self._state = _CALL
        if marker == ARGUMENT_CLOSE:
            self._end_argument(out=out)
            return idx + len(ARGUMENT_CLOSE), True
        self._drop_argument(out=out)
        return idx, True

    def _step_skip_call(self, buf: str, pos: int) -> tuple[int, bool]:
        markers = [CALL_CLOSE, TOOLS_CLOSE]
        idx, marker = _first_marker(buf, pos, markers)
        if marker is None:
            return _hold(buf=buf, pos=pos, markers=markers), False
        if marker == TOOLS_CLOSE:
            self._state = _DONE
            return len(buf), False
        self._state = _TOOLS
        return idx + len(CALL_CLOSE), True

    def _start_call(self, name: str, out: List[ToolCallItem]) -> None:
        self.current_tool_id += 1
        while len(self.prev_tool_call_arr) <= self.current_tool_id:
            self.prev_tool_call_arr.append({})
        while len(self.streamed_args_for_tool) <= self.current_tool_id:
            self.streamed_args_for_tool.append("")
        self._call_name = name
        self._call_args = {}
        self._call_arg_count = 0
        out.append(
            ToolCallItem(tool_index=self.current_tool_id, name=name, parameters="")
        )

    def _finish_call(self, out: List[ToolCallItem]) -> None:
        self._emit(out=out, fragment="}" if self._call_arg_count else "{}")
        # Recorded only for closed calls, so the serving layer never "completes"
        # the JSON of a call that was cut off mid-stream.
        self.prev_tool_call_arr[self.current_tool_id] = {
            "name": self._call_name,
            "arguments": self._call_args,
        }

    def _abort_call(self, out: List[ToolCallItem]) -> None:
        """Close the JSON of a call cut off by end of stream or by the channel.

        detect_and_parse drops such a call, but its name has already been
        sent, so the client gets what arrived rather than a dangling prefix.
        """
        if self._state == _VALUE:
            self._drop_argument(out=out)
        self._finish_call(out=out)

    def _start_argument(self, attrs: str, out: List[ToolCallItem]) -> None:
        arg_attrs = _parse_attrs(attrs)
        self._arg_key = arg_attrs.get("key", "")
        self._arg_type = arg_attrs.get("type", "string")
        self._arg_raw = ""
        if self._arg_type == "string":
            self._emit(out=out, fragment=self._next_arg_prefix() + '"')

    def _append_value(self, text: str, out: List[ToolCallItem]) -> None:
        if not text:
            return
        self._arg_raw += text
        if self._arg_type == "string":
            self._emit(out=out, fragment=_json_str_body(text))

    def _end_argument(self, out: List[ToolCallItem]) -> None:
        if self._arg_type == "string":
            value = self._arg_raw
            self._emit(out=out, fragment='"')
        else:
            try:
                value = json.loads(self._arg_raw)
            except json.JSONDecodeError:
                value = self._arg_raw
            self._emit(
                out=out,
                fragment=self._next_arg_prefix()
                + json.dumps(value, ensure_ascii=False),
            )
        self._call_args[self._arg_key] = value

    def _drop_argument(self, out: List[ToolCallItem]) -> None:
        logger.warning(
            "Kimi K3 argument %r of tool call %d has no closing tag",
            self._arg_key,
            self.current_tool_id,
        )
        if self._arg_type == "string":
            # Already streamed; close it so the arguments stay valid JSON.
            self._emit(out=out, fragment='"')
            self._call_args[self._arg_key] = self._arg_raw

    def _next_arg_prefix(self) -> str:
        sep = ", " if self._call_arg_count else "{"
        self._call_arg_count += 1
        return sep + json.dumps(self._arg_key, ensure_ascii=False) + ": "

    def _emit(self, out: List[ToolCallItem], fragment: str) -> None:
        tool_id = self.current_tool_id
        self.streamed_args_for_tool[tool_id] += fragment
        last = out[-1] if out else None
        if last is not None and last.tool_index == tool_id and last.name is None:
            last.parameters += fragment
        else:
            out.append(ToolCallItem(tool_index=tool_id, name=None, parameters=fragment))

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
