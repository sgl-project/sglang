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

_ARG_OPEN = "<|open|>argument"
_SEP = "<|sep|>"

# Incremental scanner states for the tools section.
_ST_SEEK_CALL = "seek_call"  # between calls, waiting for <|open|>call
_ST_CALL_ATTRS = "call_attrs"  # inside a call header, waiting for its <|sep|>
_ST_SEEK_ARG = "seek_arg"  # in a call body, awaiting an argument or the call close
_ST_ARG_ATTRS = "arg_attrs"  # inside an argument header, waiting for its <|sep|>
_ST_ARG_VALUE = "arg_value"  # inside an argument value, streaming it

# Markers whose trailing partial form must never be flushed as payload text.
_HOLDBACK_MARKERS = [
    ARGUMENT_CLOSE,
    CALL_CLOSE,
    _ARG_OPEN,
    CALL_OPEN,
    "<|open|>",
    "<|close|>",
]


def _json_escape(text: str) -> str:
    """Escape ``text`` for use inside a JSON string literal.

    ``json.dumps`` escapes per code point and never re-interprets a fragment
    boundary, so escaping fragments independently and concatenating the results
    is identical to escaping their concatenation.
    """
    return json.dumps(text, ensure_ascii=False)[1:-1]


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
        self._reset_call_stream()

    def _reset_call_stream(self):
        """Reset the incremental tools-section scanner to a clean state."""
        self._k3_pos = 0
        self._k3_state = _ST_SEEK_CALL
        self._k3_call_name = ""
        self._k3_args: dict = {}
        self._k3_arg_key: Optional[str] = None
        self._k3_arg_type = ""
        self._k3_val_start = 0
        self._k3_val_streamed = 0

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

    def _start_call(self, name: str) -> List[ToolCallItem]:
        """Open a new tool call: register bookkeeping and emit the name delta."""
        self.current_tool_id += 1
        while len(self.prev_tool_call_arr) <= self.current_tool_id:
            self.prev_tool_call_arr.append({})
        while len(self.streamed_args_for_tool) <= self.current_tool_id:
            self.streamed_args_for_tool.append("")
        self._k3_call_name = name
        self._k3_args = {}
        self.prev_tool_call_arr[self.current_tool_id] = {"name": name, "arguments": {}}
        self.current_tool_name_sent = True
        # The name delta follows the base-detector contract: name first, with
        # empty parameters, then the argument deltas stream behind it.
        return [
            ToolCallItem(tool_index=self.current_tool_id, name=name, parameters=""),
            self._emit_arg_delta("{"),
        ]

    def _emit_arg_delta(self, delta: str) -> ToolCallItem:
        """Record an argument fragment and wrap it as a streaming delta."""
        if delta:
            self.streamed_args_for_tool[self.current_tool_id] += delta
        return ToolCallItem(tool_index=self.current_tool_id, parameters=delta)

    def _stream_calls(self, section_start: int) -> List[ToolCallItem]:
        """Incrementally emit tool-call deltas for the in-progress tools section.

        Every ``parameters`` delta recorded here keeps ``streamed_args_for_tool``
        a byte-exact prefix of ``json.dumps(arguments, ensure_ascii=False)``,
        which is what the serving layer's final-diff logic assumes.
        """
        calls: List[ToolCallItem] = []
        buf = self._buffer
        i = max(self._k3_pos, section_start)

        while True:
            state = self._k3_state

            if state == _ST_ARG_VALUE:
                close_idx = buf.find(ARGUMENT_CLOSE, self._k3_val_start)
                if close_idx == -1:
                    # Hold back a trailing partial marker so a split
                    # "<|close|>argument<|sep|>" is never leaked as payload.
                    hold = partial_suffix_len(
                        buf[self._k3_val_start :], _HOLDBACK_MARKERS
                    )
                    raw_end = len(buf) - hold
                else:
                    raw_end = close_idx
                if raw_end < self._k3_val_start:
                    raw_end = self._k3_val_start
                # Only raw string payloads can be streamed character by
                # character. Everything else is JSON-decoded in one go once its
                # closing tag arrives, so buffering it here would emit the raw
                # text *and* the decoded value.
                if (
                    self._k3_arg_type == "string"
                    and raw_end > self._k3_val_start + self._k3_val_streamed
                ):
                    delta = buf[self._k3_val_start + self._k3_val_streamed : raw_end]
                    calls.append(self._emit_arg_delta(_json_escape(delta)))
                    self._k3_val_streamed = raw_end - self._k3_val_start
                if close_idx == -1:
                    self._k3_pos = max(i, raw_end)
                    break
                # The value is complete: flush its tail and the closing quote.
                raw = buf[self._k3_val_start : close_idx]
                if self._k3_arg_type == "string":
                    remainder = raw[self._k3_val_streamed :]
                    if remainder:
                        calls.append(self._emit_arg_delta(_json_escape(remainder)))
                    calls.append(self._emit_arg_delta('"'))
                    self._k3_args[self._k3_arg_key] = raw
                else:
                    try:
                        self._k3_args[self._k3_arg_key] = json.loads(raw)
                    except json.JSONDecodeError:
                        self._k3_args[self._k3_arg_key] = raw
                    calls.append(
                        self._emit_arg_delta(
                            json.dumps(
                                self._k3_args[self._k3_arg_key], ensure_ascii=False
                            )
                        )
                    )
                self.prev_tool_call_arr[self.current_tool_id]["arguments"] = (
                    self._k3_args
                )
                self._k3_arg_key = None
                self._k3_state = _ST_SEEK_ARG
                i = close_idx + len(ARGUMENT_CLOSE)
                self._k3_pos = i
                continue

            if state == _ST_ARG_ATTRS:
                sep_idx = buf.find(_SEP, i)
                if sep_idx == -1:
                    self._k3_pos = i
                    break
                arg_attrs = _parse_attrs(buf[i:sep_idx])
                key = arg_attrs.get("key", "")
                arg_type = arg_attrs.get("type", "string")
                # ``{`` was emitted with the call name, so only the key needs a
                # leading comma when it is not the first argument. The key must be
                # quoted exactly like ``json.dumps`` would quote it.
                prefix = (", " if self._k3_args else "") + json.dumps(
                    key, ensure_ascii=False
                )
                if arg_type == "string":
                    calls.append(self._emit_arg_delta(prefix + ': "'))
                else:
                    calls.append(self._emit_arg_delta(prefix + ": "))
                self._k3_arg_key = key
                self._k3_arg_type = arg_type
                self._k3_val_start = sep_idx + len(_SEP)
                self._k3_val_streamed = 0
                self._k3_state = _ST_ARG_VALUE
                i = self._k3_val_start
                self._k3_pos = i
                continue

            if state == _ST_CALL_ATTRS:
                sep_idx = buf.find(_SEP, i)
                if sep_idx == -1:
                    self._k3_pos = i
                    break
                name = _parse_attrs(buf[i:sep_idx]).get("tool", "")
                if name:
                    calls.extend(self._start_call(name))
                self._k3_state = _ST_SEEK_ARG
                i = sep_idx + len(_SEP)
                self._k3_pos = i
                continue

            if state == _ST_SEEK_ARG:
                close_idx = buf.find(CALL_CLOSE, i)
                arg_idx = buf.find(_ARG_OPEN, i)
                if close_idx != -1 and (arg_idx == -1 or close_idx < arg_idx):
                    calls.append(self._emit_arg_delta("}"))
                    self.current_tool_name_sent = False
                    self._k3_state = _ST_SEEK_CALL
                    i = close_idx + len(CALL_CLOSE)
                    self._k3_pos = i
                    continue
                if arg_idx == -1:
                    self._k3_pos = i
                    break
                self._k3_state = _ST_ARG_ATTRS
                i = arg_idx + len(_ARG_OPEN)
                self._k3_pos = i
                continue

            # _ST_SEEK_CALL
            open_idx = buf.find(CALL_OPEN, i)
            if open_idx == -1:
                hold = partial_suffix_len(buf[i:], _HOLDBACK_MARKERS)
                self._k3_pos = max(i, len(buf) - hold)
                break
            self._k3_state = _ST_CALL_ATTRS
            i = open_idx + len(CALL_OPEN)
            self._k3_pos = i

        return calls

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._buffer += new_text
        try:
            open_idx = self._buffer.find(self.bot_token)
            if open_idx == -1:
                return StreamingParseResult(normal_text=self._emit_normal_text())

            normal_text = self._emit_normal_text(limit=open_idx)
            calls = self._stream_calls(open_idx + len(self.bot_token))
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
            self._reset_call_stream()
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
