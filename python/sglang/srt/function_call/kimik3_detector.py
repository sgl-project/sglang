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

# Complete open tags, for the streaming path. Unlike _CALL_RE/_ARG_RE these
# match a single tag (not the whole block), so the streaming parser can react
# to each tag as soon as it arrives instead of waiting for the closing tag.
_CALL_OPEN_RE = re.compile(r"<\|open\|>call\s+(?P<attrs>(?:(?!<\|sep\|>).)*?)<\|sep\|>")
_ARG_OPEN_RE = re.compile(
    r"<\|open\|>argument\s+(?P<attrs>(?:(?!<\|sep\|>).)*?)<\|sep\|>"
)


def _json_escape_fragment(raw: str) -> str:
    """JSON-escape a raw string-argument fragment (no surrounding quotes).

    Escaping is per-character, so escaping fragments independently and
    concatenating is byte-identical to escaping the whole string at once.
    """
    return json.dumps(raw, ensure_ascii=False)[1:-1]


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

    The streaming path (``parse_streaming_increment``) emits the tool name as
    soon as a ``<|open|>call ...<|sep|>`` tag completes, then streams
    ``parameters`` as incremental JSON deltas. The concatenation of a call's
    deltas is byte-identical to the non-streaming
    ``json.dumps(arguments, ensure_ascii=False)`` output, so the serving
    layer's final-diff logic keeps working.
    """

    def __init__(self):
        super().__init__()
        self.bot_token = TOOLS_OPEN
        self.eot_token = TOOLS_CLOSE
        self._sent_normal_idx = 0
        # Absolute index into _buffer of the first unprocessed tools-section
        # char; only moves forward as complete streaming events are consumed.
        self._stream_cursor = 0
        # Per-call streaming state, or None when between calls. Keys:
        # "decoded" (dict of finished args, insertion-ordered), "in_arg",
        # "arg_key", "arg_type", "arg_pending" (received-but-unemitted raw
        # string-arg text held back against partial markers), "arg_raw"
        # (full raw text of the in-progress string arg), "saw_arg",
        # "skip" (unnamed call consumed silently).
        self._stream_call: dict | None = None

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
            base = open_idx + len(self.bot_token)
            if self._stream_cursor < base:
                self._stream_cursor = base
            section = self._buffer[base:]
            cursor = self._stream_cursor - base
            calls: List[ToolCallItem] = []
            # Every step either consumes a complete event (advancing the
            # cursor) or returns None to wait for more text; the loop always
            # terminates.
            while True:
                new_cursor = self._stream_section_step(section, cursor, calls)
                if new_cursor is None:
                    break
                cursor = new_cursor
            self._stream_cursor = base + cursor
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
            self._stream_cursor = 0
            self._stream_call = None
            return StreamingParseResult()

    def _emit_arg_delta(self, calls: List[ToolCallItem], delta: str) -> None:
        """Append one incremental ``parameters`` delta for the current call."""
        if not delta:
            return
        tool_id = self.current_tool_id
        self.streamed_args_for_tool[tool_id] += delta
        calls.append(ToolCallItem(tool_index=tool_id, name=None, parameters=delta))

    def _stream_section_step(
        self, section: str, cursor: int, calls: List[ToolCallItem]
    ) -> int | None:
        """Consume one complete streaming event at ``cursor``.

        Returns the new cursor, or None when no complete event is available
        yet and the parser must wait for more text.
        """
        state = self._stream_call
        if state is None:
            return self._stream_call_open(section, cursor, calls)
        if state["skip"]:
            close_idx = section.find(CALL_CLOSE, cursor)
            if close_idx == -1:
                return None
            self._stream_call = None
            return close_idx + len(CALL_CLOSE)
        if state["in_arg"]:
            return self._stream_argument_body(section, cursor, calls, state)
        return self._stream_call_body(section, cursor, calls, state)

    def _stream_call_open(
        self, section: str, cursor: int, calls: List[ToolCallItem]
    ) -> int | None:
        """Handle text between calls: wait for the next call open tag."""
        m = _CALL_OPEN_RE.search(section, cursor)
        if m is None:
            return None
        tool_name = _parse_attrs(m["attrs"]).get("tool", "")
        cursor = m.end()
        if not tool_name:
            logger.warning("Kimi K3 skipping call with empty tool name")
            self._stream_call = {"skip": True}
            return cursor
        self.current_tool_id += 1
        tool_id = self.current_tool_id
        while len(self.prev_tool_call_arr) <= tool_id:
            self.prev_tool_call_arr.append({})
        while len(self.streamed_args_for_tool) <= tool_id:
            self.streamed_args_for_tool.append("")
        # "decoded" is mutated as arguments complete; the serving layer reads
        # it at end-of-stream to flush any unstreamed remainder.
        decoded: dict = {}
        self.prev_tool_call_arr[tool_id] = {"name": tool_name, "arguments": decoded}
        self.streamed_args_for_tool[tool_id] = ""
        self._stream_call = {
            "decoded": decoded,
            "in_arg": False,
            "arg_key": "",
            "arg_type": "string",
            "arg_pending": "",
            "arg_raw": "",
            "saw_arg": False,
            "skip": False,
        }
        calls.append(ToolCallItem(tool_index=tool_id, name=tool_name, parameters=""))
        return cursor

    def _stream_call_body(
        self,
        section: str,
        cursor: int,
        calls: List[ToolCallItem],
        state: dict,
    ) -> int | None:
        """Handle an open call between arguments: argument open or call close."""
        arg_m = _ARG_OPEN_RE.search(section, cursor)
        close_idx = section.find(CALL_CLOSE, cursor)
        # A new call opening before this one closed is malformed; abandon the
        # current call and reprocess the orphan open tag as a new call.
        orphan_m = _CALL_OPEN_RE.search(section, cursor)
        first = min(
            [
                (m.start(), kind)
                for m, kind in (
                    (arg_m, "arg"),
                    (orphan_m, "orphan"),
                )
                if m is not None
            ]
            + ([(close_idx, "close")] if close_idx != -1 else []),
            default=None,
        )
        if first is None:
            return None
        _, kind = first
        if kind == "orphan":
            assert orphan_m is not None
            logger.warning(
                "Kimi K3 call opened before previous call closed; "
                "abandoning incomplete call"
            )
            self._stream_call = None
            return cursor
        if kind == "close":
            decoded = state["decoded"]
            self._emit_arg_delta(
                calls, "{}" if not decoded and not state["saw_arg"] else "}"
            )
            self._stream_call = None
            assert close_idx != -1
            return close_idx + len(CALL_CLOSE)
        assert arg_m is not None
        attrs = _parse_attrs(arg_m["attrs"])
        key = attrs.get("key", "")
        arg_type = attrs.get("type", "string")
        prefix = "{" if not state["decoded"] else ", "
        delta = prefix + json.dumps(key, ensure_ascii=False) + ": "
        if arg_type == "string":
            delta += '"'
        self._emit_arg_delta(calls, delta)
        state.update(
            in_arg=True,
            arg_key=key,
            arg_type=arg_type,
            arg_pending="",
            arg_raw="",
            saw_arg=True,
        )
        return arg_m.end()

    def _stream_argument_body(
        self,
        section: str,
        cursor: int,
        calls: List[ToolCallItem],
        state: dict,
    ) -> int | None:
        """Stream one argument's value until its close tag."""
        key = state["arg_key"]
        if state["arg_type"] == "string":
            # The close marker may start inside the held-back pending text
            # (a chunk boundary can split it), so search the concatenation
            # of pending + newly arrived section text, not the section alone.
            pending = state["arg_pending"]
            tail = section[cursor:]
            combined = pending + tail
            close_idx = combined.find(ARGUMENT_CLOSE)
            if close_idx == -1 and not tail:
                # Everything already consumed; waiting for more text.
                return None
            if close_idx != -1:
                raw = state["arg_raw"] + combined[:close_idx]
                self._emit_arg_delta(calls, _json_escape_fragment(combined[:close_idx]))
                self._emit_arg_delta(calls, '"')
                state["decoded"][key] = raw
                state["in_arg"] = False
                state["arg_pending"] = ""
                state["arg_raw"] = ""
                # The marker overlaps `pending`, so only the part past it
                # advances the section cursor. (The marker always extends
                # past pending: pending is a *proper* marker prefix.)
                return cursor + (close_idx + len(ARGUMENT_CLOSE) - len(pending))
            # Hold back a trailing partial marker so a chunk boundary can
            # never emit text that turns out to be `<|close|>argument<|sep|>`.
            # The held-back tail lives only in arg_pending; the cursor
            # advances past it so it is never processed twice. arg_raw
            # accumulates the full raw value for the decoded-arguments dict.
            hold = partial_suffix_len(combined, [ARGUMENT_CLOSE])
            emit_raw, state["arg_pending"] = (
                combined[: len(combined) - hold],
                combined[len(combined) - hold :],
            )
            state["arg_raw"] += emit_raw
            self._emit_arg_delta(calls, _json_escape_fragment(emit_raw))
            return len(section)
        close_idx = section.find(ARGUMENT_CLOSE, cursor)
        if close_idx == -1:
            # Non-string values are short; buffer them until the close tag.
            return None
        raw = section[cursor:close_idx]
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        state["decoded"][key] = value
        self._emit_arg_delta(calls, json.dumps(value, ensure_ascii=False))
        state["in_arg"] = False
        return close_idx + len(ARGUMENT_CLOSE)

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
