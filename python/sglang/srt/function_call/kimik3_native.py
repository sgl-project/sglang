"""Incremental validation of Kimi K3 native framing, independent of serving.

Complete calls and optional provisional deltas share one validator. Neither commits
the stream:
the caller must still finish validation before emitting a successful terminal.
Errors contain fixed classifications, never model text or tool argument values.
"""

import json
import math
import re
from dataclasses import dataclass, field

from sglang.srt.function_call.kimik3_format import partial_suffix_len

_OPEN, _CLOSE, _SEP = "<|open|>", "<|close|>", "<|sep|>"
_ARGUMENT_CLOSE = _CLOSE + "argument" + _SEP
_TOKENS = [_OPEN, _CLOSE, _SEP]
_CHANNELS = {"think", "response", "tools", "message"}
_HEADS = _CHANNELS | {"call", "argument"}
_ATTR = re.compile(r'\s+([a-z]+)="([^"<>\x00]*)"')


def restore_kimik3_stop(text, finish_reason, tokenizer=None):
    """Restore only a tools-close suffix proven generated and trimmed by stop."""
    if not isinstance(finish_reason, dict) or finish_reason.get("type") != "stop":
        return text
    tools_close = _CLOSE + "tools" + _SEP
    matched = finish_reason.get("matched")
    if matched == tools_close:
        return text if text.endswith(tools_close) else text + tools_close
    if type(matched) is int and tokenizer is not None:
        suffix = tokenizer.decode([matched], skip_special_tokens=False)
        if (
            isinstance(suffix, str)
            and suffix
            and tools_close.endswith(suffix)
            and text.endswith(tools_close[: -len(suffix)])
        ):
            return text + suffix
    return text


class KimiK3ParseError(ValueError):
    """A native framing failure which must not fall back to successful text."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Invalid Kimi K3 native output ({reason})")


def _header(pending: str):
    """Read exactly one header, or hold a partial header across chunks."""
    token = _OPEN if pending.startswith(_OPEN) else _CLOSE
    rest = pending[len(token) :]
    end = rest.find(_SEP)
    before_sep = rest if end < 0 else rest[:end]
    if end < 0:
        hold = partial_suffix_len(rest, [_SEP])
        if hold:
            before_sep = rest[:-hold]
    if _OPEN in before_sep or _CLOSE in before_sep:
        raise KimiK3ParseError("invalid_header")
    # Reject a reserved name followed by corrupt data as soon as it diverges.
    # Waiting for SEP here would buffer (and potentially expose) the bad head.
    if not any(head.startswith(before_sep) for head in _HEADS):
        match = re.match(r"([a-z]+)(.*)", before_sep, re.DOTALL)
        if not match or match[1] not in _HEADS:
            raise KimiK3ParseError("invalid_header")
        name, attrs = match.groups()
        if name in _CHANNELS or token == _CLOSE:
            if attrs:
                raise KimiK3ParseError("invalid_header")
        elif attrs and not attrs[0].isspace():
            raise KimiK3ParseError("invalid_header")
    if end < 0:
        if len(pending) > 1024:
            raise KimiK3ParseError("invalid_header")
        return None
    if len(token) + end + len(_SEP) > 1024:
        raise KimiK3ParseError("invalid_header")
    match = re.fullmatch(r"([a-z]+)(.*)", before_sep, re.DOTALL)
    if not match or match[1] not in _HEADS:
        raise KimiK3ParseError("invalid_header")
    name, attrs = match.groups()
    if (name in _CHANNELS or token == _CLOSE) and attrs:
        raise KimiK3ParseError("invalid_header")
    return token, name, attrs, len(token) + end + len(_SEP)


def _attributes(text: str, expected: set[str]) -> dict[str, str]:
    attrs = {}
    offset = 0
    while offset < len(text.rstrip()):
        match = _ATTR.match(text, offset)
        if not match or match[1] in attrs:
            raise KimiK3ParseError("invalid_attributes")
        attrs[match[1]] = match[2].replace("&quot;", '"').replace("&amp;", "&")
        offset = match.end()
    if set(attrs) != expected or any(not value for value in attrs.values()):
        raise KimiK3ParseError("invalid_attributes")
    return attrs


class _JsonQuotes:
    """Track JSON string literals without interpreting native markers in them."""

    def __init__(self):
        self.quoted = False
        self.escaped = False

    def feed(self, text):
        for char in text:
            if self.escaped:
                self.escaped = False
            elif self.quoted and char == "\\":
                self.escaped = True
            elif char == '"':
                self.quoted = not self.quoted


class KimiK3HeaderValidator:
    """Validate raw headers before reasoning parsing can consume the evidence.

    Tool-section ordering is checked by the tool parser on its own input. This
    guard checks header syntax plus the response/message boundaries that
    reasoning cleanup erases before that parser runs. Raw generation may still
    start inside a prefilled think channel, transition straight to tools or a
    response without an explicit think close, or end a skipped think channel
    with a bare response close.
    """

    def __init__(self):
        self.pending = ""
        self.failure = None
        self.json_argument = False
        self.string_argument = False
        self.quotes = _JsonQuotes()
        self.response_open = False
        self.message_closed = False

    def feed(self, text: str) -> None:
        if self.failure:
            raise self.failure
        self.pending += text
        try:
            while self.pending:
                if self.string_argument:
                    end = self.pending.find(_ARGUMENT_CLOSE)
                    if end < 0:
                        hold = partial_suffix_len(self.pending, [_ARGUMENT_CLOSE])
                        self.pending = self.pending[-hold:] if hold else ""
                        break
                    self.pending = self.pending[end:]
                indexes = [self.pending.find(t) for t in _TOKENS]
                indexes = [i for i in indexes if i >= 0]
                if not indexes:
                    hold = partial_suffix_len(self.pending, _TOKENS)
                    if self.json_argument:
                        self.quotes.feed(self.pending[:-hold] if hold else self.pending)
                    self.pending = self.pending[-hold:] if hold else ""
                    break
                if self.json_argument:
                    self.quotes.feed(self.pending[: min(indexes)])
                self.pending = self.pending[min(indexes) :]
                if self.json_argument and self.quotes.quoted:
                    token = next(t for t in _TOKENS if self.pending.startswith(t))
                    self.pending = self.pending[len(token) :]
                    continue
                if self.pending.startswith(_SEP):
                    raise KimiK3ParseError("invalid_boundary")
                header = _header(self.pending)
                if header is None:
                    break
                token, name, attrs, _ = header
                self._channel(token, name)
                if token == _OPEN and name == "argument":
                    self.json_argument = (
                        _attributes(attrs, {"key", "type"})["type"] != "string"
                    )
                    self.string_argument = not self.json_argument
                    self.quotes = _JsonQuotes()
                elif token == _CLOSE and name == "argument":
                    self.json_argument = False
                    self.string_argument = False
                self.pending = self.pending[header[3] :]
        except KimiK3ParseError as error:
            self.failure = error
            raise

    def _channel(self, token: str, name: str) -> None:
        # Reasoning cleanup strips response wrappers and message closes before
        # the tool parser sees its input, so these boundaries are enforced here.
        if self.message_closed:
            raise KimiK3ParseError("invalid_boundary")
        if name == "response":
            if token == _OPEN and self.response_open:
                raise KimiK3ParseError("invalid_boundary")
            # A bare close is valid: the prefilled think channel can be skipped.
            self.response_open = token == _OPEN
        elif self.response_open:
            raise KimiK3ParseError("invalid_boundary")
        elif token == _CLOSE and name == "message":
            self.message_closed = True

    def finish(self, *, allow_incomplete: bool = False) -> None:
        if self.failure:
            raise self.failure
        if not allow_incomplete and (
            self.response_open or self.pending.startswith((_OPEN, _CLOSE))
        ):
            self.failure = KimiK3ParseError("unclosed_channel")
            raise self.failure


@dataclass
class NativeCall:
    name: str
    arguments: dict


@dataclass
class NativeDelta:
    name: str | None = None
    parameters: str = ""


@dataclass
class NativeResult:
    normal_text: str = ""
    calls: list[NativeCall] = field(default_factory=list)
    deltas: list[NativeDelta] = field(default_factory=list)


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise KimiK3ParseError("duplicate_argument")
        result[key] = value
    return result


def _invalid_constant(_value):
    raise KimiK3ParseError("invalid_json")


def _value(kind: str, text: str):
    if kind == "string":
        return text
    try:
        value = json.loads(
            text, object_pairs_hook=_json_object, parse_constant=_invalid_constant
        )
    except (ValueError, OverflowError, RecursionError) as error:
        if isinstance(error, KimiK3ParseError):
            raise
        raise KimiK3ParseError("invalid_json") from None
    valid = {
        "integer": type(value) is int,
        "number": type(value) is int or (type(value) is float and math.isfinite(value)),
        "boolean": type(value) is bool,
        "object": type(value) is dict,
        "array": type(value) is list,
        "null": value is None,
    }
    if not valid.get(kind, False):
        raise KimiK3ParseError("invalid_argument_type")
    # Also reject non-finite numbers nested inside a typed object or array.
    try:
        json.dumps(value, allow_nan=False)
    except (ValueError, OverflowError, RecursionError):
        raise KimiK3ParseError("invalid_json") from None
    return value


class KimiK3NativeParser:
    """Consume every native span exactly once, with stream/nonstream parity."""

    def __init__(self, *, stream_arguments=False):
        self.stream_arguments = stream_arguments
        self.allowed_tools = None
        self.argument_bytes = 0
        self.pending = ""
        self.state = "text"
        self.response_open = False
        self.think_open = False
        self.message_closed = False
        self.section_calls = 0
        self.name = ""
        self.arguments = {}
        self.argument_key = ""
        self.argument_type = ""
        self.value_parts = []
        self.quotes = _JsonQuotes()
        self.failure = None

    def feed(self, text: str) -> NativeResult:
        if self.failure:
            raise self.failure
        self.pending += text
        result = NativeResult()
        try:
            while self.pending:
                if self.state == "argument" and self.argument_type == "string":
                    end = self.pending.find(_ARGUMENT_CLOSE)
                    if end < 0:
                        hold = partial_suffix_len(self.pending, [_ARGUMENT_CLOSE])
                        self._text(
                            self.pending[:-hold] if hold else self.pending, result
                        )
                        self.pending = self.pending[-hold:] if hold else ""
                        break
                    self._text(self.pending[:end], result)
                    self.pending = self.pending[end:]
                indexes = [self.pending.find(t) for t in _TOKENS]
                indexes = [i for i in indexes if i >= 0]
                if not indexes:
                    hold = partial_suffix_len(self.pending, _TOKENS)
                    emit = self.pending[:-hold] if hold else self.pending
                    self.pending = self.pending[-hold:] if hold else ""
                    self._text(emit, result)
                    break
                index = min(indexes)
                self._text(self.pending[:index], result)
                self.pending = self.pending[index:]
                if (
                    self.state == "argument"
                    and self.argument_type != "string"
                    and self.quotes.quoted
                ):
                    token = next(t for t in _TOKENS if self.pending.startswith(t))
                    self._text(token, result)
                    self.pending = self.pending[len(token) :]
                    continue
                if self.pending.startswith(_SEP):
                    raise KimiK3ParseError("invalid_boundary")
                header = _header(self.pending)
                if header is None:
                    break
                token, name, attrs, end = header
                self._control(token, name, attrs, result)
                self.pending = self.pending[end:]
        except KimiK3ParseError as error:
            self.failure = error
            raise
        return result

    def _text(self, text: str, result: NativeResult) -> None:
        if self.state == "argument":
            self.argument_bytes += len(text.encode("utf-8"))
            if self.argument_bytes > 4 * 1024 * 1024:
                raise KimiK3ParseError("argument_too_large")
            self.value_parts.append(text)
            if self.stream_arguments and self.argument_type == "string" and text:
                result.deltas.append(
                    NativeDelta(parameters=json.dumps(text, ensure_ascii=False)[1:-1])
                )
            if self.argument_type != "string":
                self.quotes.feed(text)
        elif self.state == "text" and not self.message_closed:
            result.normal_text += text
        elif text.strip():
            raise KimiK3ParseError("invalid_boundary")

    def _control(self, token, name, attrs, result):
        opening = token == _OPEN
        if self.state == "argument":
            if opening or name != "argument":
                raise KimiK3ParseError("invalid_boundary")
            self.arguments[self.argument_key] = _value(
                self.argument_type, "".join(self.value_parts)
            )
            if self.stream_arguments:
                value = self.arguments[self.argument_key]
                fragment = (
                    '"'
                    if self.argument_type == "string"
                    else json.dumps(value, ensure_ascii=False, allow_nan=False)
                )
                result.deltas.append(NativeDelta(parameters=fragment))
            self.value_parts = []
            self.state = "call"
        elif self.state == "call":
            if opening and name == "argument":
                attributes = _attributes(attrs, {"key", "type"})
                key, kind = attributes["key"], attributes["type"]
                if key in self.arguments:
                    raise KimiK3ParseError("duplicate_argument")
                if kind not in {
                    "string",
                    "integer",
                    "number",
                    "boolean",
                    "object",
                    "array",
                    "null",
                }:
                    raise KimiK3ParseError("invalid_argument_type")
                if self.stream_arguments:
                    fragment = (
                        (", " if self.arguments else "")
                        + json.dumps(key, ensure_ascii=False)
                        + ": "
                    )
                    result.deltas.append(
                        NativeDelta(
                            parameters=fragment + ('"' if kind == "string" else "")
                        )
                    )
                self.argument_bytes = 0
                self.argument_key, self.argument_type = key, kind
                self.quotes = _JsonQuotes()
                self.state = "argument"
            elif not opening and name == "call":
                if self.stream_arguments:
                    result.deltas.append(NativeDelta(parameters="}"))
                result.calls.append(NativeCall(self.name, self.arguments))
                self.arguments = {}
                self.section_calls += 1
                self.state = "tools"
            else:
                raise KimiK3ParseError("invalid_boundary")
        elif self.state == "tools":
            if opening and name == "call":
                attributes = _attributes(attrs, {"tool", "index"})
                if not re.fullmatch(r"[1-9][0-9]*", attributes["index"]):
                    raise KimiK3ParseError("invalid_attributes")
                self.name = attributes["tool"]
                if (
                    self.allowed_tools is not None
                    and self.name not in self.allowed_tools
                ):
                    raise KimiK3ParseError("unknown_tool")
                if self.stream_arguments:
                    result.deltas.append(NativeDelta(name=self.name, parameters="{"))
                self.arguments = {}
                self.state = "call"
            elif not opening and name == "tools":
                if not self.section_calls:
                    raise KimiK3ParseError("empty_tools")
                self.state = "text"
            else:
                raise KimiK3ParseError("invalid_boundary")
        elif self.message_closed:
            raise KimiK3ParseError("invalid_boundary")
        elif name == "tools" and opening and not self.response_open:
            self.think_open = False  # K3 can jump directly from think to tools.
            self.section_calls = 0
            self.state = "tools"
        elif name == "response":
            if opening and self.response_open:
                raise KimiK3ParseError("invalid_boundary")
            # The reasoning parser can remove the opening wrapper, so a
            # residual complete response close is intentionally accepted.
            self.response_open = opening
            self.think_open = False
        elif (
            name == "think"
            and opening
            and not self.response_open
            and not self.think_open
        ):
            # separate_reasoning=False sends the full native turn to this
            # parser. Preserve the text while consuming complete wrappers.
            self.think_open = True
        elif name == "think" and not opening and not self.response_open:
            self.think_open = False
        elif name == "message" and not opening and not self.response_open:
            self.message_closed = True
        else:
            raise KimiK3ParseError("invalid_boundary")

    def finish(self, *, allow_incomplete: bool = False) -> NativeResult:
        if self.failure:
            raise self.failure
        if not allow_incomplete:
            if self.state in {"call", "argument"}:
                self.failure = KimiK3ParseError("unclosed_call")
            elif (
                self.state == "tools"
                or self.response_open
                or self.think_open
                or self.pending.startswith((_OPEN, _CLOSE))
            ):
                self.failure = KimiK3ParseError("unclosed_channel")
            if self.failure:
                raise self.failure
        result = NativeResult()
        # A literal unfinished control-token prefix is ordinary text. A
        # complete token/header cut by a known length limit is never exposed.
        if self.state == "text" and not self.pending.startswith((_OPEN, _CLOSE)):
            self._text(self.pending, result)
        self.pending = ""
        self.value_parts = []
        return result
