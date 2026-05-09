import ast
import json
import re
from enum import Enum, auto
from typing import Any, List, Optional

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)


class _ParseState(Enum):
    """5 FSM states for the streaming parser.

    Entry guard: READING_VALUE is reachable only from READING_KEY, so the
    "stray <arg_value> before <tool_call>" bug class is structurally
    impossible.

    Exit guard: both READING_KEY and READING_VALUE recover on `</tool_call>`
    by closing the active call (orphan key dropped if any). READING_VALUE
    additionally recovers on `<arg_key>` by replacing the orphan pending key
    with the new one. Both guards match the (regex-tightened) non-streaming
    path. Without them, malformed inputs would leave the FSM stuck in
    READING_VALUE and mis-attribute subsequent values to stale state.
    """

    OUTSIDE = auto()
    READING_NAME = auto()
    READING_KEY = auto()
    READING_VALUE = auto()
    DRAINING = auto()


class PoolsideV1Detector(BaseFormatDetector):
    """
    Detector for poolside Laguna-XS.2 (poolside_v1 series) tool-call wire format.

    Wire format:
        <tool_call>{name}\\n
        <arg_key>{key}</arg_key>\\n
        <arg_value>{val}</arg_value>\\n
        ...
        </tool_call>

    String values are emitted as raw text; non-strings are JSON-encoded by
    the chat template. The parser does schema-based type coercion to round-trip
    them: schema type `string` keeps the raw value; other types attempt
    `json.loads` and fall back to `ast.literal_eval`, then to the raw string.
    """

    # Wire-format tag tokens — constants, not per-instance.
    tool_call_start_token = "<tool_call>"
    tool_call_end_token = "</tool_call>"
    arg_key_start = "<arg_key>"
    arg_key_end = "</arg_key>"
    arg_value_start = "<arg_value>"
    arg_value_end = "</arg_value>"

    tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    # Key uses [^<]*? to prevent the non-greedy `.*?` from backtracking
    # across an `</arg_key>` boundary on malformed inputs like
    # `<arg_key>K1</arg_key><arg_key>K2</arg_key><arg_value>V</arg_value>`
    # — without the `[^<]` constraint, the regex matches the entire orphan
    # span as a single key (`K1</arg_key><arg_key>K2`). Param names never
    # contain `<` in practice, so this is safe. The value side keeps `.*?`
    # because legitimate values can contain `<` (HTML, paths, etc.); the
    # `</arg_value>` boundary is anchored enough.
    arg_pair_regex = re.compile(
        r"<arg_key>([^<]*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
        re.DOTALL,
    )

    _partial_tag_prefixes = (
        tool_call_start_token,
        tool_call_end_token,
        arg_key_start,
        arg_key_end,
        arg_value_start,
        arg_value_end,
    )

    def __init__(self):
        super().__init__()
        self.parsed_pos: int = 0
        self._state: _ParseState = _ParseState.OUTSIDE
        self.current_func_name: Optional[str] = None
        self.current_pending_key: Optional[str] = None
        self.json_started: bool = False

    # ---------- Helpers ----------

    def _reset_call_state(self) -> None:
        """Reset per-call FSM scratch fields. Called when entering a new
        <tool_call> and on </tool_call> close."""
        self.current_func_name = None
        self.current_pending_key = None
        self.json_started = False

    def _consume_arg_key(self, slice_: str) -> bool:
        """Consume `<arg_key>K</arg_key>`, set `current_pending_key` to K.
        Returns True if consumed, False if `</arg_key>` hasn't arrived yet
        (caller should break to wait for more bytes). Shared by READING_KEY
        (well-formed: transitions to READING_VALUE) and READING_VALUE
        (orphan-key-replace: stays in READING_VALUE)."""
        end = slice_.find(self.arg_key_end)
        if end == -1:
            return False
        self.current_pending_key = slice_[len(self.arg_key_start) : end].strip()
        self.parsed_pos += end + len(self.arg_key_end)
        return True

    def _close_current_call(self, calls: List[ToolCallItem]) -> None:
        """Emit the closing `}` (or `{}` for zero-arg) for the active call,
        advance past `</tool_call>`, return to OUTSIDE, and reset per-call
        state. Called from both READING_KEY (the well-formed close path) and
        READING_VALUE (malformed close: `<arg_key>...</arg_key></tool_call>`
        with no value — orphan key is discarded, matching the regex
        non-streaming path which drops unmatched <arg_key>...</arg_key>
        pairs)."""
        fragment = "}" if self.json_started else "{}"
        calls.append(
            ToolCallItem(
                tool_index=self.current_tool_id,
                parameters=fragment,
            )
        )
        self.streamed_args_for_tool[self.current_tool_id] += fragment
        self.parsed_pos += len(self.tool_call_end_token)
        self._state = _ParseState.OUTSIDE
        self._reset_call_state()

    def has_tool_call(self, text: str) -> bool:
        return self.tool_call_start_token in text

    @staticmethod
    def _get_param_schema(
        func_name: Optional[str], tools: Optional[List[Tool]]
    ) -> dict:
        if not tools or not func_name:
            return {}
        for tool in tools:
            try:
                if (
                    tool.type == "function"
                    and tool.function.name == func_name
                    and isinstance(tool.function.parameters, dict)
                ):
                    return tool.function.parameters.get("properties", {})
            except AttributeError:
                continue
        return {}

    _STRING_TYPES = frozenset({"string", "str", "text", "enum"})

    @staticmethod
    def _convert_param_value(raw: str, schema: dict, key: str) -> Any:
        """Coerce a raw arg_value string per schema; fall back to raw on failure.

        Decoder selection by schema type:
          - string-like types         → identity (raw text)
          - no schema entry           → json.loads only (conservative; don't
                                        ast-eval untyped values)
          - everything else (int,
            number, bool, object, …)  → json.loads, then ast.literal_eval

        Each decoder result is round-tripped through `json.dumps` before being
        returned; non-JSON-serializable values (sets / complex / bytes from
        `ast.literal_eval`) are rejected to the next decoder, ultimately
        falling through to the raw-string fallback rather than crashing the
        streaming JSON emission downstream.
        """
        spec = schema.get(key) if isinstance(schema, dict) else None
        param_type = str(spec.get("type", "")).lower() if isinstance(spec, dict) else ""
        if param_type in PoolsideV1Detector._STRING_TYPES:
            return raw

        decoders = (json.loads,) if not param_type else (json.loads, ast.literal_eval)
        for decoder in decoders:
            try:
                result = decoder(raw)
                # ast.literal_eval can return non-JSON-serializable values
                # (sets, complex numbers); reject so json.dumps downstream
                # doesn't choke.
                json.dumps(result)
                return result
            except (ValueError, SyntaxError, TypeError):
                continue
        return raw

    def _find_name_boundary(self, text: str) -> int:
        """Earliest of `\\n`, `<arg_key>`, `</tool_call>`. -1 if none."""
        hits = (
            text.find("\n"),
            text.find(self.arg_key_start),
            text.find(self.tool_call_end_token),
        )
        positive = [h for h in hits if h != -1]
        return min(positive) if positive else -1

    def _is_partial_tag(self, slice_: str) -> bool:
        """True if slice_ is a strict prefix of any known tag — i.e. more
        bytes might complete it into a real tag."""
        return any(
            tag.startswith(slice_) and tag != slice_
            for tag in self._partial_tag_prefixes
        )

    # ---------- Non-streaming ----------

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        if self.tool_call_start_token not in text:
            return StreamingParseResult(normal_text=text)

        tool_indices = self._get_tool_indices(tools)
        first_idx = text.find(self.tool_call_start_token)
        normal_text = text[:first_idx] if first_idx > 0 else ""

        calls: List[ToolCallItem] = []
        for body in self.tool_call_regex.findall(text):
            # _find_name_boundary searches for `\n` / `<arg_key>` /
            # `</tool_call>`, but the regex already stripped `</tool_call>`,
            # so a no-arg call without a trailing newline
            # (`<tool_call>now</tool_call>`) gives boundary == -1. Treat
            # that case as "name == entire body".
            boundary = self._find_name_boundary(body)
            name = (body if boundary == -1 else body[:boundary]).strip()
            if not name or name not in tool_indices:
                continue

            schema = self._get_param_schema(name, tools)
            args: dict = {}
            for raw_key, raw_val in self.arg_pair_regex.findall(body):
                key = raw_key.strip()
                # Strip at most one wrapping `\n` on each side (template adds
                # them around the value); preserve newlines that are part of
                # the value itself.
                val = raw_val.removeprefix("\n").removesuffix("\n")
                args[key] = self._convert_param_value(val, schema, key)

            calls.append(
                ToolCallItem(
                    tool_index=tool_indices[name],
                    name=name,
                    parameters=json.dumps(args, ensure_ascii=False),
                )
            )

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    # ---------- Streaming ----------

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._buffer += new_text
        if not self._buffer:
            return StreamingParseResult()

        tool_indices = self._get_tool_indices(tools)
        calls: List[ToolCallItem] = []
        normal_text_chunks: List[str] = []

        # No try/except: the FSM's invariants make the prior masked-IndexError
        # class unreachable, and TypeError from json.dumps is prevented at the
        # source (_convert_param_value round-trips its decoder output). If a
        # real bug surfaces, let it surface.
        while True:
            slice_ = self._buffer[self.parsed_pos :]
            if not slice_:
                break
            state = self._state

            if state is _ParseState.OUTSIDE:
                if slice_.startswith(self.tool_call_start_token):
                    self.parsed_pos += len(self.tool_call_start_token)
                    self._state = _ParseState.READING_NAME
                    self._reset_call_state()
                    continue
                if slice_.startswith("<"):
                    if self._is_partial_tag(slice_):
                        break  # could be a partial <tool_call>
                    normal_text_chunks.append("<")
                    self.parsed_pos += 1
                    continue
                next_lt = slice_.find("<")
                segment = slice_ if next_lt == -1 else slice_[:next_lt]
                normal_text_chunks.append(segment)
                self.parsed_pos += len(segment)
                continue

            if state is _ParseState.READING_NAME:
                boundary = self._find_name_boundary(slice_)
                if boundary == -1:
                    break  # name still incoming
                name = slice_[:boundary].strip()
                # Consume the name and a single delimiting newline (if
                # present). The other boundary types (<arg_key>,
                # </tool_call>) are left for the next state. boundary may
                # be 0 for a malformed `<tool_call><arg_key>...` (no
                # name); the state transition below is the loop-progress
                # guarantee.
                consume = boundary
                if boundary < len(slice_) and slice_[boundary : boundary + 1] == "\n":
                    consume += 1
                self.parsed_pos += consume

                if name and name in tool_indices:
                    self.current_tool_id += 1
                    while len(self.streamed_args_for_tool) <= self.current_tool_id:
                        self.streamed_args_for_tool.append("")
                    self.current_func_name = name
                    # Per-response sequential index — OpenAI clients group
                    # chunks by tool_index, so the name event and later
                    # parameter fragments must share this value.
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=name,
                            parameters="",
                        )
                    )
                    self._state = _ParseState.READING_KEY
                else:
                    # Unknown / empty name — drain to </tool_call> with no
                    # client-visible emission.
                    self._state = _ParseState.DRAINING
                continue

            if state is _ParseState.READING_KEY:
                if slice_.startswith(self.tool_call_end_token):
                    self._close_current_call(calls)
                    continue
                if slice_.startswith(self.arg_key_start):
                    if not self._consume_arg_key(slice_):
                        break  # incomplete <arg_key>
                    self._state = _ParseState.READING_VALUE
                    continue
                if slice_.startswith("<"):
                    if self._is_partial_tag(slice_):
                        break
                    # Bare '<' that's not any known tag — discard silently
                    # (inside a tool call, this is not normal_text).
                    self.parsed_pos += 1
                    continue
                # Inter-tag whitespace / newline — discard.
                next_lt = slice_.find("<")
                self.parsed_pos += len(slice_) if next_lt == -1 else next_lt
                continue

            if state is _ParseState.READING_VALUE:
                # Recover from a malformed `<arg_key>K</arg_key></tool_call>`
                # (no <arg_value>) by closing the call here. Without this
                # branch the FSM would stay stuck in READING_VALUE and
                # mis-attribute the next call's <arg_value> to the orphan
                # `current_pending_key`, silently swallowing the next call's
                # name. Matches the regex non-streaming path, which drops
                # unmatched <arg_key>...</arg_key> pairs.
                if slice_.startswith(self.tool_call_end_token):
                    self._close_current_call(calls)
                    continue
                # Recover from a malformed `<arg_key>K1</arg_key><arg_key>K2`
                # (no value for K1, model went straight to a new key) by
                # replacing the orphan pending_key with the new one. Stays
                # in READING_VALUE so the next <arg_value> binds to K2.
                # Without this branch the FSM treats the second <arg_key>
                # as bare-`<` garbage and the next <arg_value> binds to
                # the stale K1 — wrong-argument corruption.
                if slice_.startswith(self.arg_key_start):
                    if not self._consume_arg_key(slice_):
                        break  # incomplete <arg_key>
                    continue  # stay in READING_VALUE: orphan replaced
                if slice_.startswith(self.arg_value_start):
                    end = slice_.find(self.arg_value_end)
                    if end == -1:
                        break  # incomplete <arg_value> — no partial emission
                    raw = (
                        slice_[len(self.arg_value_start) : end]
                        .removeprefix("\n")
                        .removesuffix("\n")
                    )
                    # READING_VALUE is reachable only via READING_KEY
                    # consuming an <arg_key>...</arg_key>, so
                    # current_pending_key is set by construction.
                    schema = self._get_param_schema(self.current_func_name, tools)
                    converted = self._convert_param_value(
                        raw, schema, self.current_pending_key
                    )
                    kv = (
                        f"{json.dumps(self.current_pending_key)}: "
                        f"{json.dumps(converted, ensure_ascii=False)}"
                    )
                    fragment = "{" + kv if not self.json_started else ", " + kv
                    self.json_started = True
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            parameters=fragment,
                        )
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += fragment
                    self.current_pending_key = None
                    self.parsed_pos += end + len(self.arg_value_end)
                    self._state = _ParseState.READING_KEY
                    continue
                if slice_.startswith("<"):
                    if self._is_partial_tag(slice_):
                        break
                    self.parsed_pos += 1
                    continue
                next_lt = slice_.find("<")
                self.parsed_pos += len(slice_) if next_lt == -1 else next_lt
                continue

            if state is _ParseState.DRAINING:
                end_idx = slice_.find(self.tool_call_end_token)
                if end_idx != -1:
                    self.parsed_pos += end_idx + len(self.tool_call_end_token)
                    self._state = _ParseState.OUTSIDE
                    continue
                # Hold back trailing bytes that could be a prefix of
                # </tool_call>; the next chunk extends the tail.
                holdback = self._ends_with_partial_token(
                    slice_, self.tool_call_end_token
                )
                self.parsed_pos += len(slice_) - holdback
                break

        if self.parsed_pos > 0:
            self._buffer = self._buffer[self.parsed_pos :]
            self.parsed_pos = 0

        return StreamingParseResult(
            calls=calls,
            normal_text="".join(normal_text_chunks),
        )

    # ---------- Constrained generation ----------

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=f"<tool_call>{name}\n",
            end="</tool_call>",
            trigger="<tool_call>",
        )
