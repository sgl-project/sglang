"""Incremental parsing engine for tag-style tool-call wire formats.

Most tag formats are variations of one shape::

    [content] BLOCK_OPEN [per-call: name tag, parameter tags] BLOCK_CLOSE [content] ...

with the function name in a tag attribute and **each parameter in its own
tag whose body is raw text** (not JSON). Parsing such formats needs what a
regex + ``json.loads`` cannot do: convert tag names to JSON keys, convert raw
text values via the tool schema (``compatibility/param_types.py``), stream string values as
escaped JSON fragments, and keep state when a tag is split across streaming
chunks. Formats that wrap a complete JSON object (Qwen2.5, Hermes, ...) should
use ``BaseFormatDetector`` directly instead.

This module is the format-agnostic half of the tag machinery and an
implementation detail of the detectors that use it; nothing here is a public
extension point. It contains:

- ``PatternMismatched``: the engine's parse-failure signal.
- ``GeneratorParser``: the coroutine engine. Subclasses write their format as
  a generator ``_process()`` using primitives like ``_literal`` and
  ``_take_any``; ``update(text)`` advances the coroutine, which suspends
  automatically when it needs more input.
- ``TagToolCallParser``: shared helpers for tag formats (content/skip/emit/
  parameter-value pieces), so each format's ``_process()`` stays a short
  transcription of its grammar.

The SGLang-facing adapter lives in ``tag_format_detector.py``. Dependency rule
for this package: detectors -> ``tag_format_detector`` -> ``parsing`` ->
{``compatibility``, ``param_types``}; this module must not import serving types.

The delta contract between a parser and the adapter is the dict shape built
by ``default_tool_call_output_key``::

    {
        "content": str,                  # optional normal text
        "tool_calls": [
            {
                "index": int,            # 0-based ordinal of the call
                "function": {
                    "name": str,         # exactly once, on the first delta
                    "arguments": str,    # JSON fragments; concatenate to a
                                         # valid JSON object
                },
            }
        ],
    }

Parsers emit only ``content`` and ``tool_calls`` (reasoning is handled by
SGLang's separate ``--reasoning-parser`` stage).

Tolerances applied while parsing are recorded on the parser's
:class:`CompatibilityMode` (injected at construction — the detector's policy, so
all tolerances for a request share one audit trail); coroutine code cannot
wrap ``yield`` in ``CompatibilityMode.absorb``, so the generator helpers here call
``note()`` directly. Errors the grammar cannot absorb propagate as
``PatternMismatched`` to ``FunctionCallParser``, which fails open to text.

TODO: accept token-level inputs (match special-token markers by token id
instead of by string) to remove partial-marker holdback.
"""

import functools
import json
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from typing_extensions import Self

from sglang.srt.environ import envs
from sglang.srt.function_call.compatibility.mode import (
    CompatibilityEvent,
    CompatibilityMode,
    CompatibilityRecord,
)
from sglang.srt.function_call.compatibility.param_types import (
    NULL_STRINGS,
    FunctionCallParameterDataType,
)

OutputKey = Union[str, Callable[[Any], Dict]]


class PatternMismatched(Exception):
    def __init__(
        self,
        *,
        offset: int,
        expected: Any,
        actual: Any,
        reason: str,
        context_fn: Optional[Callable[[], str]] = None,
    ):
        self.offset = offset
        self.expected = expected
        self.actual = actual
        self.reason = reason
        self.context_fn = context_fn

    def __str__(self):
        main_message = f"at offset {self.offset}, {self.reason}, expected {self.expected!r} but got {self.actual!r}"
        if self.context_fn is None:
            return main_message
        else:
            return f"around context:\n{self.context_fn()}\n\n{main_message}"


class GeneratorParser(ABC):
    """Coroutine-based incremental text parser; see the module docstring."""

    @abstractmethod
    def _process(self) -> Generator[Dict[str, Any], None, None]:
        pass

    def __init__(self, *, compatibility: CompatibilityMode):
        # The policy is required by construction so a request's tolerances
        # cannot silently fork onto a second audit trail. Must be set before
        # `_process()` starts: the grammar may consult it from its first step.
        self.compatibility = compatibility
        self._full_text: List[str] = []
        self._input_length = 0
        # The unconsumed window is ``self._buf[self._pos:]``; ``update()``
        # compacts the consumed prefix away on each call.
        self._buf = ""
        self._pos = 0
        # Raw-text offset up to which everything is *accounted for* — either
        # emitted (content) or structure of a successfully completed
        # construct. Grammars advance it via ``_commit()``; the error path
        # re-surfaces ``uncommitted_text()`` so no raw text is lost.
        self._committed_offset = 0
        self._delta: Optional[Dict[str, Any]] = None
        self._final: Optional[Dict[str, Any]] = None
        self._generator = self._process()
        next(self._generator)

    def __del__(self):
        # Necessary when GC is disabled (gc.disable()),
        # otherwise the cyclic reference will never be collected.
        if hasattr(self, "_generator"):
            self._generator.close()
            del self._generator

    def update(self, input: str):
        self._full_text.append(input)
        self._input_length += len(input)
        self._buf = self._buf[self._pos :] + input
        self._pos = 0
        # Resume the coroutine until it stops making progress (it is starved,
        # holding back a possible partial marker) or the grammar ends.
        while self._pos < len(self._buf):
            before = self._pos
            try:
                next(self._generator)
            except StopIteration:
                if self._pos < len(self._buf):
                    raise self._error(
                        expected=None,
                        actual=self._buf[self._pos],
                        reason="pattern exhausted",
                    )
                break
            if self._pos == before:
                break

    def get_delta(self) -> Optional[Dict[str, Any]]:
        if self._delta:
            delta = _get_dict_with_joint_strings(self._delta)
            self._delta = None
            return delta

    def get_final(self) -> Optional[Dict[str, Any]]:
        return _get_dict_with_joint_strings(self._final)

    def restash_delta(self, delta: Optional[Dict[str, Any]]):
        """Put a delta taken by ``get_delta()`` back, ahead of newer output.

        Used by the adapter when post-parse processing of a drained delta
        raises (e.g. a strict-mode gate): the delta must stay available for
        the recovery drain or its content would be lost.
        """
        if delta:
            self._delta = _merge_dicts(delta, self._delta)

    def unconsumed_text(self) -> str:
        """Text received so far that the grammar has not yet consumed.

        Read-only; does not advance the generator. Used by TagToolCallDetector
        on the error path to recover the raw tail as normal content.
        """
        return self._buf[self._pos :]

    def uncommitted_text(self) -> str:
        """Raw text after the last ``_commit()`` point (see ``__init__``).

        On the error path this is the entire failing construct — consumed
        markers, silently buffered names, skipped garbage — plus the
        unconsumed tail; everything before it was either emitted as content
        or belongs to a successfully completed construct.
        """
        return "".join(self._full_text)[self._committed_offset :]

    def full_text_received(self) -> str:
        """All text passed to ``update()`` so far (consumed and not)."""
        return "".join(self._full_text)

    def _consumed_offset(self) -> int:
        return self._input_length - (len(self._buf) - self._pos)

    def _commit(self):
        """Mark everything consumed or emitted so far as accounted for."""
        self._committed_offset = self._consumed_offset()

    def _note(self, event: CompatibilityEvent, detail: str = "") -> Optional[CompatibilityRecord]:
        """Record an applied tolerance at the current consume offset.

        Raises ``CompatibilityViolation`` in strict mode; see the ``compatibility`` package.
        """
        return self.compatibility.note(event, detail, offset=self._consumed_offset())

    def _peek(self, index: int) -> Generator[None, None, str]:
        while self._pos + index >= len(self._buf):
            yield
        return self._buf[self._pos + index]

    def _read_one(self) -> str:
        # intentionally unchecked, will raise IndexError if the input buffer is empty.
        ch = self._buf[self._pos]
        self._pos += 1
        return ch

    def _consume(self, length: int):
        self._pos += length

    def _append(self, key: OutputKey, value: Any):
        if isinstance(key, str):
            delta = {key: value}
        else:
            delta = key(value)
        self._append_delta(delta)

    def _append_delta(self, delta: Dict[str, Any]):
        self._delta = _merge_dicts(delta, self._delta)
        self._final = _merge_dicts(delta, self._final)

    def _error(self, *, expected: Any, actual: Any, reason: str) -> PatternMismatched:
        offset = self._consumed_offset()
        # _full_text is a plain list; capturing it directly (instead of self)
        # keeps the context_fn closure free of a reference cycle.
        full_text_ref = self._full_text
        return PatternMismatched(
            offset=offset,
            expected=expected,
            actual=actual,
            reason=reason,
            context_fn=functools.partial(
                _get_context_standalone, full_text_ref, offset
            ),
        )

    def _take_any(
        self,
        *,
        until: Optional[Union[str, Tuple[str, ...]]] = None,
        key: Optional[OutputKey] = None,
        should_consume_suffix: bool = True,
        commit_each: bool = False,
    ) -> Generator[str, None, None]:
        """``commit_each`` commits after every emitted run; pass it when the
        output is surfaced live (content), so the error path never
        re-surfaces it (see ``uncommitted_text``)."""
        if until is None:
            # Never ends without `until`, thus no need to collect the values.
            while True:
                yield from self._peek(0)
                run = self._buf[self._pos :]
                self._pos = len(self._buf)
                self._append(key, run)
                if commit_each:
                    self._commit()
        else:
            firsts = _target_first_chars(until)
            values = []
            while True:
                tried = yield from self._literal(
                    until,
                    should_consume=should_consume_suffix,
                    should_raise=False,
                )
                if tried is not None:
                    break
                run = self._take_run(firsts)
                values.append(run)
                if key is not None:
                    self._append(key, run)
                if commit_each:
                    self._commit()
            return "".join(values)

    def _take_run(self, firsts: Tuple[str, ...]) -> str:
        """Consume the char at the current position (it starts no target —
        ``_literal`` just said so) plus, in bulk with C-speed ``str.find``,
        every following char that provably cannot start any target either."""
        start = self._pos
        self._pos += 1
        nxt = len(self._buf)
        for c in firsts:
            i = self._buf.find(c, self._pos)
            if i != -1 and i < nxt:
                nxt = i
        self._pos = nxt
        return self._buf[start:nxt]

    def _take_data_type_as_json(
        self,
        *,
        until: Union[str, Tuple[str, ...]],
        key: OutputKey,
        data_type: FunctionCallParameterDataType,
        always_nullable: bool = False,
        should_consume_suffix: bool = True,
    ):
        """
        Unlike `.take_any`, `until` and `key` is required here, because
        - it is unnecessary and a little bit annoying to maintain the streaming state
          for most data types other than `str`.
        - if you don't need the output, just use `.take_any`.

        NOTE: if there are various acceptable data types, we choose the FIRST convertible
        one, indicating, if the first choice is `str`, the following choices will be
        IGNORED, as everything could be a string.

        TODO: if there are acceptable data types before `str`, we should peek until they
        are excluded and then choose `str`, instead of choosing at the end.
        """
        if not data_type.streaming:
            value = yield from self._take_any(
                until=until, should_consume_suffix=should_consume_suffix
            )
            value = data_type.convert(
                value, always_nullable=always_nullable, compatibility=self.compatibility
            )
            self._append(key, json_dumps(value))
            return value
        else:
            if always_nullable:
                # very tricky.
                # even if the data type is string, we still need to check if it's null.
                tried = yield from self._literal(
                    _string_cartesian_product(NULL_STRINGS, until),
                    should_consume=should_consume_suffix,
                    should_raise=False,
                )
                if tried is not None:
                    if not should_consume_suffix:
                        null = next(n for n in NULL_STRINGS if tried.startswith(n))
                        self._consume(len(null))
                    self._append(key, "null")
                    return None
            firsts = _target_first_chars(until)
            values = []
            self._append(key, '"')
            while True:
                tried = yield from self._literal(
                    until, should_consume=should_consume_suffix, should_raise=False
                )
                if tried is not None:
                    break
                run = self._take_run(firsts)
                values.append(run)
                # JSON string escaping is per-character and context-free, so
                # escaping a run equals concatenating escaped characters;
                # [1:-1] strips only the surrounding quotes.
                self._append(key, json_dumps(run)[1:-1])
            self._append(key, '"')
            return "".join(values)

    def _literal(
        self,
        target: Union[str, Tuple[str, ...]],
        *,
        should_raise: bool = True,
        should_consume: bool = True,
    ) -> Generator[Optional[str], None, None]:
        """
        NOTE: stops at the first matched target, even if it is a substring of another later target.
        """
        if isinstance(target, str):
            target = (target,)
        char_index = 0
        target_index = 0
        target_index_max = len(target) - 1
        while True:
            ith = yield from self._peek(char_index)
            if ith != target[target_index][char_index]:
                already_matched_and_ith = target[target_index][:char_index] + ith
                prefix_matching_result = PrefixMatchingResult.mismatch
                target_index += 1
                while target_index <= target_index_max:
                    prefix_matching_result = PrefixMatchingResult.check(
                        target[target_index], already_matched_and_ith
                    )
                    if prefix_matching_result != PrefixMatchingResult.mismatch:
                        break
                    target_index += 1
                if prefix_matching_result == PrefixMatchingResult.substring_of_prefix:
                    break
                elif prefix_matching_result == PrefixMatchingResult.mismatch:
                    if should_raise:
                        if len(target) == 1:
                            raise self._error(
                                expected=target[0][char_index:],
                                actual=ith,
                                reason=f"matching {target[0]!r}",
                            )
                        else:
                            raise self._error(
                                expected=target,
                                actual=ith,
                                reason="matching any of the list",
                            )
                    else:
                        return
            char_index += 1
            if char_index == len(target[target_index]):
                break
        if should_consume:
            self._consume(len(target[target_index]))
        return target[target_index]

    def _whitespace(self) -> Iterator[None]:
        total = 0
        while True:
            peek = yield from self._peek(total)
            if peek.isspace():
                total += 1
            else:
                self._consume(total)
                break


def default_tool_call_output_key(tool_call_index: int, function: Dict) -> Dict:
    return {
        "tool_calls": [
            {
                "index": tool_call_index,
                "function": function,
            }
        ]
    }


def _get_context_standalone(full_text_parts: List[str], offset: int) -> str:
    """Standalone version of the error context that doesn't capture self."""
    full_text = "".join(full_text_parts)
    return f"{full_text[:offset]}💥{full_text[offset:]}"


def _merge_dicts(
    src: Dict[str, Any],
    dst: Optional[Dict[str, Any]],
    whitelist: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    if dst is None:
        return deepcopy(src)
    for key, val_src in src.items():
        if whitelist is not None and key in whitelist:
            continue
        val_dst = dst.get(key)
        if val_dst is None:
            dst[key] = deepcopy(val_src)
        elif isinstance(val_src, dict):
            dst[key] = _merge_dicts(val_src, val_dst)
        elif isinstance(val_src, list):
            # NOTE: don't just `val_dst.extend(val_src)` here, as we need a carefully constructed delta object.
            # Assume that all elements have `index: int`, indicating its position in the final merged list.
            if len(val_src) == len(val_dst) and all(
                val_src[i]["index"] == val_dst[i]["index"] for i in range(len(val_src))
            ):
                for i in range(len(val_src)):
                    _merge_dicts(val_src[i], val_dst[i], whitelist={"index"})
            elif len(val_src):
                index_to_item = {item["index"]: item for item in val_dst}
                for item_src in val_src:
                    idx = item_src["index"]
                    item_dst = index_to_item.get(idx)
                    if item_dst is None:
                        index_to_item[idx] = deepcopy(item_src)
                    else:
                        _merge_dicts(item_src, item_dst, whitelist={"index"})
                dst[key] = sorted(
                    index_to_item.values(), key=lambda item: item["index"]
                )
        elif isinstance(val_src, str):
            if isinstance(val_dst, list):
                dst[key].append(val_src)
            else:
                # NOTE: to avoid O(n²) string concatenation, we store one string as a list of substrings.
                # Later, join the substrings back by `_get_dict_with_joint_strings`.
                dst[key] = [val_dst, val_src]
        else:
            raise TypeError(
                f"key {key} has unsupported type {type(val_src)} of {val_src!r} against {val_dst!r}"
            )
    return dst


def _get_dict_with_joint_strings(
    data: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if data is None:
        return
    updated = {}
    for key, val in data.items():
        if isinstance(val, list):
            if val:
                if isinstance(val[0], str):
                    updated[key] = "".join(val)
                elif isinstance(val[0], dict):
                    updated[key] = [_get_dict_with_joint_strings(_) for _ in val]
                else:
                    updated[key] = val
            else:
                updated[key] = val
        elif isinstance(val, dict):
            updated[key] = _get_dict_with_joint_strings(val)
        else:
            updated[key] = val
    return updated


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


class PrefixMatchingResult(int, Enum):
    startswith = auto()
    substring_of_prefix = auto()
    mismatch = auto()

    @classmethod
    def check(cls, string: str, prefix: str) -> Self:
        if not prefix:
            return cls.startswith
        for i in range(len(prefix)):
            if i >= len(string):
                return cls.substring_of_prefix
            if string[i] != prefix[i]:
                return cls.mismatch
        return cls.startswith


@functools.cache
def _target_first_chars(until: Union[str, Tuple[str, ...]]) -> Tuple[str, ...]:
    targets = (until,) if isinstance(until, str) else until
    return tuple({t[0] for t in targets})


@functools.cache
def _string_cartesian_product(
    prefix: Tuple[str, ...], suffix: Union[str, Tuple[str, ...]]
) -> Tuple[str, ...]:
    if isinstance(suffix, str):
        return tuple(_ + suffix for _ in prefix)
    else:
        return tuple(_prefix + _suffix for _prefix in prefix for _suffix in suffix)


class TagToolCallParser(GeneratorParser, ABC):
    """Shared building blocks for tag-format grammars; see module docstring."""

    def __init__(
        self,
        *,
        compatibility: CompatibilityMode,
        functions: Optional[Dict] = None,
        content_field: str = "content",
        tool_call_output_key: Callable[
            [int, Dict], Dict
        ] = default_tool_call_output_key,
        gate_unknown_tools: bool = True,
    ):
        self._functions = functions
        self._content_field = content_field
        self._tool_call_output_key = tool_call_output_key
        # Whether _drop_unknown_invoke applies. Detectors whose historical
        # contract forwards unknown names (Qwen3-Coder) pass False.
        self._gate_unknown_tools = gate_unknown_tools
        super().__init__(compatibility=compatibility)

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------

    def _get_function(self, function_name: str) -> Optional[Dict]:
        if isinstance(self._functions, dict):
            return self._functions.get(function_name)

    def _param_data_type(
        self, function: Optional[Dict], parameter_name: str
    ) -> FunctionCallParameterDataType:
        return FunctionCallParameterDataType.get_schema_of_parameter(
            function, parameter_name
        )

    # ------------------------------------------------------------------
    # Emit helpers (deltas in the adapter contract shape)
    # ------------------------------------------------------------------

    def _emit_args(self, tool_call_index: int, fragment: str) -> None:
        self._append_delta(
            self._tool_call_output_key(tool_call_index, {"arguments": fragment})
        )

    def _emit_name(self, tool_call_index: int, function_name: str) -> None:
        """Emit the function name plus the opening ``{`` of its arguments."""
        self._append_delta(
            self._tool_call_output_key(
                tool_call_index, {"name": function_name, "arguments": "{"}
            )
        )

    def _emit_call_close(self, tool_call_index: int) -> None:
        """Close the call's argument object and commit it as one step.

        Closing and committing are fused so a grammar cannot complete a call
        without also marking its raw text as accounted for — a later
        failure's ``uncommitted_text()`` must never re-surface a call that
        was already delivered as tool-call deltas.
        """
        self._emit_args(tool_call_index, "}")
        self._commit()

    def _emit_param_key(
        self, tool_call_index: int, parameter_name: str, is_first_parameter: bool
    ) -> None:
        self._emit_args(
            tool_call_index,
            "{}{}: ".format(
                "" if is_first_parameter else ", ", json_dumps(parameter_name)
            ),
        )

    # ------------------------------------------------------------------
    # Parsing helpers (generators; use with ``yield from``)
    # ------------------------------------------------------------------

    def _content_until(self, marker: str) -> Generator:
        """Stream content up to ``marker``, then consume the marker.

        Partial-marker holdback at chunk boundaries is inherent: text is only
        emitted as content once it provably is not a prefix of ``marker``.

        Content is committed as it is emitted, and once more when the marker
        matches, so the committed offset sits exactly at the marker: if the
        construct it opens later fails, ``uncommitted_text()`` re-surfaces
        the whole construct including this marker.
        """
        self._commit()
        yield from self._take_any(
            until=marker,
            key=self._content_field,
            should_consume_suffix=False,
            commit_each=True,
        )
        self._commit()
        yield from self._literal(marker)

    def _skip_garbage(
        self,
        *,
        until: Union[str, Tuple[str, ...]],
        detail: str = "",
        consume_suffix: bool = False,
    ) -> Generator:
        """Compatibility scope 1: drop unparseable text up to a recognizable marker.

        Records ``SKIPPED_GARBAGE`` only when non-whitespace text was actually
        dropped (in strict mode that record raises ``CompatibilityViolation``); the
        recorded offset points at the start of the dropped text.
        """
        offset = self._consumed_offset()
        skipped = yield from self._take_any(
            until=until, should_consume_suffix=consume_suffix
        )
        if skipped.strip():
            self.compatibility.note(
                CompatibilityEvent.SKIPPED_GARBAGE,
                detail=f"{detail}: {skipped[:80]!r}" if detail else repr(skipped[:80]),
                offset=offset,
            )
        return skipped

    def _recover_invoke(self, tool_call_index: int, end_marker: str) -> Generator:
        """Compatibility scope 1, last resort inside a call: drop the rest of it.

        Consume (and drop) everything up to ``end_marker`` and close the
        argument object so the streamed fragments stay valid JSON, hoping the
        format recovers after this call ends. Records ``DROPPED_INVOKE_TAIL``
        (raises in strict mode, before the synthesized close is emitted).
        """
        offset = self._consumed_offset()
        dropped = yield from self._take_any(until=end_marker)
        self.compatibility.note(
            CompatibilityEvent.DROPPED_INVOKE_TAIL,
            detail=repr(dropped[:80]),
            offset=offset,
        )
        self._emit_call_close(tool_call_index)

    def _drop_unknown_invoke(self, function_name: str, end_marker: str) -> Generator:
        """Compatibility scope 1: gate a call to a tool absent from the request.

        Runs the moment the grammar has parsed the function name — before
        anything for the call has been emitted — so in strict mode the
        ``note`` raises with the whole call still uncommitted and the
        fail-open path re-surfaces its raw text exactly, independent of
        chunking. In compatibility mode the invoke is consumed (through
        ``end_marker``), audited as ``UNKNOWN_TOOL_DROPPED``, and committed;
        surviving calls keep dense ordinals because the caller skips its
        index bookkeeping entirely. Honors ``SGLANG_FORWARD_UNKNOWN_TOOLS``
        and the detector's ``gate_unknown_tools`` contract.

        Returns True when the invoke was dropped (the caller continues with
        the next construct).
        """
        if (
            not self._gate_unknown_tools
            or self._functions is None
            or function_name in self._functions
            or envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get()
        ):
            return False
        self._note(
            CompatibilityEvent.UNKNOWN_TOOL_DROPPED, detail=repr(function_name)
        )
        yield from self._take_any(until=end_marker)
        self._commit()
        return True

    def _param_value(
        self,
        *,
        tool_call_index: int,
        until: Union[str, Tuple[str, ...]],
        data_type: FunctionCallParameterDataType,
        always_nullable: bool = False,
        strip: Optional[str] = None,
        should_consume_suffix: bool = True,
    ) -> Generator:
        """Read one parameter value and emit it as a JSON argument fragment.

        With ``strip=None``, delegates to ``_take_data_type_as_json``:
        string-typed parameters stream character by character, other types are
        buffered until ``until`` and converted via the schema. To exclude a
        single newline before ``until`` from the value, pass it as an
        alternative (e.g. ``until=("\\n</parameter>", "</parameter>")``).

        ``strip`` (``"one_newline"`` or ``"all"``) buffers the whole value
        first — required when the format strips more than an ``until``
        alternative can express.
        """
        if strip is None:
            value = yield from self._take_data_type_as_json(
                until=until,
                key=lambda fragment: self._tool_call_output_key(
                    tool_call_index, {"arguments": fragment}
                ),
                data_type=data_type,
                always_nullable=always_nullable,
                should_consume_suffix=should_consume_suffix,
            )
            return value

        raw = yield from self._take_any(
            until=until, should_consume_suffix=should_consume_suffix
        )
        if strip == "one_newline":
            if raw.startswith("\n"):
                raw = raw[1:]
            if raw.endswith("\n"):
                raw = raw[:-1]
        elif strip == "all":
            raw = raw.strip()
        value = data_type.convert(
            raw, always_nullable=always_nullable, compatibility=self.compatibility
        )
        self._emit_args(tool_call_index, json_dumps(value))
        return value
