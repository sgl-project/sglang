"""Adapter from a tag-format :class:`GeneratorParser` to the detector API.

The SGLang-facing half of the tag machinery (the engine half is ``parsing.py``;
dependency rule: detectors -> this module -> ``parsing`` -> {the
``compatibility`` model, ``param_types``}).

Per the compatibility ladder (the ``compatibility`` package), detectors are pure may-raise
parsers: tolerances the grammar can absorb are recorded on the detector's
policy; anything else propagates out of ``parse_streaming_increment`` /
``detect_and_parse`` and is handled by ``FunctionCallParser`` — the single
recovery boundary — through the ``fail_open_stream`` / ``fail_open_nonstream``
hooks overridden here.
"""

import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.compatibility.mode import (
    CompatibilityEvent,
    CompatibilityMode,
)
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.parsing import GeneratorParser


class TagToolCallDetector(BaseFormatDetector, ABC):
    """Owns one tag parser per request and maps its deltas to ToolCallItems.

    Subclasses implement :meth:`_make_grammar` (and ``has_tool_call``), and
    keep their format-specific ``structure_info`` / structural-tag overrides.
    Streaming bookkeeping consumed by the serving layer
    (``prev_tool_call_arr``, ``streamed_args_for_tool``, ``current_tool_id``)
    is maintained here.

    Not picklable: holds a live generator. Detector instances are per-request
    locals in the serving layer and never cross process boundaries.
    """

    def __init__(self):
        super().__init__()
        self._grammar: Optional[GeneratorParser] = None

    @abstractmethod
    def _make_grammar(
        self, functions: Optional[Dict], compatibility: CompatibilityMode
    ) -> GeneratorParser:
        """Build a fresh parser for this request.

        ``functions`` maps each tool name to ``{"parameters": <json schema>}``
        (or is None when the request has no tools), for schema-driven
        parameter conversion. ``compatibility`` is the detector's policy; passing it
        into the parser keeps the request's tolerances on one audit trail.
        """
        raise NotImplementedError()

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError()

    def _init_grammar(self, tools: List[Tool]) -> GeneratorParser:
        if self._grammar is None:
            functions = (
                {
                    tool.function.name: {
                        "parameters": tool.function.parameters,
                    }
                    for tool in tools
                }
                if tools
                else None
            )
            self._grammar = self._make_grammar(functions, self.compatibility)
        return self._grammar

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        grammar = self._init_grammar(tools)
        # May raise (input past the grammar's local tolerances, including the
        # strict-mode unknown-tool gate, which fires before anything for the
        # call is emitted); FunctionCallParser catches and calls
        # fail_open_stream().
        grammar.update(new_text)
        return self._delta_to_result(grammar.get_delta())

    def _delta_to_result(self, delta: Optional[Dict]) -> StreamingParseResult:
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

    def _has_open_call(self) -> bool:
        """True when a call's streamed argument fragments are incomplete JSON
        (its name/arguments are on the wire and cannot be unsent)."""
        tid = self.current_tool_id
        if not (0 <= tid < len(self.streamed_args_for_tool)):
            return False
        streamed = self.streamed_args_for_tool[tid]
        if not streamed:
            return False
        try:
            json.loads(streamed)
            return False
        except json.JSONDecodeError:
            return True

    def fail_open_stream(
        self, error: Exception, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """Scope-4 recovery.

        On top of the base behavior (close a mid-stream call's JSON), drain
        output the parser produced before the error and re-surface raw
        unconsumed text as content. ``new_text`` is ignored: the failing chunk
        already sits in the parser's buffers.
        """
        grammar = self._grammar
        if grammar is None:
            return super().fail_open_stream(error, new_text, tools)

        # Drain output parsed before the error so it is not lost.
        result = self._delta_to_result(grammar.get_delta())
        calls = list(result.calls)
        normal_text = result.normal_text

        if self._has_open_call():
            # A call is mid-stream and cannot be unsent. Close its streamed
            # arguments so they concatenate to valid JSON, and keep
            # prev_tool_call_arr consistent so the serving layer's
            # end-of-stream completion does not double-send.
            calls.extend(self._close_mid_stream_call())
            # The unconsumed tail was never surfaced anywhere; flush it as
            # content. (Consumed markers of the failed call are dropped — its
            # semantic content was already emitted as tool-call deltas.)
            normal_text += grammar.unconsumed_text()
        else:
            # No call is on the wire: re-surface the entire failing construct
            # (consumed markers, skipped text, the unconsumed tail) as raw
            # text. Content already emitted — including the drain above — is
            # committed as it is emitted, so nothing is duplicated.
            normal_text += grammar.uncommitted_text()

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        grammar = self._init_grammar(tools)
        # May raise; FunctionCallParser catches and calls fail_open_nonstream().
        grammar.update(text)
        return self._nonstream_result(grammar, text, tools, failed=False)

    def fail_open_nonstream(
        self, error: Exception, full_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """Scope-4 recovery: salvage complete calls parsed before the error.

        Calls truncated by the error are dropped; their raw text and the
        unconsumed tail are re-surfaced as content (``uncommitted_text``).
        When no call survived, ``FunctionCallParser.parse_non_stream`` falls
        back to returning the full original text.
        """
        grammar = self._grammar
        if grammar is None:
            return super().fail_open_nonstream(error, full_text, tools)
        return self._nonstream_result(grammar, full_text, tools, failed=True)

    def _nonstream_result(
        self, grammar: GeneratorParser, text: str, tools: List[Tool], *, failed: bool
    ) -> StreamingParseResult:
        delta = grammar.get_delta()
        if delta is None:
            return StreamingParseResult(normal_text=text, calls=[])

        normal_text = delta.get("content", "")
        calls: List[ToolCallItem] = []
        try:
            for tc in delta.get("tool_calls", []):
                func = tc.get("function", {})
                name = func.get("name", "")
                arguments = func.get("arguments", "{}")
                try:
                    args_dict = json.loads(arguments)
                except json.JSONDecodeError:
                    if not failed:
                        # Generation ended mid-call (e.g. max_tokens). Never
                        # emit a half-specified call — an agent loop would
                        # execute it with missing or mangled arguments. Drop
                        # it; its raw text is re-surfaced below. In strict
                        # mode the note raises and the parse fails open to
                        # the same drop via the failed=True pass.
                        self.compatibility.note(
                            CompatibilityEvent.TRUNCATED_CALL_DROPPED,
                            detail=f"{name[:40]!r}: arguments cut at {arguments[-40:]!r}",
                        )
                    # failed=True: calls truncated by the parse error are
                    # dropped the same way.
                    continue
                calls.extend(
                    self._nonstream_call_items(
                        name, args_dict, tc.get("index", 0), tools
                    )
                )
        except Exception:
            # A strict-mode note raised mid-processing; put the delta back so
            # the recovery pass (failed=True) still sees the whole parse.
            grammar.restash_delta(delta)
            raise

        # Re-surface everything after the last completed construct: markers
        # and buffered text of a dropped call, skipped garbage, or the
        # held-back tail of trailing content. Completed calls and emitted
        # content are committed, so nothing is duplicated.
        normal_text += grammar.uncommitted_text()

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def _nonstream_call_items(
        self, name: str, args_dict: Dict, index: int, tools: List[Tool]
    ) -> List[ToolCallItem]:
        """Map one completed call to ToolCallItems for the non-streaming path.

        Default: route through ``parse_base_json`` (``tool_index`` is the
        index into the tools list; unknown tools are dropped per
        ``SGLANG_FORWARD_UNKNOWN_TOOLS``). Detectors that historically emitted
        the call ordinal and forwarded unknown names (e.g. Qwen3-Coder)
        override this.
        """
        return self.parse_base_json({"name": name, "arguments": args_dict}, tools)
