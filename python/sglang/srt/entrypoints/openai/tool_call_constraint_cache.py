from __future__ import annotations

import marshal
import sys
import threading
from collections import OrderedDict
from typing import Any, Iterable, List, Optional

from pydantic import BaseModel

from sglang.srt.entrypoints.openai.protocol import Tool, ToolCallConstraint, ToolChoice
from sglang.utils import convert_json_schema_to_str


def tool_call_constraint_key(
    *,
    tool_call_parser: str,
    tools: List[Tool],
    tool_strict_level: int,
    tool_choice: Any,
    parallel_tool_calls: Optional[bool],
    thinking_mode: bool,
    stripped_excludes: Optional[Iterable[str]],
) -> Optional[bytes]:
    """Everything the constraint build reads, or None when it cannot be keyed."""
    if isinstance(tool_choice, ToolChoice):
        tool_choice = (tool_choice.type, tool_choice.function.name)
    try:
        # marshal is exact-typed: NaN stays distinct from null and subclasses
        # are rejected, so equal keys mean equal build inputs.
        return marshal.dumps(
            (
                tool_call_parser,
                int(tool_strict_level),
                tool_choice,
                parallel_tool_calls,
                thinking_mode,
                None if stripped_excludes is None else tuple(stripped_excludes),
                tuple(_tool_key(tool) for tool in tools),
            )
        )
    except ValueError:
        return None


def _tool_key(tool: Tool) -> tuple:
    function = tool.function
    return (
        tool.type,
        tool.defer_loading,
        function.name,
        function.description,
        function.parameters,
        function.strict,
        function.defer_loading,
        function.key_order,
    )


class ToolCallConstraintCache:
    """LRU of serialized tool-call constraints, bounded in entries and bytes."""

    def __init__(self, *, max_entries: int, max_bytes: int):
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._entries: OrderedDict[bytes, tuple[ToolCallConstraint, int]] = (
            OrderedDict()
        )
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key: bytes) -> Optional[ToolCallConstraint]:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
            return entry[0]

    def put(
        self, key: bytes, constraint: Optional[ToolCallConstraint]
    ) -> Optional[ToolCallConstraint]:
        """Serialize `constraint` as `to_sampling_params` would, store it, and
        return it; constraints of other kinds are returned unchanged."""
        if constraint is None:
            return None
        kind, value = constraint
        if kind == "structural_tag" and isinstance(value, BaseModel):
            serialized = convert_json_schema_to_str(value.model_dump(by_alias=True))
        elif kind == "json_schema" and isinstance(value, dict):
            serialized = convert_json_schema_to_str(value)
        else:
            return constraint
        constraint = (kind, serialized)
        size = sys.getsizeof(key) + sys.getsizeof(serialized)
        if size > self._max_bytes:
            return constraint
        with self._lock:
            if key in self._entries:
                return constraint
            self._entries[key] = (constraint, size)
            self._bytes += size
            while len(self._entries) > self._max_entries or (
                self._bytes > self._max_bytes
            ):
                _, (_, evicted) = self._entries.popitem(last=False)
                self._bytes -= evicted
        return constraint
