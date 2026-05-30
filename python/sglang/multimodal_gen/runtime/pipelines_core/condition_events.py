# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_MISSING = object()


@dataclass(frozen=True)
class ConditionEvent:
    kind: str
    payload: Any


@dataclass(frozen=True)
class ConditionSamplingParams:
    chunk_size: int
    default_item: Any = _MISSING
    repeat_last: bool = True
    expand_payload: bool = True


class ConditionEventQueue:
    """generic per-session queue for prompt, camera, audio, and future events"""

    def __init__(
        self,
        max_events: int | Mapping[str, int] = 512,
    ) -> None:
        self._max_events = max_events
        self._events: dict[str, deque[ConditionEvent]] = {}
        self._pending_items: dict[str, deque[Any]] = {}
        self._last_items: dict[str, Any] = {}
        self._seen_kinds: set[str] = set()

    def push(self, event: ConditionEvent) -> None:
        queue = self._queue_for(event.kind)
        queue.append(event)
        self._seen_kinds.add(event.kind)

    def pop_latest(self, kind: str) -> Any | None:
        queue = self._events.get(kind)
        if not queue:
            return None
        latest = queue.pop().payload
        queue.clear()
        self._seen_kinds.add(kind)
        return latest

    def has_events(self, kind: str) -> bool:
        queue = self._events.get(kind)
        pending = self._pending_items.get(kind)
        return bool(queue) or bool(pending)

    def sample_chunk(
        self,
        kind: str,
        params: ConditionSamplingParams,
    ) -> list[Any] | None:
        if params.chunk_size <= 0:
            return None

        chunk: list[Any] = []
        pending = self._pending_items.get(kind)
        self._drain_items(kind, pending, chunk, params.chunk_size)

        queue = self._events.get(kind)
        while len(chunk) < params.chunk_size and queue:
            event = queue.popleft()
            items = deque(self._iter_event_items(event.payload, params.expand_payload))
            self._drain_items(kind, items, chunk, params.chunk_size)
            if items:
                self._pending_items[kind] = items

        if len(chunk) == 0 and kind not in self._seen_kinds:
            if params.default_item is _MISSING:
                return None
            return [params.default_item for _ in range(params.chunk_size)]

        if not params.repeat_last:
            return chunk

        pad_item = self._last_items.get(kind, params.default_item)
        if pad_item is _MISSING:
            return chunk
        while len(chunk) < params.chunk_size:
            chunk.append(pad_item)
        return chunk

    def clear(self) -> None:
        self._events.clear()
        self._pending_items.clear()
        self._last_items.clear()
        self._seen_kinds.clear()

    def _queue_for(self, kind: str) -> deque[ConditionEvent]:
        queue = self._events.get(kind)
        if queue is None:
            if isinstance(self._max_events, Mapping):
                maxlen = self._max_events.get(kind, 512)
            else:
                maxlen = self._max_events
            queue = deque(maxlen=maxlen)
            self._events[kind] = queue
        return queue

    @staticmethod
    def _iter_event_items(payload: Any, expand_payload: bool):
        if (
            expand_payload
            and isinstance(payload, Sequence)
            and not isinstance(payload, (str, bytes, bytearray))
        ):
            yield from payload
        else:
            yield payload

    def _drain_items(
        self,
        kind: str,
        items: deque[Any] | None,
        chunk: list[Any],
        chunk_size: int,
    ) -> None:
        while items and len(chunk) < chunk_size:
            item = items.popleft()
            chunk.append(item)
            self._last_items[kind] = item
