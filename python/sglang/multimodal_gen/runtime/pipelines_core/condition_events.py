# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_MISSING = object()


@dataclass(frozen=True)
class ControlSignal:
    kind: str
    payload: Any
    timestamp_ms: int | None = None
    seq_id: int | None = None


@dataclass(frozen=True)
class ConditionEvent:
    """transport envelope for one or more same-kind control signals"""

    kind: str
    payload: Any

    def iter_signals(self, expand_payload: bool = True):
        items = (
            self.payload
            if self._should_expand_payload(self.payload, expand_payload)
            else (self.payload,)
        )
        for item in items:
            if isinstance(item, ControlSignal):
                if item.kind != self.kind:
                    raise ValueError(
                        "control signal kind "
                        f"{item.kind!r} does not match event kind {self.kind!r}"
                    )
                yield item
            else:
                yield ControlSignal(kind=self.kind, payload=item)

    @staticmethod
    def _should_expand_payload(payload: Any, expand_payload: bool) -> bool:
        return (
            expand_payload
            and isinstance(payload, Sequence)
            and not isinstance(payload, (str, bytes, bytearray))
        )


@dataclass(frozen=True)
class ConditionSamplingParams:
    chunk_size: int
    default_item: Any = _MISSING
    repeat_last: bool = True
    expand_payload: bool = True


class ConditionEventQueue:
    """per-session queue for prompt, camera, audio, and future events

      all events are stored here for per-chunk sampling
    """

    def __init__(
        self,
        max_events: int | Mapping[str, int] = 512,
    ) -> None:
        self._max_events = max_events
        self._events: dict[str, deque[ConditionEvent]] = {}
        self._pending_signals: dict[str, deque[ControlSignal]] = {}
        self._last_payloads: dict[str, Any] = {}
        self._seen_kinds: set[str] = set()

    def push(self, event: ConditionEvent) -> None:
        queue = self._queue_for(event.kind)
        queue.append(event)
        self._seen_kinds.add(event.kind)

    def replace(self, event: ConditionEvent) -> None:
        self.clear_kind(event.kind)
        self.push(event)

    def pop_latest(self, kind: str) -> Any | None:
        queue = self._events.get(kind)
        if not queue:
            return None
        latest_payload = None
        has_signal = False
        for signal in queue.pop().iter_signals():
            latest_payload = signal.payload
            has_signal = True
        queue.clear()
        self._seen_kinds.add(kind)
        if not has_signal:
            return None
        return latest_payload

    def has_events(self, kind: str) -> bool:
        queue = self._events.get(kind)
        pending = self._pending_signals.get(kind)
        return bool(queue) or bool(pending)

    def sample_chunk(
        self,
        kind: str,
        params: ConditionSamplingParams,
    ) -> list[Any] | None:
        """samples a list of actions for a chunk

        Args:
            params: the sampling strategy

        """
        if params.chunk_size <= 0:
            return None

        chunk: list[Any] = []
        pending = self._pending_signals.get(kind)
        self._drain_signals(kind, pending, chunk, params.chunk_size)

        queue = self._events.get(kind)
        while len(chunk) < params.chunk_size and queue:
            event = queue.popleft()
            signals = deque(event.iter_signals(params.expand_payload))
            self._drain_signals(kind, signals, chunk, params.chunk_size)
            if signals:
                self._pending_signals[kind] = signals

        if len(chunk) == 0 and kind not in self._seen_kinds:
            if params.default_item is _MISSING:
                return None
            return [params.default_item for _ in range(params.chunk_size)]

        if len(chunk) == 0:
            if params.default_item is _MISSING:
                return None
            return [params.default_item for _ in range(params.chunk_size)]

        if not params.repeat_last:
            return chunk

        pad_item = self._last_payloads.get(kind, params.default_item)
        if pad_item is _MISSING:
            return chunk
        while len(chunk) < params.chunk_size:
            chunk.append(pad_item)
        return chunk

    def clear(self) -> None:
        self._events.clear()
        self._pending_signals.clear()
        self._last_payloads.clear()
        self._seen_kinds.clear()

    def clear_kind(self, kind: str) -> None:
        self._events.pop(kind, None)
        self._pending_signals.pop(kind, None)
        self._last_payloads.pop(kind, None)
        self._seen_kinds.discard(kind)

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

    def _drain_signals(
        self,
        kind: str,
        signals: deque[ControlSignal] | None,
        chunk: list[Any],
        chunk_size: int,
    ) -> None:
        while signals and len(chunk) < chunk_size:
            signal = signals.popleft()
            chunk.append(signal.payload)
            self._last_payloads[kind] = signal.payload
