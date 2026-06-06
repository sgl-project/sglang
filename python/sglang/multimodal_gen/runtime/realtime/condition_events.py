# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
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
class ControlStateTransition:
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
    repeat_last_across_empty_chunks: bool = False
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
        self._last_sampled_seq_ids: dict[str, int | None] = {}
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
            self._last_sampled_seq_ids[kind] = signal.seq_id
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
            if params.repeat_last_across_empty_chunks and kind in self._last_payloads:
                return [self._last_payloads[kind] for _ in range(params.chunk_size)]
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
        self._last_sampled_seq_ids.clear()
        self._seen_kinds.clear()

    def clear_kind(self, kind: str) -> None:
        self._events.pop(kind, None)
        self._pending_signals.pop(kind, None)
        self._last_payloads.pop(kind, None)
        self._last_sampled_seq_ids.pop(kind, None)
        self._seen_kinds.discard(kind)

    def last_sampled_seq_id(self, kind: str) -> int | None:
        return self._last_sampled_seq_ids.get(kind)

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
            self._last_sampled_seq_ids[kind] = signal.seq_id


class ControlStateSamplingQueue:
    """state-based control sampler for realtime inputs"""

    def __init__(
        self,
        *,
        default_item: Any,
        min_pulse_items: int = 1,
        max_transitions: int = 512,
    ) -> None:
        self.default_item = default_item
        self.min_pulse_items = min_pulse_items
        self._pending: deque[ControlStateTransition] = deque(maxlen=max_transitions)
        self._current_item = default_item
        self._current_seq_id: int | None = None
        self._latest_sampled_seq_id: int | None = None

    def clear(self) -> None:
        self._pending.clear()
        self._current_item = self.default_item
        self._current_seq_id = None
        self._latest_sampled_seq_id = None

    def push(self, transition: ControlStateTransition) -> None:
        self._pending.append(transition)

    def push_many(self, transitions: Sequence[ControlStateTransition]) -> None:
        for transition in transitions:
            self.push(transition)

    def sample_chunk(self, chunk_size: int) -> list[Any] | None:
        if chunk_size <= 0:
            return None

        transitions = self._drain_pending()
        if not transitions:
            self._latest_sampled_seq_id = self._current_seq_id
            return [self._copy_item(self._current_item) for _ in range(chunk_size)]

        pulse = self._latest_non_default_transition(transitions)
        final = transitions[-1]
        self._current_item = final.payload
        self._current_seq_id = final.seq_id

        if pulse is not None and pulse.payload != final.payload:
            pulse_items = min(self.min_pulse_items, chunk_size)
            chunk = [self._copy_item(pulse.payload) for _ in range(pulse_items)]
            chunk.extend(
                self._copy_item(final.payload) for _ in range(chunk_size - pulse_items)
            )
            self._latest_sampled_seq_id = (
                final.seq_id if len(chunk) > pulse_items else pulse.seq_id
            )
            return chunk

        self._latest_sampled_seq_id = final.seq_id
        return [self._copy_item(final.payload) for _ in range(chunk_size)]

    def latest_sampled_seq_id(self) -> int | None:
        return self._latest_sampled_seq_id

    def _drain_pending(self) -> list[ControlStateTransition]:
        transitions = list(self._pending)
        self._pending.clear()
        return transitions

    def _latest_non_default_transition(
        self,
        transitions: Sequence[ControlStateTransition],
    ) -> ControlStateTransition | None:
        for transition in reversed(transitions):
            if transition.payload != self.default_item:
                return transition
        return None

    @staticmethod
    def _copy_item(item: Any) -> Any:
        if isinstance(item, list):
            return copy.deepcopy(item)
        if isinstance(item, dict):
            return copy.deepcopy(item)
        return item
