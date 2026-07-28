# SPDX-License-Identifier: Apache-2.0

"""Realtime control signal primitives.

This module owns the small, model-agnostic data structures that turn external
realtime control inputs into chunk-sized payloads consumed by model adapters.
It intentionally does not know about cameras, LingBot, or SANA-WM semantics;
callers provide validation/normalization functions and decide how sampled
payloads map to request ``condition_inputs``.

There are two control modes:

* Script mode is a finite per-frame timeline, for example
  ``[["w"], ["w"], [], ...]``. It is already expanded by the caller, so
  ``ControlScriptQueue`` consumes it in order and pads the tail with a neutral
  default item when configured. Script mode is useful for tests, presets, and
  deterministic replay.
* State mode is a level-triggered stream of transitions, for example "these
  keys are currently held". ``ControlStateQueue`` keeps the latest state and
  samples a stable chunk from it; a short non-default pulse can be preserved
  even when press/release transitions arrive between two render chunks. State
  mode is the natural shape for live keyboard/gamepad controls.

``ControlSignalQueue`` is the lower-level FIFO for discrete signals. It is used
directly for one-shot controls such as prompt updates, and wrapped by
``ControlScriptQueue`` for finite timelines. ``ControlStateQueue`` is separate
because held controls need stateful sampling rather than FIFO draining.
"""

from __future__ import annotations

import copy
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_MISSING = object()


@dataclass(frozen=True)
class ControlSignal:
    """a discrete, minimal input signal for a specific control kind.

    Generated from script-mode inputs such as receive_camera_action_script
    Consumed by chunk samplers such as sample_camera_actions

    """

    # camera_actions, prompt
    kind: str
    # two formats:
    # 1. script mode: a sequence of flatten actions (e.g., [["w"], ["w"], ["a"], []])
    # 2. state mode: a sequence of state changes (e.g., "actions": ["w"], "client_ts_ms": 1000)
    payload: Any
    # the timestep of this signal
    timestamp_ms: int | None = None
    seq_id: int | None = None


@dataclass(frozen=True)
class ControlStateTransition:
    payload: Any
    timestamp_ms: int | None = None
    seq_id: int | None = None


@dataclass(frozen=True)
class ControlSignalSamplingParams:
    chunk_size: int
    default_item: Any = _MISSING
    repeat_last: bool = True
    repeat_last_across_empty_chunks: bool = False


@dataclass(frozen=True)
class ParsedControlEventPayload:
    mode: str
    payload: Any


ControlStatePayloadNormalizer = Callable[[list[Any]], Any]
ControlScriptPayloadValidator = Callable[[Any], Any]


def parse_control_event_payload(
    payload: Any,
    *,
    event_id: int | None,
    kind: str,
    normalize_state_payload: ControlStatePayloadNormalizer,
    validate_script_payload: ControlScriptPayloadValidator,
) -> ParsedControlEventPayload:
    """parse external control event from endpoint"""
    if isinstance(payload, dict) and payload.get("mode") == "state":
        return ParsedControlEventPayload(
            mode="state",
            payload=_control_state_transitions_from_event_payload(
                payload,
                event_id=event_id,
                kind=kind,
                normalize_state_payload=normalize_state_payload,
            ),
        )
    return ParsedControlEventPayload(
        mode="script",
        payload=validate_script_payload(payload),
    )


def _control_state_transitions_from_event_payload(
    payload: dict[str, Any],
    *,
    event_id: int | None,
    kind: str,
    normalize_state_payload: ControlStatePayloadNormalizer,
) -> list[ControlStateTransition]:
    transitions = payload.get("transitions")
    if not isinstance(transitions, list):
        raise ValueError(f"{kind} state payload requires transitions")
    result = []
    for transition in transitions:
        if not isinstance(transition, dict):
            raise ValueError(f"{kind} transition must be a map")
        actions = transition.get("actions")
        if not isinstance(actions, list):
            raise ValueError(f"{kind} transition actions must be a list")
        timestamp_ms = transition.get("client_ts_ms")
        if timestamp_ms is not None:
            timestamp_ms = int(timestamp_ms)
        result.append(
            ControlStateTransition(
                payload=normalize_state_payload(actions),
                seq_id=event_id,
                timestamp_ms=timestamp_ms,
            )
        )
    return result


class ControlSignalQueue:
    """FIFO storage for discrete realtime control signals

    Script-mode controls and one-shot signals are already expressed as discrete
    payloads, so sampling only consumes queued signals and applies the requested
    padding strategy.
    """

    def __init__(
        self,
        max_events: int | Mapping[str, int] = 512,
    ) -> None:
        self._max_events = max_events
        # [control_kind, deque of control signals]
        self._signals: dict[str, deque[ControlSignal]] = {}
        self._last_payloads: dict[str, Any] = {}
        self._last_sampled_seq_ids: dict[str, int | None] = {}
        self._seen_kinds: set[str] = set()

    def push(
        self,
        kind: str,
        payload: Any,
        *,
        event_id: int | None = None,
        timestamp_ms: int | None = None,
        expand_payload: bool = True,
    ) -> None:
        queue = self._queue_for(kind)
        for signal in self._iter_signals(
            kind,
            payload,
            event_id=event_id,
            timestamp_ms=timestamp_ms,
            expand_payload=expand_payload,
        ):
            queue.append(signal)
        self._seen_kinds.add(kind)

    def replace(
        self,
        kind: str,
        payload: Any,
        *,
        event_id: int | None = None,
        timestamp_ms: int | None = None,
        expand_payload: bool = True,
    ) -> None:
        self.clear_kind(kind)
        self.push(
            kind,
            payload,
            event_id=event_id,
            timestamp_ms=timestamp_ms,
            expand_payload=expand_payload,
        )

    def pop_latest(self, kind: str) -> Any | None:
        queue = self._signals.get(kind)
        if not queue:
            return None
        signal = queue.pop()
        latest_payload = signal.payload
        self._last_payloads[kind] = signal.payload
        self._last_sampled_seq_ids[kind] = signal.seq_id
        queue.clear()
        self._seen_kinds.add(kind)
        return latest_payload

    def has_events(self, kind: str) -> bool:
        queue = self._signals.get(kind)
        return bool(queue)

    def sample_chunk(
        self,
        kind: str,
        params: ControlSignalSamplingParams,
    ) -> list[Any] | None:
        """sample queued signals for one realtime chunk"""
        if params.chunk_size <= 0:
            return None

        chunk: list[Any] = []
        queue = self._signals.get(kind)
        self._drain_signals(kind, queue, chunk, params.chunk_size)

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
        self._signals.clear()
        self._last_payloads.clear()
        self._last_sampled_seq_ids.clear()
        self._seen_kinds.clear()

    def clear_kind(self, kind: str) -> None:
        self._signals.pop(kind, None)
        self._last_payloads.pop(kind, None)
        self._last_sampled_seq_ids.pop(kind, None)
        self._seen_kinds.discard(kind)

    def last_sampled_seq_id(self, kind: str) -> int | None:
        return self._last_sampled_seq_ids.get(kind)

    def _queue_for(self, kind: str) -> deque[ControlSignal]:
        queue = self._signals.get(kind)
        if queue is None:
            if isinstance(self._max_events, Mapping):
                maxlen = self._max_events.get(kind, 512)
            else:
                maxlen = self._max_events
            queue = deque(maxlen=maxlen)
            self._signals[kind] = queue
        return queue

    def _iter_signals(
        self,
        kind: str,
        payload: Any,
        *,
        event_id: int | None,
        timestamp_ms: int | None,
        expand_payload: bool,
    ):
        items = (
            payload
            if self._should_expand_payload(payload, expand_payload)
            else (payload,)
        )
        for item in items:
            if isinstance(item, ControlSignal):
                if item.kind != kind:
                    raise ValueError(
                        "control signal kind "
                        f"{item.kind!r} does not match queue kind {kind!r}"
                    )
                yield item
            else:
                yield ControlSignal(
                    kind=kind,
                    payload=item,
                    timestamp_ms=timestamp_ms,
                    seq_id=event_id,
                )

    @staticmethod
    def _should_expand_payload(payload: Any, expand_payload: bool) -> bool:
        return (
            expand_payload
            and isinstance(payload, Sequence)
            and not isinstance(payload, (str, bytes, bytearray))
        )

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


class ControlScriptQueue:
    """Script-mode queue for finite per-frame control timelines."""

    def __init__(
        self,
        kind: str,
        *,
        max_events: int = 512,
        default_item: Any = _MISSING,
    ) -> None:
        self.kind = kind
        self.default_item = default_item
        self._signals = ControlSignalQueue(max_events={kind: max_events})

    def clear(self) -> None:
        self._signals.clear_kind(self.kind)

    def push_script(
        self,
        script: Sequence[Any],
        *,
        event_id: int | None = None,
    ) -> None:
        self.clear()
        self._signals.push(self.kind, script, event_id=event_id)

    def has_script(self) -> bool:
        return self._signals.has_events(self.kind)

    def sample_script(self, chunk_size: int) -> list[Any]:
        chunk = self._signals.sample_chunk(
            self.kind,
            ControlSignalSamplingParams(
                chunk_size=chunk_size,
                default_item=self.default_item,
                repeat_last=False,
            ),
        )
        if chunk is None:
            chunk = []
        while len(chunk) < chunk_size and self.default_item is not _MISSING:
            chunk.append(self._copy_item(self.default_item))
        return chunk

    def last_sampled_seq_id(self) -> int | None:
        return self._signals.last_sampled_seq_id(self.kind)

    @staticmethod
    def _copy_item(item: Any) -> Any:
        if isinstance(item, list):
            return copy.deepcopy(item)
        if isinstance(item, dict):
            return copy.deepcopy(item)
        return item


class ControlStateQueue:
    """State-mode sampler for level-triggered realtime controls."""

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
