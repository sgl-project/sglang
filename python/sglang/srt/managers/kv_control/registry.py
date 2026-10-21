# SPDX-License-Identifier: Apache-2.0
"""Bounded owner-thread operation state, independent of cache resources."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

import msgspec

from sglang.srt.managers.kv_control.contracts import (
    MAX_INTEGER,
    Acknowledgement,
    ControlContext,
    ControlError,
    ExecutionUpdate,
    Identifier,
    Limits,
    OperationRef,
    OperationResult,
    OutcomeSummary,
    Status,
    Submission,
    Value,
    validated,
)
from sglang.srt.managers.kv_hints import KvHintAction, KvHintsEnvelope


class FrozenAction(Value):
    action_id: str
    action_type: str
    action_version: str
    payload_json: bytes
    invalid_numbers: bool

    def thaw(self) -> KvHintAction:
        return KvHintAction(
            action_id=self.action_id,
            action_type=self.action_type,
            action_version=self.action_version,
            payload=json.loads(self.payload_json),
        )


class NativeBatch(Value):
    message_id: str
    protocol_version: str
    actions: tuple[FrozenAction, ...]
    fingerprint: bytes


def snapshot_envelope(
    envelope: KvHintsEnvelope, limits: Limits
) -> tuple[NativeBatch, tuple[KvHintAction, ...]]:
    """Snapshot native JSON without inspecting independent consumers' payloads.

    Structural/serialization budgets fail before accepting any native action.
    Numeric payload errors remain per-action failures, including NaN/Infinity
    received through unchecked Python constructors.
    """
    if not isinstance(envelope, KvHintsEnvelope) or type(envelope.actions) is not list:
        raise ControlError("invalid_envelope", "Expected the existing typed envelope")
    native, unhandled = [], []
    for action in envelope.actions:
        if not isinstance(action, KvHintAction) or type(action.action_type) is not str:
            raise ControlError("invalid_envelope", "Malformed action wrapper")
        (native if action.action_type.startswith("sglang.") else unhandled).append(
            action
        )
    if len(native) > limits.max_actions:
        raise ControlError("capacity", "Native action count exceeds limit")
    if not native:
        return NativeBatch(
            message_id="", protocol_version="", actions=(), fingerprint=b""
        ), tuple(unhandled)
    validated(envelope.message_id, Identifier)
    validated(envelope.protocol_version, Identifier)
    nodes = 0
    size = 0

    def inspect(value, depth=0):
        nonlocal nodes, size
        nodes += 1
        size += 1
        if (
            depth > 32
            or nodes > limits.max_json_nodes
            or size > limits.max_envelope_bytes
        ):
            raise ControlError("capacity", "Native JSON complexity exceeds limit")
        if value is None or type(value) is bool:
            return False
        if type(value) is int:
            if value.bit_length() > 128:
                raise ControlError("capacity", "JSON integer encoding exceeds limit")
            return abs(value) > MAX_INTEGER
        if type(value) is float:
            return not math.isfinite(value)
        if type(value) is str:
            size += len(value)
            if size > limits.max_envelope_bytes:
                raise ControlError("capacity", "Native JSON size exceeds limit")
            return False
        if type(value) not in (list, dict):
            raise ControlError("invalid_envelope", "Native payload must be JSON")
        invalid = False
        if type(value) is dict:
            for key, item in value.items():
                if type(key) is not str:
                    raise ControlError("invalid_envelope", "JSON keys must be strings")
                inspect(key, depth + 1)
                invalid |= inspect(item, depth + 1)
        else:
            for item in value:
                invalid |= inspect(item, depth + 1)
        return invalid

    frozen = []
    seen = set()
    for action in native:
        for field in (action.action_id, action.action_type, action.action_version):
            validated(field, Identifier)
        if action.action_id in seen:
            raise ControlError("invalid_envelope", "Duplicate native action_id")
        seen.add(action.action_id)
        invalid = inspect(action.payload)
        try:
            encoded = json.dumps(
                action.payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode()
        except (TypeError, ValueError, OverflowError) as error:
            raise ControlError(
                "invalid_envelope", "Native payload cannot be serialized"
            ) from error
        frozen.append(
            FrozenAction(
                action_id=action.action_id,
                action_type=action.action_type,
                action_version=action.action_version,
                payload_json=encoded,
                invalid_numbers=invalid,
            )
        )
    body = msgspec.json.encode(
        (envelope.protocol_version, envelope.message_id, tuple(frozen))
    )
    if len(body) > limits.max_envelope_bytes:
        raise ControlError("capacity", "Native envelope byte limit exceeded")
    return NativeBatch(
        message_id=envelope.message_id,
        protocol_version=envelope.protocol_version,
        actions=tuple(frozen),
        fingerprint=hashlib.sha256(body).digest(),
    ), tuple(unhandled)


@dataclass
class _Message:
    fingerprint: bytes
    results: dict[str, OperationResult]
    expires_at: float | None = None
    summaries: tuple[OutcomeSummary, ...] = ()
    acknowledged: bool = False


class OperationRegistry:
    """Reserve all record/replay slots before effects; never evict active work.

    All calls belong to one owner thread. Expiry starts when the entire native
    message is terminal, and ACK does not move that deadline. A replay slot is
    reserved on submission, so ACK cannot fail due to replay capacity.
    """

    def __init__(
        self,
        engine_id: str,
        namespace: str,
        *,
        engine_epoch: str | None = None,
        limits: Limits = Limits(),
        clock: Callable[[], float] = time.monotonic,
    ):
        self.engine_id = validated(engine_id, Identifier)
        self.namespace = validated(namespace, Identifier)
        self.engine_epoch = validated(engine_epoch or uuid.uuid4().hex, Identifier)
        self.limits = limits
        self._clock = clock
        self._messages: dict[tuple[str, str | None, str], _Message] = {}
        self._expiry: list[tuple[float, int, tuple[str, str | None, str]]] = []
        self._records = 0
        self._active = 0
        self._sequence = 0

    def validate_context(self, context: ControlContext) -> ControlContext:
        context = validated(context, ControlContext)
        if context.engine_id != self.engine_id:
            raise ControlError("wrong_engine", "Engine identity differs")
        if context.engine_epoch != self.engine_epoch:
            raise ControlError("stale_epoch", "Engine epoch differs")
        if context.namespace != self.namespace:
            raise ControlError("incompatible_namespace", "Cache namespace differs")
        return context

    def _key(self, context, message_id):
        self.validate_context(context)
        validated(message_id, Identifier)
        return context.caller_id, context.binding_id, message_id

    def reserve(self, batch: NativeBatch, context: ControlContext) -> Submission:
        key = self._key(context, batch.message_id)
        self.prune()
        previous = self._messages.get(key)
        if previous is not None:
            if batch.fingerprint != previous.fingerprint:
                raise ControlError(
                    "conflict", "Message ID has different native content"
                )
            return self._submission(previous, replayed=True)
        count = len(batch.actions)
        if not count:
            raise ControlError(
                "invalid_envelope", "Cannot reserve an empty native batch"
            )
        if (
            self._records + count > self.limits.max_records
            or self._active + count > self.limits.max_active
            or len(self._messages) >= self.limits.max_messages
        ):
            raise ControlError("capacity", "Operation or replay capacity exhausted")
        results = {}
        for action in batch.actions:
            self._sequence += 1
            ref = OperationRef(
                context=context,
                message_id=batch.message_id,
                action_id=action.action_id,
                operation_id=uuid.uuid4().hex,
            )
            results[action.action_id] = OperationResult(
                ref=ref,
                action_type=action.action_type,
                action_version=action.action_version,
                sequence=self._sequence,
                update=ExecutionUpdate(status=Status.RECEIVED, quiescent=True),
            )
        message = _Message(batch.fingerprint, results)
        self._messages[key] = message
        self._records += count
        self._active += count
        return self._submission(message, replayed=False)

    @staticmethod
    def _submission(message, *, replayed):
        return Submission(
            replayed=replayed,
            acknowledged=message.acknowledged,
            results=tuple(message.results.values()),
            summaries=message.summaries,
            replay_expires_at=message.expires_at,
        )

    def query(self, context: ControlContext, message_id: str) -> Submission:
        key = self._key(context, message_id)
        self.prune()
        message = self._messages.get(key)
        if message is None:
            raise ControlError("missing", "Unknown or expired native message")
        return self._submission(message, replayed=False)

    def get(self, ref: OperationRef) -> OperationResult:
        ref = validated(ref, OperationRef)
        message = self.query(ref.context, ref.message_id)
        for result in message.results:
            if result.ref.action_id == ref.action_id:
                if result.ref != ref:
                    raise ControlError(
                        "stale_operation", "Operation incarnation differs"
                    )
                return result
        raise ControlError("missing", "Unknown, acknowledged or expired operation")

    def lookup(self, context: ControlContext, message_id: str, *, limit: int = 256):
        """Discover bounded fan-out incarnations without guessing binding IDs."""
        self._key(context, message_id)
        if type(limit) is not int or not 1 <= limit <= self.limits.max_page_size:
            raise ControlError("capacity", "Lookup limit exceeds page budget")
        self.prune()
        matches = [
            self._submission(message, replayed=False)
            for (caller, binding, identity), message in self._messages.items()
            if caller == context.caller_id
            and identity == message_id
            and (context.binding_id is None or binding == context.binding_id)
        ]
        if len(matches) > limit:
            raise ControlError(
                "capacity", "Too many incarnations; query an explicit binding"
            )
        return tuple(matches)

    def update(self, ref: OperationRef, update: ExecutionUpdate) -> OperationResult:
        current = self.get(ref)
        update = validated(update, ExecutionUpdate)
        self._validate_update(update)
        if update == current.update:
            return current
        if current.status.terminal:
            raise ControlError("terminal", "Terminal outcomes are immutable")
        if update.status.terminal and not update.quiescent:
            raise ControlError(
                "busy", "Terminal result requires retired resource obligations"
            )
        if update.status == Status.QUEUED and not update.quiescent:
            raise ControlError(
                "transition", "Queued work cannot own physical resources"
            )
        if (
            update.status == Status.RECEIVED
            or (
                update.status == Status.QUEUED
                and current.status not in (Status.RECEIVED, Status.QUEUED)
            )
            or (
                update.status == Status.ACCEPTED
                and current.status
                not in (Status.RECEIVED, Status.QUEUED, Status.ACCEPTED)
            )
            or (
                update.status == Status.RUNNING
                and current.status not in (Status.ACCEPTED, Status.RUNNING)
            )
            or (
                current.status == Status.CANCELLING
                and not update.status.terminal
                and update.status != Status.CANCELLING
            )
        ):
            raise ControlError("transition", "Invalid operation transition")
        steps = current.update.completed_steps
        if update.completed_steps[: len(steps)] != steps:
            raise ControlError(
                "transition", "Completed stages cannot be removed or reordered"
            )
        self._sequence += 1
        result = msgspec.structs.replace(
            current, sequence=self._sequence, update=update
        )
        key = self._key(ref.context, ref.message_id)
        message = self._messages[key]
        message.results[ref.action_id] = result
        if result.status.terminal:
            self._active -= 1
        if all(item.status.terminal for item in message.results.values()):
            message.expires_at = self._clock() + self.limits.retention_seconds
            heapq.heappush(self._expiry, (message.expires_at, self._sequence, key))
        return result

    def _validate_update(self, update):
        if len(update.reason) > 4096 or len(update.code) > 256:
            raise ControlError("capacity", "Result reason/code exceeds limit")
        if len(update.completed_steps) > 32 or len(set(update.completed_steps)) != len(
            update.completed_steps
        ):
            raise ControlError(
                "capacity", "Completed stages must be bounded and unique"
            )
        if len(update.target_outcomes) > self.limits.max_scope_spans:
            raise ControlError("capacity", "Result coverage exceeds limit")
        for outcome in update.target_outcomes:
            coverage = outcome.coverage
            point = outcome.coverage_kind == "checkpoint"
            if (
                (coverage.start_token != coverage.end_token or coverage.end_token == 0)
                if point
                else coverage.start_token >= coverage.end_token
            ):
                raise ControlError("invalid_payload", "Invalid component coverage")
            if (
                (
                    outcome.coverage_kind == "window"
                    and (outcome.window_size is None or outcome.window_size == 0)
                )
                or (
                    outcome.coverage_kind != "window"
                    and outcome.window_size is not None
                )
                or len(outcome.reason) > 4096
                or len(outcome.code) > 256
                or len(outcome.completed_steps) > 32
            ):
                raise ControlError("invalid_payload", "Invalid target outcome")
        if (
            update.logical_change
            and len(update.logical_change.removed_tasks) > self.limits.max_sessions
        ):
            raise ControlError("capacity", "Task cleanup result exceeds limit")

    def acknowledge(self, context: ControlContext, message_id: str) -> Acknowledgement:
        key = self._key(context, message_id)
        self.query(context, message_id)
        message = self._messages[key]
        if message.expires_at is None:
            raise ControlError("busy", "Message still has active operations")
        if not message.acknowledged:
            message.summaries = tuple(
                OutcomeSummary(
                    ref=r.ref,
                    status=r.status,
                    code=r.update.code,
                    reason=r.update.reason,
                )
                for r in message.results.values()
            )
            self._records -= len(message.results)
            message.results.clear()
            message.acknowledged = True
        return Acknowledgement(
            summaries=message.summaries, replay_expires_at=message.expires_at
        )

    def prune(self) -> None:
        now = self._clock()
        while self._expiry and self._expiry[0][0] <= now:
            _, _, key = heapq.heappop(self._expiry)
            message = self._messages.pop(key)
            self._records -= len(message.results)

    def capacity(self) -> dict[str, int]:
        self.prune()
        return {
            "active": self._active,
            "records": self._records,
            "messages": len(self._messages),
        }
