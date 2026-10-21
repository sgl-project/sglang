# SPDX-License-Identifier: Apache-2.0
"""Native control values. Importing these contracts has no serving side effects."""

from __future__ import annotations

import math
from enum import Enum
from typing import Annotated, Literal, TypeAlias

import msgspec

from sglang.srt.managers.kv_hints import KvHintAction

PROTOCOL_VERSION = "0.1"
ACTION_VERSION = "1.0"
MAX_INTEGER = (1 << 53) - 1
Identifier: TypeAlias = Annotated[str, msgspec.Meta(min_length=1, max_length=256)]
Position: TypeAlias = Annotated[int, msgspec.Meta(ge=0, le=MAX_INTEGER)]
Priority: TypeAlias = Annotated[int, msgspec.Meta(ge=-(1 << 31), le=(1 << 31) - 1)]
Seconds: TypeAlias = Annotated[float, msgspec.Meta(gt=0, le=86400 * 365)]


class ControlError(ValueError):
    def __init__(self, code: str, reason: str):
        super().__init__(reason)
        self.code = code


class Value(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    """Transport values use immutable children as well as frozen fields."""


class Limits(Value):
    max_actions: int = 128
    max_envelope_bytes: int = 1 << 20
    max_json_nodes: int = 100000
    max_active: int = 1024
    max_records: int = 4096
    max_messages: int = 8192
    retention_seconds: float = 300.0
    max_sessions: int = 4096
    max_continuations: int = 16384
    max_index_spans: int = 65536
    max_index_tokens: int = 1048576
    max_scope_spans: int = 4096
    max_scope_tokens: int = 65536
    max_branches: int = 256
    max_page_size: int = 256

    def __post_init__(self):
        for name in self.__struct_fields__:
            value = getattr(self, name)
            if name == "retention_seconds":
                if (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError("retention_seconds must be finite and positive")
            elif type(value) is not int or not 1 <= value <= MAX_INTEGER:
                raise ValueError(f"{name} must be a positive bounded integer")


class ControlContext(Value):
    engine_id: Identifier
    engine_epoch: Identifier
    namespace: Identifier
    caller_id: Identifier
    # A unique request *incarnation*, including batch/sampling fan-out identity.
    binding_id: Identifier | None = None


class TokenRange(Value):
    start_token: Position
    end_token: Position
    component: Identifier | None = None


class ContentSpan(Value):
    """One indivisible logical descriptor, never an allocator address.

    component identifies a FULL/SWA/Mamba component in the namespace's layout.
    dependency_start describes earlier state required to reuse this descriptor;
    it does not assert that such state is known or resident.
    """

    content_id: Identifier
    component: Identifier
    start_token: Position
    end_token: Position
    dependency_start: Position = 0
    coverage_kind: Literal["tokens", "window", "checkpoint"] = "tokens"
    window_size: Position | None = None


class CacheCompatibility(Value):
    cache_salt: Identifier | None = None
    extra_key: Identifier | None = None
    layout_id: Identifier | None = None
    is_multimodal: bool = False


def cache_compatibility(target) -> CacheCompatibility:
    return CacheCompatibility(
        **{name: getattr(target, name) for name in CacheCompatibility.__struct_fields__}
    )


class ContentTarget(CacheCompatibility, tag="content", tag_field="kind", kw_only=True):
    namespace: Identifier
    token_count: Position
    components: Annotated[
        tuple[Identifier, ...], msgspec.Meta(min_length=1, max_length=64)
    ]
    spans: Annotated[tuple[ContentSpan, ...], msgspec.Meta(max_length=4096)]
    start_token: Position = 0
    end_token: Position | None = None
    allow_clip: bool = False


class TokensTarget(CacheCompatibility, tag="tokens", tag_field="kind", kw_only=True):
    namespace: Identifier
    token_ids: Annotated[
        tuple[Position, ...], msgspec.Meta(min_length=1, max_length=65536)
    ]
    start_token: Position = 0
    end_token: Position | None = None
    allow_clip: bool = False


class SessionTarget(Value, tag="session", tag_field="kind"):
    namespace: Identifier
    session_id: Identifier
    continuation_id: Identifier | None = None
    expected_revision: Position | None = None
    start_token: Position = 0
    end_token: Position | None = None
    allow_clip: bool = False


LogicalTarget: TypeAlias = ContentTarget | TokensTarget | SessionTarget
Target: TypeAlias = LogicalTarget | Literal["request_cached"]


class Location(Value):
    tier: Literal["gpu", "host", "storage"]
    engine_id: Identifier | None = None
    backend: Identifier | None = None


class SetPriority(Value):
    target: Target
    location: Location
    claim_id: Identifier
    revision: Position
    value: Priority | None
    ttl_seconds: Seconds | None = None


class CopyTo(Value):
    target: Target
    src: Location
    dst: Location


class MoveTo(Value):
    target: Target
    src: Location
    dst: Location


Primitive: TypeAlias = SetPriority | CopyTo | MoveTo


class TaskReference(Value):
    session_id: Identifier
    expected_revision: Position


class ParentPrefix(Value):
    session_id: Identifier
    continuation_id: Identifier
    expected_revision: Position
    end_token: Position


class Associate(Value):
    session_id: Identifier
    continuation_id: Identifier
    expected_revision: Position
    target: ContentTarget | TokensTarget
    task_parent: TaskReference | None = None
    parent_prefix: ParentPrefix | None = None


class FinishTask(Value):
    session_id: Identifier
    expected_revision: Position


class Signal(Value):
    target: LogicalTarget
    signal: Literal["tool_wait", "resume", "importance", "resume_prediction"]
    importance: Priority | None = None
    resume_after_seconds: Seconds | None = None


Payload: TypeAlias = Primitive | Associate | FinishTask | Signal


class Status(str, Enum):
    RECEIVED = "received"
    QUEUED = "queued"
    ACCEPTED = "accepted"
    RUNNING = "running"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    PARTIAL = "partial"
    DEFERRED = "deferred"
    MISSING = "missing"
    UNSUPPORTED = "unsupported"
    REJECTED = "rejected"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNCONSUMED = "unconsumed"

    @property
    def terminal(self) -> bool:
        return self not in (
            Status.RECEIVED,
            Status.QUEUED,
            Status.ACCEPTED,
            Status.RUNNING,
            Status.CANCELLING,
        )


class OperationRef(Value):
    context: ControlContext
    message_id: Identifier
    action_id: Identifier
    operation_id: Identifier


class TargetOutcome(Value):
    coverage: TokenRange
    status: Status
    # Position in this operation's frozen logical branches, not a physical ID.
    branch_index: Position = 0
    content_id: Identifier | None = None
    location: Location | None = None
    completed_steps: tuple[Identifier, ...] = ()
    code: str = ""
    reason: str = ""
    # Checkpoints are points [end, end], not per-token physical coverage.
    coverage_kind: Literal["tokens", "window", "checkpoint"] = "tokens"
    window_size: Position | None = None


class TaskIdentity(Value):
    session_id: Identifier
    incarnation: Position


class LogicalChange(Value):
    index_generation: Position
    session_revision: Position | None = None
    continuation_revision: Position | None = None
    incarnation: Position | None = None
    removed_tasks: tuple[TaskIdentity, ...] = ()


class ExecutionUpdate(Value):
    status: Status
    # True certifies all access/retirement obligations for this operation ended.
    # Shared jobs may still run under their own independently owned holds.
    quiescent: bool
    code: str = ""
    reason: str = ""
    completed_steps: tuple[Identifier, ...] = ()
    target_outcomes: tuple[TargetOutcome, ...] = ()
    logical_change: LogicalChange | None = None


class OperationResult(Value):
    ref: OperationRef
    action_type: Identifier
    action_version: Identifier
    sequence: Position
    update: ExecutionUpdate

    @property
    def status(self) -> Status:
        return self.update.status


class OutcomeSummary(Value):
    ref: OperationRef
    status: Status
    code: str
    reason: str


class Submission(Value):
    replayed: bool
    acknowledged: bool = False
    results: tuple[OperationResult, ...] = ()
    summaries: tuple[OutcomeSummary, ...] = ()
    replay_expires_at: float | None = None
    # Borrowed, opaque transport objects. Never stored in the native registry.
    unhandled_actions: tuple[KvHintAction, ...] = ()


class Acknowledgement(Value):
    summaries: tuple[OutcomeSummary, ...]
    replay_expires_at: float


def validated(value, schema):
    """Validate even directly constructed msgspec values; constructors are unchecked."""
    try:
        return msgspec.convert(msgspec.to_builtins(value), type=schema, strict=True)
    except (msgspec.ValidationError, TypeError, ValueError, OverflowError) as error:
        raise ControlError("invalid_payload", str(error)[:4096]) from error


def validate_target(target: LogicalTarget) -> None:
    if target.end_token is not None and target.end_token <= target.start_token:
        raise ControlError(
            "invalid_payload", "Token ranges must be nonempty and half-open"
        )
    if isinstance(target, ContentTarget):
        if not target.token_count or len(set(target.components)) != len(
            target.components
        ):
            raise ControlError(
                "invalid_payload", "Content needs a length and unique components"
            )
        by_component: dict[str, list[ContentSpan]] = {}
        for span in target.spans:
            if (
                not 0
                <= span.dependency_start
                <= span.start_token
                <= span.end_token
                <= target.token_count
            ):
                raise ControlError(
                    "invalid_payload", "Invalid content or dependency range"
                )
            if (span.coverage_kind == "checkpoint") != (
                span.start_token == span.end_token
            ) or span.end_token == 0:
                raise ControlError(
                    "invalid_payload",
                    "Checkpoint descriptors are nonzero points; token spans are nonempty",
                )
            if (span.coverage_kind == "window" and not span.window_size) or (
                span.coverage_kind != "window" and span.window_size is not None
            ):
                raise ControlError(
                    "invalid_payload",
                    "Window descriptors require a positive window size",
                )
            if span.component not in target.components:
                raise ControlError("invalid_payload", "Undeclared content component")
            by_component.setdefault(span.component, []).append(span)
        for spans in by_component.values():
            ordered = sorted(spans, key=lambda span: span.start_token)
            if len({(s.coverage_kind, s.window_size) for s in spans}) > 1 or any(
                a.end_token > b.start_token or a.start_token == b.start_token
                for a, b in zip(ordered, ordered[1:])
            ):
                raise ControlError(
                    "invalid_payload", "Overlapping component descriptors"
                )


def parse_action(action: KvHintAction) -> Payload:
    """Interpret only native names; generic forwarding never calls this parser."""
    schemas = {
        "sglang.set_priority": SetPriority,
        "sglang.copy_to": CopyTo,
        "sglang.move_to": MoveTo,
        "sglang.associate": Associate,
        "sglang.finish_task": FinishTask,
        "sglang.signal": Signal,
    }
    schema = schemas.get(action.action_type)
    if schema is None or action.action_version != ACTION_VERSION:
        raise ControlError("unsupported", "Unsupported native action or action version")
    payload = validated(action.payload, schema)
    if hasattr(payload, "target") and payload.target != "request_cached":
        validate_target(payload.target)
    if isinstance(payload, SetPriority):
        if payload.value is None:
            if payload.ttl_seconds is not None:
                raise ControlError("invalid_payload", "Clearing a claim must omit TTL")
        elif payload.ttl_seconds is None or not math.isfinite(payload.ttl_seconds):
            raise ControlError(
                "invalid_payload", "Priority requires a finite positive TTL"
            )
    if isinstance(payload, Signal) and payload.resume_after_seconds is not None:
        if not math.isfinite(payload.resume_after_seconds):
            raise ControlError("invalid_payload", "Prediction must be finite")
    if isinstance(payload, Associate):
        if payload.target.start_token != 0 or payload.target.end_token is not None:
            raise ControlError(
                "invalid_payload", "Association describes a complete logical prefix"
            )
        if payload.target.allow_clip:
            raise ControlError(
                "invalid_payload", "Association cannot clip content identity"
            )
    return payload
