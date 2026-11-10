# SPDX-License-Identifier: Apache-2.0
"""Native control values. Importing these contracts has no serving side effects."""

from __future__ import annotations

import math
import re
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


class ResolvedTarget(Value):
    engine_epoch: str
    content: TokensTarget
    requested: TokenRange

    @property
    def namespace(self):
        return self.content.namespace

    @property
    def allow_clip(self):
        return self.content.allow_clip


def resolve_prefix(target, context, limits):
    if not isinstance(target, TokensTarget):
        raise ControlError(
            "unsupported", "Engine control requires a concrete token prefix"
        )
    target = validated(target, TokensTarget)
    validate_target(target)
    if target.namespace != context.namespace:
        raise ControlError("incompatible_namespace", "Target namespace differs")
    if len(target.token_ids) > limits.max_scope_tokens:
        raise ControlError("capacity", "Resolved prefix exceeds the token budget")
    end = len(target.token_ids) if target.end_token is None else target.end_token
    if not target.start_token < end <= len(target.token_ids):
        raise ControlError("missing", "Requested range exceeds the supplied prefix")
    return ResolvedTarget(
        engine_epoch=context.engine_epoch,
        content=target,
        requested=TokenRange(start_token=target.start_token, end_token=end),
    )


class Location(Value, omit_defaults=True):
    tier: Literal["gpu", "host", "storage"]
    engine_id: Identifier | None = None
    backend: Identifier | None = None
    backend_namespace: Identifier | None = None


# Event hashes are signed i64 decimal strings, never JSON numbers or storage keys.
HASH_SCHEME = "sglang.sha256.u32le.i64.v1"
RESULT_VERSION = "1.0"
Hash64: TypeAlias = Annotated[
    str, msgspec.Meta(pattern=r"^(0|-[1-9][0-9]*|[1-9][0-9]*)$", max_length=20)
]
TokenId: TypeAlias = Annotated[int, msgspec.Meta(ge=0, le=(1 << 32) - 1)]


class PrefixCompatibility(Value):
    namespace: Identifier
    cache_salt: Annotated[str, msgspec.Meta(max_length=256)] | None
    layout_id: Identifier
    page_size: Annotated[int, msgspec.Meta(ge=1, le=256)]
    components: Annotated[
        tuple[Identifier, ...], msgspec.Meta(min_length=1, max_length=64)
    ]
    hash_scheme: Identifier


class PrefixBlock(Value):
    block_hash: Hash64
    component: Identifier
    start_token: Position
    end_token: Position
    dependency_start: Position = 0


class PrefixTarget(Value, tag="prefix", tag_field="kind"):
    compatibility: PrefixCompatibility
    start_token: Position
    end_token: Position
    # Material always begins at token zero, including all ancestor dependencies.
    token_ids: (
        Annotated[tuple[TokenId, ...], msgspec.Meta(min_length=1, max_length=65536)]
        | None
    ) = None
    blocks: Annotated[tuple[PrefixBlock, ...], msgspec.Meta(max_length=4096)] = ()

    @property
    def namespace(self):
        return self.compatibility.namespace


class PublicLocation(Value):
    tier: Literal["gpu", "host", "storage"]
    engine_id: Identifier | None = None
    backend: Identifier | None = None
    backend_namespace: Identifier | None = None


PublicTarget: TypeAlias = Target | PrefixTarget


class Prefetch(Value):
    target: PublicTarget
    source: PublicLocation
    destination: PublicLocation
    allow_partial: bool = False
    timeout_ms: Annotated[int, msgspec.Meta(ge=1, le=86400000)] | None = None


class Demote(Prefetch):
    pass


class Pin(Value):
    mode: Literal["acquire", "renew", "release"]
    target: PublicTarget
    location: PublicLocation
    claim_id: Identifier
    revision: Position
    ttl_seconds: Seconds | None = None
    priority: Priority | None = None
    required_protection: (
        Literal["eviction_exclusion", "eviction_priority", "backend_soft_pin"] | None
    ) = None
    # Explicit demand ownership, independent of the selected content/session.
    owner_task: TaskIdentity | None = None


class ClaimState(Value):
    """Current owner-confirmed protection, independent of operation completion."""

    caller_id: Identifier
    task_incarnation: Identifier
    claim_id: Identifier
    revision: Position
    state: Literal["active", "released", "expired", "invalidated", "unknown"]
    owner_epoch: Identifier
    # A release tombstone fences late renewals without asserting live protection.
    protection_version: Literal["1.0"] = "1.0"
    tombstone: bool = False
    effective_protection: (
        Literal["eviction_exclusion", "eviction_priority", "backend_soft_pin"] | None
    ) = None
    remaining_ttl: Seconds | None = None
    priority: Priority | None = None
    owner_task: TaskIdentity | None = None
    actual_scope: tuple[TargetOutcome, ...] = ()
    location: PublicLocation | None = None


class CopyTo(Value):
    target: Target
    src: Location
    dst: Location


class MoveTo(Value):
    target: Target
    src: Location
    dst: Location


Primitive: TypeAlias = CopyTo | MoveTo


Payload: TypeAlias = Prefetch | Demote | Pin


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
    # Owner-created children have independent results/cancellation lifetimes.
    derived_operations: tuple[OperationRef, ...] = ()


class PublicResult(Value, kw_only=True):
    """Cumulative observations, scoped by branch/component, never a residency lease."""

    result_version: Literal["1.0"] = RESULT_VERSION
    requested_scope: tuple[TargetOutcome, ...] = ()
    accepted_scope: tuple[TargetOutcome, ...] = ()
    ready_scope: tuple[TargetOutcome, ...] = ()
    ready_location: PublicLocation | None = None
    preserved_scope: tuple[TargetOutcome, ...] = ()
    reclaimed_scope: tuple[TargetOutcome, ...] = ()
    missing_scope: tuple[TargetOutcome, ...] = ()
    blocked_scope: tuple[TargetOutcome, ...] = ()
    owner_epoch: Identifier
    state_version: Position
    outcome_sequence: Position
    cache_generation: Position | None = None
    source_observed_at: float | None = None
    claim: ClaimState | None = None


class CacheWatermark(Value):
    source_id: Identifier
    source_epoch: Identifier
    generation: Position
    version: Position


class OperationResult(Value, omit_defaults=True):
    ref: OperationRef
    action_type: Identifier
    action_version: Identifier
    sequence: Position
    update: ExecutionUpdate
    cache_watermark: CacheWatermark | None = None
    public_result: PublicResult | None = None

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


def validate_target(target: LogicalTarget | PrefixTarget) -> None:
    if isinstance(target, PrefixTarget):
        validate_prefix(target)
        return
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


def validate_prefix(target: PrefixTarget) -> None:
    c = target.compatibility
    if len(set(c.components)) != len(c.components):
        raise ControlError("invalid_payload", "Components must be unique")
    if not target.start_token < target.end_token or any(
        x % c.page_size for x in (target.start_token, target.end_token)
    ):
        raise ControlError(
            "invalid_payload", "Prefix boundaries must be nonempty and page aligned"
        )
    if target.token_ids is not None and len(target.token_ids) != target.end_token:
        raise ControlError(
            "invalid_payload", "Token material must cover exactly [0, end_token)"
        )
    previous = {}
    for block in target.blocks:
        if not re.fullmatch(r"0|-[1-9][0-9]*|[1-9][0-9]*", block.block_hash) or not -(
            1 << 63
        ) <= int(block.block_hash) < (1 << 63):
            raise ControlError(
                "invalid_payload", "Block hash must be a canonical decimal i64 string"
            )
        if (
            block.component not in c.components
            or not 0
            <= block.dependency_start
            <= block.start_token
            < block.end_token
            <= target.end_token
            or block.end_token - block.start_token != c.page_size
            or block.dependency_start % c.page_size
            or block.start_token % c.page_size
            or block.start_token < previous.get(block.component, 0)
        ):
            raise ControlError(
                "invalid_payload", "Invalid ordered block/dependency descriptor"
            )
        previous[block.component] = block.end_token


def validate_public(payload):
    if isinstance(payload, Pin):
        if (payload.mode == "release") != (
            payload.ttl_seconds is None and payload.priority is None
        ):
            raise ControlError(
                "invalid_payload",
                "Acquire/renew require TTL or priority; release omits both",
            )
        if payload.ttl_seconds is not None and not math.isfinite(payload.ttl_seconds):
            raise ControlError("invalid_payload", "TTL must be finite")
        locations = (payload.location,)
    else:
        locations = (payload.source, payload.destination)
        ranks = {"gpu": 0, "host": 1, "storage": 2}
        src, dst = ranks[payload.source.tier], ranks[payload.destination.tier]
        if (isinstance(payload, Demote) and src >= dst) or (
            not isinstance(payload, Demote) and src <= dst
        ):
            raise ControlError("invalid_payload", "Invalid public transfer direction")
    for location in locations:
        if location.tier == "storage":
            if location.engine_id is not None:
                raise ControlError(
                    "invalid_payload", "Storage locations cannot name an engine"
                )
        elif location.backend is not None or location.backend_namespace is not None:
            raise ControlError("invalid_payload", "Engine tiers cannot name a backend")


def parse_action(action: KvHintAction) -> Payload:
    """Interpret only native names; generic forwarding never calls this parser."""
    schemas = {
        "sglang.prefetch": Prefetch,
        "sglang.demote": Demote,
        "sglang.pin": Pin,
    }
    schema = schemas.get(action.action_type)
    if schema is None or action.action_version != ACTION_VERSION:
        raise ControlError("unsupported", "Unsupported native action or action version")
    payload = validated(action.payload, schema)
    if payload.target != "request_cached":
        validate_target(payload.target)
    validate_public(payload)
    return payload
