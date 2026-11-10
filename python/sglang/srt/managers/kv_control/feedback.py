# SPDX-License-Identifier: Apache-2.0
"""Bounded, owner-thread polling feedback; never sent on the KV location socket."""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from collections import deque
from typing import Annotated, Literal

import msgspec
from sglang.srt.managers.kv_control.contracts import (
    CacheWatermark,
    ControlContext,
    ControlError,
    Identifier,
    OperationResult,
    Position,
    Status,
    TokensTarget,
    Value,
)


class FeedbackLimits(Value):
    max_events: Annotated[int, msgspec.Meta(ge=1, le=65536)] = 4096
    max_bytes: Annotated[int, msgspec.Meta(ge=1024, le=1 << 28)] = 8 << 20
    max_event_bytes: Annotated[int, msgspec.Meta(ge=1024, le=1 << 24)] = 1 << 20
    max_requests: Annotated[int, msgspec.Meta(ge=1, le=65536)] = 4096


class FeedbackPublisherConfig(Value):
    # Always bind: accidentally connecting this channel to a legacy PUB/SUB
    # endpoint would expose unknown control events to old subscribers.
    endpoint: Identifier
    replay_endpoint: Identifier | None = None
    buffer_steps: Annotated[int, msgspec.Meta(ge=1, le=4096)] = 256
    max_queue_size: Annotated[int, msgspec.Meta(ge=1, le=1024)] = 32
    hwm: Annotated[int, msgspec.Meta(ge=1, le=4096)] = 256


class ControlEventBatch(msgspec.Struct, kw_only=True):
    source_id: str
    source_epoch: str
    engine_id: str
    engine_epoch: str
    namespace: str
    events: tuple[ControlEvent, ...]
    earliest_sequence: int
    latest_sequence: int
    gap: bool
    version: Literal["1.0"] = "1.0"
    attn_dp_rank: int | None = None


class FeedbackCursor(Value):
    source_id: Identifier
    source_epoch: Identifier
    sequence: Position


class RequestAssociation(Value):
    """Explicit application provenance, not authority to modify Engine sessions."""

    request_id: Identifier | None = None
    session_id: Identifier | None = None
    session_incarnation: Identifier | None = None
    branch_id: Identifier | None = None
    branch_revision: Position | None = None


class RequestUse(Value):
    context: ControlContext
    request_id: str
    association: RequestAssociation | None
    association_state: Literal["provided", "unknown"]
    cached_tokens: Position
    device_tokens: Position | None
    host_tokens: Position | None
    storage_tokens: Position | None
    tier_accounting: Literal["measured", "unknown"]
    checkpointed: bool
    checkpointed_tokens: Position
    aborted: bool
    committed_prefix: TokensTarget | None
    # Completion is a request fact; it never implies task/session completion.
    provenance: Literal["engine_request_accounting"] = "engine_request_accounting"


class ControlEvent(Value):
    sequence: Position
    kind: Literal["operation", "association", "request_use", "claim", "invalidation"]
    caller_id: Identifier | None
    cache_watermark: CacheWatermark
    # Private JSON snapshot: journal size includes every byte of this payload.
    payload: dict


class FeedbackPage(Value):
    source_id: Identifier
    source_epoch: Identifier
    engine_id: Identifier
    engine_epoch: Identifier
    namespace: Identifier
    items: tuple[ControlEvent, ...]
    next_cursor: FeedbackCursor
    gap: bool
    earliest_sequence: Position
    latest_sequence: Position
    cache_watermark: CacheWatermark
    recovery: Literal["query_known_scopes"] = "query_known_scopes"


class FeedbackJournal:
    def __init__(
        self,
        *,
        engine_id,
        engine_epoch,
        namespace,
        limits,
        watermark,
        publisher_config=None,
        clock=time.monotonic,
    ):
        self.engine_id, self.engine_epoch, self.namespace = (
            engine_id,
            engine_epoch,
            namespace,
        )
        self.source_id, self.source_epoch = (
            "control:" + hashlib.sha256(engine_id.encode()).hexdigest(),
            uuid.uuid4().hex,
        )
        self._thread = threading.get_ident()
        self.limits, self.watermark = limits, watermark
        self._events: deque[tuple[ControlEvent, int]] = deque()
        self._bytes, self._sequence = 0, 0
        self._requests: dict[
            tuple[str, str], tuple[str, RequestAssociation | None]
        ] = {}
        self.publisher = None
        self._published = 0
        self.publisher_config = publisher_config
        self._clock = clock
        self._publish_retry_at = 0.0
        if publisher_config is not None:
            from sglang.srt.disaggregation.kv_events import EventPublisherFactory

            if publisher_config.endpoint == publisher_config.replay_endpoint:
                raise ControlError(
                    "invalid_payload", "Control PUB and replay endpoints must differ"
                )
            config = {
                "publisher": "zmq",
                "bind": True,
                "topic": "sglang.control.v1",
                **msgspec.to_builtins(publisher_config),
            }
            self.publisher = EventPublisherFactory.create(
                msgspec.json.encode(config).decode()
            )

    @property
    def has_outbox(self):
        return self.publisher is not None and self._published < self._sequence

    @property
    def next_deadline(self):
        return max(self._clock(), self._publish_retry_at) if self.has_outbox else None

    def flush(self, *, limit=64):
        self._owner()
        if not self.has_outbox or self._clock() < self._publish_retry_at:
            return
        earliest = self._events[0][0].sequence if self._events else self._sequence + 1
        selected, size = [], 0
        for event, event_size in self._events:
            if event.sequence <= self._published:
                continue
            if selected and (
                len(selected) >= limit
                or size + event_size > self.limits.max_event_bytes
            ):
                break
            selected.append(event)
            size += event_size
        events = tuple(selected)
        latest = events[-1].sequence if events else self._sequence
        batch = ControlEventBatch(
            source_id=self.source_id,
            source_epoch=self.source_epoch,
            engine_id=self.engine_id,
            engine_epoch=self.engine_epoch,
            namespace=self.namespace,
            events=events,
            earliest_sequence=earliest,
            latest_sequence=latest,
            gap=self._published < earliest - 1,
        )
        if self.publisher.try_publish(batch):
            self._published = latest
            self._publish_retry_at = 0.0
        else:
            self._publish_retry_at = self._clock() + 0.01

    def describe(self):
        source = (
            self.publisher.describe_local_source(0)
            if self.publisher is not None
            else None
        )
        return {
            "query": "events",
            "version": "1.0",
            "publisher": self.publisher is not None,
            "source_id": self.source_id,
            "source_epoch": self.source_epoch,
            "endpoint": source["endpoint"]
            if source is not None
            else (
                self.publisher_config.endpoint if self.publisher is not None else None
            ),
            "replay_endpoint": source.get("replay_endpoint")
            if source is not None
            else (
                self.publisher_config.replay_endpoint
                if self.publisher is not None
                else None
            ),
            "topic": "sglang.control.v1" if self.publisher is not None else None,
            "transport_sequence": "separate_frame_sequence",
            "cursor_sequence": "ControlEvent.sequence",
            "recovery": "query_known_scopes",
            "request_association": "bind_request_optional_provenance",
            "limits": self.limits,
        }

    def close(self):
        if self.publisher is not None:
            self.flush()
            self.publisher.shutdown()
            self.publisher = None

    def append(self, kind, caller_id, payload):
        self._owner()
        # A dropped/oversized event consumes a sequence. Polling surfaces a gap
        # instead of claiming that the absent effect did not happen.
        self._sequence += 1
        encoded = msgspec.json.encode(payload)
        if len(encoded) > self.limits.max_event_bytes:
            self._events.clear()
            self._bytes = 0
            return
        event = ControlEvent(
            sequence=self._sequence,
            kind=kind,
            caller_id=caller_id,
            cache_watermark=self.watermark(),
            payload=msgspec.json.decode(encoded),
        )
        size = len(msgspec.json.encode(event))
        if size > min(self.limits.max_bytes, self.limits.max_event_bytes):
            self._events.clear()
            self._bytes = 0
            return
        self._events.append((event, size))
        self._bytes += size
        while (
            len(self._events) > self.limits.max_events
            or self._bytes > self.limits.max_bytes
        ):
            _, removed = self._events.popleft()
            self._bytes -= removed

    def operation(self, result: OperationResult):
        # A synchronous PIN attaches its claim receipt immediately after its
        # terminal update. Publish that complete receipt once, not an earlier
        # terminal snapshot with its claim temporarily absent.
        if (
            result.action_type == "sglang.pin"
            and result.status == Status.COMPLETED
            and (result.public_result is None or result.public_result.claim is None)
        ):
            return
        self.append("operation", result.ref.context.caller_id, result)
        if result.public_result is not None and result.public_result.claim is not None:
            self.append(
                "claim", result.ref.context.caller_id, result.public_result.claim
            )
        if (
            result.update.logical_change is not None
            and result.update.logical_change.removed_tasks
        ):
            self.append(
                "association",
                result.ref.context.caller_id,
                {
                    "operation_ref": result.ref,
                    "change": result.update.logical_change,
                    "association_state": "local_index_change",
                },
            )

    def register_request(self, context, request_id):
        self._owner()
        key = context.caller_id, context.binding_id
        if (
            key not in self._requests
            and len(self._requests) >= self.limits.max_requests
        ):
            self.append(
                "invalidation",
                context.caller_id,
                {"reason": "request_feedback_capacity", "request_binding": context},
            )
            return
        self._requests[key] = request_id, None
        self.append(
            "association",
            context.caller_id,
            {
                "context": context,
                "request_id": request_id,
                "association_state": "unknown",
                "association": None,
            },
        )

    def bind_request(self, context, binding_id, association):
        self._owner()
        key = context.caller_id, binding_id
        if key not in self._requests:
            raise ControlError("missing", "Request binding is not live for this caller")
        request_id, previous = self._requests[key]
        if previous is not None and previous != association:
            raise ControlError(
                "conflict", "Request binding already has different explicit provenance"
            )
        if previous is None:
            self._requests[key] = request_id, association
            self.append(
                "association",
                context.caller_id,
                {
                    "binding_id": binding_id,
                    "association": association,
                    "association_state": "provided",
                },
            )
        return association

    def finish_request(
        self, context, req, *, checkpointed, aborted, committed, committed_tokens
    ):
        self._owner()
        tracked = self._requests.pop((context.caller_id, context.binding_id), None)
        if tracked is None:
            return
        request_id, association = tracked
        measured = req._cache_breakdown_computed
        use = RequestUse(
            context=context,
            request_id=request_id,
            association=association,
            association_state="provided" if association is not None else "unknown",
            cached_tokens=max(0, req.cached_tokens),
            device_tokens=req.cached_tokens_device if measured else None,
            host_tokens=req.cached_tokens_host if measured else None,
            storage_tokens=req.cached_tokens_storage if measured else None,
            tier_accounting="measured" if measured else "unknown",
            checkpointed=checkpointed,
            checkpointed_tokens=committed_tokens if checkpointed else 0,
            aborted=aborted,
            committed_prefix=committed,
        )
        self.append("request_use", context.caller_id, use)

    def query(self, *, caller_id, cursor, limit):
        self._owner()
        earliest = self._events[0][0].sequence if self._events else self._sequence + 1
        mismatch = cursor is not None and (
            cursor.source_id != self.source_id
            or cursor.source_epoch != self.source_epoch
        )
        after = 0 if cursor is None or mismatch else cursor.sequence
        if after > self._sequence:
            raise ControlError(
                "invalid_payload", "Feedback cursor exceeds this source watermark"
            )
        items = tuple(
            e
            for e, _ in self._events
            if e.sequence > after and e.caller_id in (None, caller_id)
        )[:limit]
        sequence = items[-1].sequence if len(items) == limit else self._sequence
        return FeedbackPage(
            source_id=self.source_id,
            source_epoch=self.source_epoch,
            engine_id=self.engine_id,
            engine_epoch=self.engine_epoch,
            namespace=self.namespace,
            items=items,
            next_cursor=FeedbackCursor(
                source_id=self.source_id,
                source_epoch=self.source_epoch,
                sequence=sequence,
            ),
            gap=mismatch or after < earliest - 1,
            earliest_sequence=earliest,
            latest_sequence=self._sequence,
            cache_watermark=self.watermark(),
        )

    def _owner(self):
        if threading.get_ident() != self._thread:
            raise ControlError(
                "wrong_owner", "Feedback is owned by the scheduler thread"
            )
