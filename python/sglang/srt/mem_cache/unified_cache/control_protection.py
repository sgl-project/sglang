# SPDX-License-Identifier: Apache-2.0
"""Synchronous local L2 PIN execution and bounded owner-confirmed feedback."""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated

import msgspec
from sglang.srt.managers.kv_control.contracts import (
    ClaimState,
    ControlContext,
    ControlError,
    ExecutionUpdate,
    Location,
    OperationRef,
    Pin,
    PublicLocation,
    Seconds,
    Status,
    TargetOutcome,
    TaskIdentity,
    Value,
)
from sglang.srt.managers.kv_control.protection import (
    HostPrefix,
    ProtectionKey,
    ProtectionRegistry,
    ProtectionSnapshot,
)

if TYPE_CHECKING:
    from sglang.srt.managers.kv_control.contracts import ResolvedTarget
    from sglang.srt.mem_cache.unified_cache.kv_control_adapter import (
        UnifiedCacheExecutor,
    )

Budget = Annotated[int, msgspec.Meta(ge=1, le=1 << 24)]


class ProtectionLimits(Value):
    max_claims: Budget = 1024
    max_records: Budget = 4096
    max_protected_tokens: Budget = 65536
    max_scope_tokens: Budget = 65536
    max_scope_prefixes: Budget = 256
    max_ttl_seconds: Seconds = 86400.0 * 365
    max_events: Budget = 4096


class ClaimEvent(Value):
    sequence: int
    claim: ClaimState


def protection_key(
    caller_id: str, claim_id: str, owner_task: TaskIdentity | None
) -> ProtectionKey:
    # Content/session selection never establishes demand ownership. Unowned
    # claims live in a disjoint caller scope and require explicit release/TTL.
    task = (
        "task:" + hashlib.sha256(owner_task.session_id.encode()).hexdigest()
        if owner_task
        else "caller"
    )
    return ProtectionKey(
        caller_id=caller_id,
        task_id=task,
        task_incarnation=owner_task.incarnation if owner_task else 0,
        claim_id=claim_id,
    )


def effective_claim_id(context: ControlContext, claim_id: str) -> str:
    """Return a public query/release handle, isolated by request incarnation.

    Attached actions use their original claim ID with the same bound context.
    Independent follow-ups use the returned ClaimState.claim_id with the same
    caller and no binding. A sibling sample therefore cannot renew/release the
    first sample by reusing its envelope's original claim ID.
    """
    if context.binding_id is None:
        return claim_id
    material = (
        "sglang.kv.attached.pin.v1",
        context.caller_id,
        context.binding_id,
        claim_id,
    )
    return "request:" + hashlib.sha256(msgspec.json.encode(material)).hexdigest()


class HostCacheProtection:
    def __init__(
        self,
        executor: UnifiedCacheExecutor,
        *,
        limits: ProtectionLimits,
        clock: Callable[[], float],
        max_result_targets: int,
    ):
        if type(limits.max_events) is not int or not 1 <= limits.max_events <= 1 << 24:
            raise ControlError(
                "invalid_payload",
                "Protection event budget must be bounded and positive",
            )
        self.executor = executor
        self.clock = clock
        self._max_result_targets = min(
            max_result_targets, executor.config.max_result_targets
        )
        self.registry = ProtectionRegistry(
            executor.cache,
            engine_id=executor.engine_id,
            engine_epoch=executor.engine_epoch,
            namespace=executor.config.namespace,
            clock=clock,
            **{
                name: getattr(limits, name)
                for name in limits.__struct_fields__
                if name != "max_events"
            },
        )
        executor.control_state.host_priorities = self.registry.priorities
        self._scopes: dict[ProtectionKey, tuple[TargetOutcome, ...]] = {}
        self._owners: dict[ProtectionKey, TaskIdentity | None] = {}
        self._events: deque[ClaimEvent] = deque(maxlen=limits.max_events)
        self._sequence = 0
        self._root = executor.core.root_node
        self._deadline = None

    @property
    def next_deadline(self) -> float | None:
        return self._deadline

    @property
    def has_background_work(self) -> bool:
        return self._deadline is not None

    def state(self, snapshot: ProtectionSnapshot) -> ClaimState:
        remaining = None
        if snapshot.state == "active" and snapshot.expires_at is not None:
            remaining = snapshot.expires_at - self.clock()
            if remaining <= 0:
                snapshot = self.registry.get(snapshot.key)
                remaining = None
        active = snapshot.state == "active"
        owner = self._owners.get(snapshot.key)
        return ClaimState(
            caller_id=snapshot.key.caller_id,
            task_incarnation=str(owner.incarnation) if owner else "caller",
            claim_id=snapshot.key.claim_id,
            revision=snapshot.revision,
            state=snapshot.state,
            owner_epoch=snapshot.engine_epoch,
            tombstone=not active,
            effective_protection=(
                "eviction_exclusion"
                if remaining is not None
                else "eviction_priority"
                if active and snapshot.priority is not None
                else None
            ),
            remaining_ttl=remaining,
            priority=snapshot.priority,
            owner_task=owner,
            actual_scope=self._scopes.get(snapshot.key, ()),
            location=PublicLocation(tier="host", engine_id=self.executor.engine_id),
        )

    def _publish(self, snapshot: ProtectionSnapshot) -> ClaimState:
        state = self.state(snapshot)
        self._sequence += 1
        self._events.append(ClaimEvent(sequence=self._sequence, claim=state))
        self._deadline = self.registry.next_deadline
        return state

    def get(
        self, *, caller_id: str, claim_id: str, owner_task: TaskIdentity | None
    ) -> ClaimState:
        snapshot = self.registry.get(protection_key(caller_id, claim_id, owner_task))
        if snapshot is None:
            raise ControlError(
                "missing", "Protection claim is not recorded in this epoch"
            )
        return self.state(snapshot)

    def changes(self, *, caller_id: str, after_sequence: int, limit: int):
        self.progress(limit=limit)
        earliest = self._events[0].sequence if self._events else self._sequence + 1
        items = tuple(
            e
            for e in self._events
            if e.sequence > after_sequence and e.claim.caller_id == caller_id
        )[:limit]
        return {
            "owner_epoch": self.executor.engine_epoch,
            "gap": after_sequence < earliest - 1,
            "items": items,
            "next_sequence": items[-1].sequence
            if len(items) == limit
            else self._sequence,
        }

    def progress(self, *, limit=64):
        reset = self.executor.core.root_node is not self._root
        if not reset and (self._deadline is None or self.clock() < self._deadline):
            return ()
        changed = tuple(self._publish(s) for s in self.registry.expire(limit=limit))
        self._deadline = self.registry.next_deadline
        self._root = self.executor.core.root_node
        return changed

    def invalidate(self, *, reason):
        changed = tuple(
            self._publish(s) for s in self.registry.invalidate(reason=reason)
        )
        self._deadline = self.registry.next_deadline
        return changed

    def host_evicted(self, values):
        page_size = self.executor.core.page_size
        pages = (page for value in values for page in value[::page_size].tolist())
        return tuple(self._publish(s) for s in self.registry.host_evicted(pages))

    def _prefixes(self, scope):
        self.executor._check(scope)
        end = scope.requested.end_token
        if scope.requested.start_token or end % self.executor.core.page_size:
            raise ControlError(
                "unsupported", "L2 PIN requires an exact page-aligned whole prefix"
            )
        compatibility = self.executor._compatibility(scope.content)
        return (
            HostPrefix(
                token_ids=scope.content.token_ids[:end],
                cache_salt=compatibility.cache_salt,
            ),
        )

    def outcomes(self, scope, *, status, code="", reason="", step=None):
        return (
            TargetOutcome(
                coverage=msgspec.structs.replace(scope.requested, component="full"),
                status=status,
                code=code,
                reason=reason,
                location=Location(tier="host", engine_id=self.executor.engine_id),
                completed_steps=(step,) if step else (),
            ),
        )

    def start(
        self, *, ref: OperationRef, payload: Pin, scope: ResolvedTarget | None
    ) -> tuple[ExecutionUpdate, ClaimState | None]:
        key = protection_key(
            ref.context.caller_id,
            effective_claim_id(ref.context, payload.claim_id),
            payload.owner_task,
        )
        previous = self.registry.get(key)
        try:
            if payload.mode == "acquire":
                if scope is None:
                    raise ControlError(
                        "missing", "PIN acquire requires a resolved complete prefix"
                    )
                snapshot = self.registry.acquire(
                    key=key,
                    revision=payload.revision,
                    prefixes=self._prefixes(scope),
                    ttl_seconds=payload.ttl_seconds,
                    priority=payload.priority,
                )
                if snapshot.state == "active":
                    self._scopes[key] = self.outcomes(
                        scope, status=Status.COMPLETED, step="protection_active"
                    )
            elif payload.mode == "renew":
                snapshot = self.registry.renew(
                    key=key,
                    revision=payload.revision,
                    ttl_seconds=payload.ttl_seconds,
                    priority=payload.priority,
                )
            else:
                snapshot = self.registry.release(key=key, revision=payload.revision)
            self._owners[key] = payload.owner_task
            state = (
                self._publish(snapshot)
                if previous != snapshot
                else self.state(snapshot)
            )
            outcomes = (
                self._scopes.get(key, ())
                if payload.mode == "acquire" and snapshot.state == "active"
                else ()
            )
            step = (
                "protection_active"
                if snapshot.state == "active"
                else "protection_" + snapshot.state
            )
            return ExecutionUpdate(
                status=Status.COMPLETED,
                quiescent=True,
                completed_steps=(step,),
                target_outcomes=outcomes,
            ), state
        except ControlError as error:
            status = {
                "missing": Status.MISSING,
                "incomplete": Status.MISSING,
                "capacity": Status.DEFERRED,
                "unsupported": Status.UNSUPPORTED,
                "unavailable": Status.DEFERRED,
            }.get(error.code, Status.REJECTED)
            outcomes = (
                self.outcomes(scope, status=status, code=error.code, reason=str(error))
                if scope
                else ()
            )
            return ExecutionUpdate(
                status=status,
                quiescent=True,
                code=error.code,
                reason=str(error),
                target_outcomes=outcomes,
            ), None
