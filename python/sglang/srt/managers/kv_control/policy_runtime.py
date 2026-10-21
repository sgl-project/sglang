# SPDX-License-Identifier: Apache-2.0
"""Opt-in retention and quota guidance over registered native operations."""

from __future__ import annotations

import hashlib
import math
import uuid
from dataclasses import replace
from itertools import islice
from typing import Annotated, TypeAlias

import msgspec

from sglang.srt.managers.kv_control.contracts import (
    MAX_INTEGER,
    ControlError,
    ExecutionUpdate,
    Identifier,
    Location,
    MoveTo,
    Priority,
    Seconds,
    SessionTarget,
    SetPriority,
    Status,
    Value,
)
from sglang.srt.managers.kv_control.policy import (
    KVPlanComposer,
    KVPolicy,
    OfferedProposal,
    PolicyObservation,
    PolicyProposal,
    PolicySignal,
    RequestGuidance,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType


class RetentionOptions(Value):
    priority: Priority = 50
    ttl_seconds: Seconds = 30.0


class PolicyDefinition(Value, tag="retention", tag_field="implementation"):
    name: Identifier
    options: RetentionOptions = RetentionOptions()


class SessionShare(Value):
    session_id: Identifier
    weight: Annotated[float, msgspec.Meta(gt=0, le=1e6)] = 1.0


class QuotaOptions(Value):
    shares: Annotated[tuple[SessionShare, ...], msgspec.Meta(max_length=4096)] = ()
    max_wait_seconds: Seconds = 1.0
    aging_seconds: Annotated[float, msgspec.Meta(ge=1e-6, le=86400 * 365)] = 1.0
    importance_ttl: Seconds = 60.0
    max_states: Annotated[int, msgspec.Meta(ge=1, le=65536)] = 1024
    reuse_score_weight: Annotated[float, msgspec.Meta(ge=0, le=1e6)] = 0.1
    borrow_idle_capacity: bool = True


class QuotaDefinition(Value, tag="quota", tag_field="implementation"):
    name: Identifier
    options: QuotaOptions = QuotaOptions()


PolicyConfig: TypeAlias = PolicyDefinition | QuotaDefinition


def validate_definitions(definitions):
    if len(definitions) > 16 or len({p.name for p in definitions}) != len(definitions):
        raise ValueError("Configure at most 16 uniquely named local policies")
    for definition in definitions:
        values = (
            (definition.options.ttl_seconds,)
            if isinstance(definition, PolicyDefinition)
            else (
                definition.options.max_wait_seconds,
                definition.options.aging_seconds,
                definition.options.importance_ttl,
                definition.options.reuse_score_weight,
                *(s.weight for s in definition.options.shares),
            )
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Policy timing, scores and shares must be finite")
        if isinstance(definition, QuotaDefinition) and len(
            {s.session_id for s in definition.options.shares}
        ) != len(definition.options.shares):
            raise ValueError("Quota share names must be unique")


def _digest(value):
    return hashlib.sha256(msgspec.json.encode(value)).hexdigest()


def _owner(tasks, target=None):
    return _digest(("task", tasks)) if tasks else _digest(("content", target))


def _claim(policy, owner):
    return _digest((policy, owner))


class RetentionPolicy(KVPolicy):
    consumed_signals = frozenset({"importance", "resume"})

    def __init__(self, options):
        self.options = options
        self.revision = 0

    def on_signal(self, signal, view):
        if signal.payload.signal not in self.consumed_signals:
            return None
        if self.revision >= MAX_INTEGER:
            raise ControlError("capacity", "Policy revision space exhausted")
        self.revision += 1
        return (
            PolicyProposal(
                "retain",
                SetPriority(
                    target=signal.payload.target,
                    location=Location(tier="gpu"),
                    claim_id=signal.owner_key,
                    revision=self.revision,
                    value=signal.payload.importance
                    if signal.payload.importance is not None
                    else self.options.priority,
                    ttl_seconds=self.options.ttl_seconds,
                ),
                signal.scope,
            ),
        )


class QuotaPolicy(KVPolicy):
    consumed_signals = frozenset({"importance", "resume"})
    observation_categories = frozenset({"capacity", "requests"})
    uses_guidance = True

    def __init__(self, options):
        self.options = options
        self._importance = {}

    def consumes(self, payload):
        return super().consumes(payload) and isinstance(payload.target, SessionTarget)

    def _expire(self, now):
        self._importance = {
            key: value for key, value in self._importance.items() if value[1] > now
        }

    def on_signal(self, signal, view):
        self._expire(view.now)
        if (
            signal.payload.signal == "importance"
            and signal.payload.importance is not None
        ):
            if (
                len(set(signal.tasks) - self._importance.keys()) + len(self._importance)
                > self.options.max_states
            ):
                raise ControlError(
                    "capacity", "Quota importance state capacity exhausted"
                )
            for task in signal.tasks:
                self._importance[task] = (
                    signal.payload.importance,
                    view.now + self.options.importance_ttl,
                )
        return ()

    def guidance(self, view):
        self._expire(view.now)
        gpu = next(
            (p for p in view.pools if p.component == "full" and p.tier == "gpu"), None
        )
        configured = {s.session_id: s.weight for s in self.options.shares}
        active = {
            r.task.session_id if r.task is not None else "" for r in view.requests
        }
        weights = {name: configured.get(name, 1.0) for name in active}
        total_weight = sum(weights.values()) or 1.0
        occupied = {}
        for request in view.requests:
            if not request.waiting and request.task is not None:
                # Per-request held rows are an upper bound when requests share
                # a prefix. They are advice, never physical reservations.
                occupied[request.task] = (
                    occupied.get(request.task, 0) + request.held_tokens
                )
        growth_known = all(
            r.estimated_running_growth is not None
            for r in view.requests
            if not r.waiting
        )
        growth = sum(
            r.estimated_running_growth or 0 for r in view.requests if not r.waiting
        )
        available = (
            None
            if gpu is None or gpu.reclaimable_tokens is None or not growth_known
            else gpu.free_tokens + gpu.reclaimable_tokens - growth
        )
        spare = (
            self.options.borrow_idle_capacity
            and view.requests_complete
            and view.estimated_waiting_tokens is not None
            and available is not None
            and view.estimated_waiting_tokens <= available
        )
        results = []
        for request in view.requests:
            if not request.waiting:
                continue
            importance = self._importance.get(request.task, (0, 0))[0]
            score = (
                importance / 100 + request.waiting_seconds / self.options.aging_seconds
            )
            if request.cached_tokens is not None:
                score += self.options.reuse_score_weight * request.cached_tokens / 1024
            admit = True
            known = (
                view.requests_complete
                and request.task is not None
                and request.estimated_incremental_tokens is not None
                and gpu is not None
            )
            if (
                known
                and not spare
                and request.waiting_seconds < self.options.max_wait_seconds
            ):
                name = request.task.session_id
                share = weights[name] / total_weight
                admit = occupied.get(
                    request.task, 0
                ) + request.estimated_incremental_tokens <= max(
                    1, int(gpu.total_tokens * share)
                )
            results.append(
                RequestGuidance(
                    request.identity, admit, score, "" if admit else "session_share"
                )
            )
        return tuple(results)

    def cleanup(self, tasks):
        for task in tasks:
            self._importance.pop(task, None)


class KVPolicyRuntime:
    """Retention consumes values; the native core owns every physical result.

    Physical proposals currently use priority only; quota supplies read-only
    request guidance. Movement needs the preparation/admission boundary.
    """

    def __init__(self, service):
        self.service, self.control = service, service.control
        self.claims = service.cache.tree_core.priority_claims
        definitions = service.config.policies
        validate_definitions(definitions)
        self.policies = {
            p.name: (
                RetentionPolicy(p.options)
                if isinstance(p, PolicyDefinition)
                else QuotaPolicy(p.options)
            )
            for p in definitions
        }
        self.has_guidance = any(p.uses_guidance for p in self.policies.values())
        if self.has_guidance and service.cache.tree_core.component_types != (
            ComponentType.FULL,
        ):
            raise ValueError(
                "Quota guidance currently requires the built-in FULL cache"
            )
        self._binding_tasks = {}
        self._arrival_sequence = 0
        self._guidance = {}
        self._view = None
        self.composer = KVPlanComposer(self.control.limits)

    def consumes(self, signal):
        return any(policy.consumes(signal) for policy in self.policies.values())

    def on_signal(self, ref, payload, *, scope=None):
        consumers = [
            (name, policy)
            for name, policy in self.policies.items()
            if policy.consumes(payload)
        ]
        if not consumers:
            return False
        # A consumer exists before either history or an observation is read.
        scope = self.control.index.resolve(payload.target) if scope is None else scope
        tasks = tuple(
            sorted(
                {
                    b.continuation.task
                    for b in scope.branches
                    if b.continuation is not None
                },
                key=lambda t: (t.session_id, t.incarnation),
            )
        )
        signal = PolicySignal(payload, scope, _owner(tasks, payload.target), tasks)
        if ref.context.binding_id is not None and tasks:
            key = ref.context.caller_id, ref.context.binding_id
            if (
                key not in self._binding_tasks
                and len(self._binding_tasks) >= self.control.limits.max_active
            ):
                raise ControlError(
                    "capacity", "Policy request binding capacity exhausted"
                )
            combined = set(self._binding_tasks.get(key, ())) | set(tasks)
            if len(combined) > self.control.limits.max_branches:
                raise ControlError("capacity", "Policy request task capacity exhausted")
            self._binding_tasks[key] = tuple(
                sorted(combined, key=lambda t: (t.session_id, t.incarnation))
            )
        view = PolicyObservation(
            self.control._clock(),
            self.control.registry.namespace,
            self.control.registry.engine_epoch,
        )
        offered = []
        for name, policy in consumers:
            for proposal in policy.on_signal(signal, view) or ():
                if not isinstance(proposal.action, SetPriority):
                    raise ControlError(
                        "unsupported",
                        "This runtime profile executes priority proposals only",
                    )
                action = msgspec.structs.replace(
                    proposal.action, claim_id=_claim(name, signal.owner_key)
                )
                authoritative = (
                    scope
                    if action.target == payload.target
                    else self.control.index.resolve(action.target)
                )
                offered.append(
                    OfferedProposal(
                        name, replace(proposal, action=action, scope=authoritative)
                    )
                )
        # Only pending explicit demotions compete with retention. Live explicit
        # priorities are merged by the cache's private-origin authority rule.
        explicit = (
            tuple(
                islice(
                    (
                        PolicyProposal(p.ref.operation_id, p.payload, p.target)
                        for p in self.control._pending.values()
                        if p.origin == "explicit"
                        and isinstance(p.payload, MoveTo)
                        and p.target is not None
                    ),
                    self.control.limits.max_actions + 1,
                )
            )
            if offered
            else ()
        )
        plan = (
            self.composer.compose(tuple(offered), view, explicit=explicit)
            if offered
            else ()
        )
        refs = ()
        try:
            if plan:
                refs = self.control._reserve_policy_operations(
                    tuple(
                        (_digest((p.ref.policy, p.ref.proposal)), p.proposal.action)
                        for p in plan
                    ),
                    resolutions=tuple(p.proposal.scope for p in plan),
                )
            outcomes = {}
            remaining = list(zip(plan, refs, strict=True))
            while remaining:
                ready = [
                    (entry, child)
                    for entry, child in remaining
                    if all(d in outcomes for d in entry.dependencies)
                ]
                if not ready:
                    raise ControlError(
                        "invalid_payload", "Unresolved policy dependency graph"
                    )
                for entry, child in ready:
                    if not entry.accepted:
                        result = self.control._finish(
                            child,
                            ExecutionUpdate(
                                status=Status.DEFERRED, quiescent=True, code=entry.code
                            ),
                        )
                    elif any(
                        outcomes[d].status != Status.COMPLETED
                        for d in entry.dependencies
                    ):
                        result = self.control._finish(
                            child,
                            ExecutionUpdate(
                                status=Status.DEFERRED,
                                quiescent=True,
                                code="dependency_failed",
                            ),
                        )
                    else:
                        result = self.control._start_policy_operation(child)
                    outcomes[entry.ref] = result
                    remaining.remove((entry, child))
            self.control.registry.update(
                ref,
                ExecutionUpdate(
                    status=Status.COMPLETED,
                    quiescent=True,
                    code="policy_context_updated",
                    completed_steps=("policy_context_updated",),
                    derived_operations=refs,
                ),
            )
        except Exception as error:
            for child in refs:
                if not self.control.registry.get(child).status.terminal:
                    self.control.cancel(child)
            if not refs and isinstance(error, ControlError):
                raise
            self.control.registry.update(
                ref,
                ExecutionUpdate(
                    status=Status.FAILED,
                    quiescent=True,
                    code=getattr(error, "code", "policy_error"),
                    reason=str(error)[:4096],
                    derived_operations=refs,
                ),
            )
        return True

    def cleanup(self, tasks):
        identities = frozenset(
            ("local-policy", _claim(name, _owner((task,))))
            for task in tasks
            for name in self.policies
        )
        if identities:
            self.claims.drop_policy_claims(identities)
        for policy in self.policies.values():
            policy.cleanup(tasks)
        self._guidance.clear()
        self._view = None

    def register_request(self, req):
        if not self.has_guidance or hasattr(req, "kv_control_policy_identity"):
            return
        self._arrival_sequence += 1
        req.kv_control_policy_identity = uuid.uuid4().hex
        req.kv_control_policy_arrival = self.control._clock()
        req.kv_control_policy_order = self._arrival_sequence
        session_id = getattr(req, "session_id", None)
        task = self.control.index.task_identity(session_id) if session_id else None
        req.kv_control_policy_tasks = (task,) if task is not None else ()

    def bind_request(self, req, context):
        tasks = self._binding_tasks.pop((context.caller_id, context.binding_id), ())
        if self.has_guidance:
            self.register_request(req)
            if tasks:
                req.kv_control_policy_identity = context.binding_id
                req.kv_control_policy_tasks = tasks

    def finish_request(self, req):
        context = getattr(req, "kv_control_context", None)
        if context is not None:
            self._binding_tasks.pop((context.caller_id, context.binding_id), None)

    def observe_scheduler(self, scheduler, running_batch):
        from sglang.srt.managers.kv_control.policy_observer import observe_scheduler

        self._view = observe_scheduler(self, scheduler, running_batch)
        guidance = tuple(
            g
            for policy in self.policies.values()
            if policy.uses_guidance
            for g in policy.guidance(self._view)
        )
        self._guidance = {g.identity: g for g in self.composer.combine(guidance)}

    def waiting_seconds(self, req):
        return max(
            0.0,
            self.control._clock()
            - getattr(req, "kv_control_policy_arrival", self.control._clock()),
        )

    def allows_request(self, req):
        if self.waiting_seconds(req) >= self.service.config.policy_max_wait_seconds:
            return True
        guidance = self._guidance.get(getattr(req, "kv_control_policy_identity", None))
        return guidance.admit if guidance is not None else True

    def request_key(self, req, original_order, priority_sign):
        identity = getattr(req, "kv_control_policy_identity", None)
        guidance = self._guidance.get(identity)
        priority = (getattr(req, "priority", 0) or 0) * priority_sign
        arrival = getattr(req, "kv_control_policy_order", original_order)
        expired = (
            self.waiting_seconds(req) >= self.service.config.policy_max_wait_seconds
        )
        return (
            priority,
            0 if expired else 1,
            0 if expired or guidance is None or guidance.admit else 1,
            arrival if expired else -(guidance.score if guidance is not None else 0),
            arrival,
            original_order,
        )
