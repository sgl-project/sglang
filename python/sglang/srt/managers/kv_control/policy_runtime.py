# SPDX-License-Identifier: Apache-2.0
"""Opt-in retention policy runtime over registered native operations."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from itertools import islice
from typing import Literal

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
)


class RetentionOptions(Value):
    priority: Priority = 50
    ttl_seconds: Seconds = 30.0


class PolicyDefinition(Value):
    name: Identifier
    implementation: Literal["retention"] = "retention"
    options: RetentionOptions = RetentionOptions()


def validate_definitions(definitions):
    if len(definitions) > 16 or len({p.name for p in definitions}) != len(definitions):
        raise ValueError("Configure at most 16 uniquely named local policies")


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


class KVPolicyRuntime:
    """Retention consumes values; the native core owns every physical result.

    This first executable policy profile uses priority only. Movement/guidance
    policies require the separate preparation and scheduler admission boundary.
    """

    def __init__(self, service):
        self.service, self.control = service, service.control
        self.claims = service.cache.tree_core.priority_claims
        definitions = service.config.policies
        validate_definitions(definitions)
        self.policies = {p.name: RetentionPolicy(p.options) for p in definitions}
        self.composer = KVPlanComposer(self.control.limits)

    def consumes(self, signal):
        return any(
            signal in policy.consumed_signals for policy in self.policies.values()
        )

    def on_signal(self, ref, payload):
        consumers = [
            (name, policy)
            for name, policy in self.policies.items()
            if payload.signal in policy.consumed_signals
        ]
        if not consumers:
            return False
        # A consumer exists before either history or an observation is read.
        scope = self.control.index.resolve(payload.target)
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
                        "unsupported", "This runtime profile executes retention only"
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
        explicit = tuple(
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
        plan = self.composer.compose(tuple(offered), view, explicit=explicit)
        refs = ()
        try:
            if plan:
                refs = self.control._reserve_policy_operations(
                    tuple(
                        (_digest((p.ref.policy, p.ref.proposal)), p.proposal.action)
                        for p in plan
                    )
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
