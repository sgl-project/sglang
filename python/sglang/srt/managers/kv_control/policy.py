# SPDX-License-Identifier: Apache-2.0
"""Read-only policy values and bounded composition before owner execution."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import msgspec

from sglang.srt.managers.kv_control.contracts import (
    ContentTarget,
    ControlError,
    CopyTo,
    Identifier,
    Limits,
    MoveTo,
    Primitive,
    SetPriority,
    Signal,
    TaskIdentity,
    TokensTarget,
    cache_compatibility,
    parse_action,
    validated,
)
from sglang.srt.managers.kv_control.index import ResolvedTarget
from sglang.srt.managers.kv_hints import KvHintAction


@dataclass(frozen=True)
class PoolObservation:
    component: str
    tier: str
    total_tokens: int
    free_tokens: int
    referenced_tokens: int
    reclaimable_tokens: int | None = None


@dataclass(frozen=True)
class RequestObservation:
    identity: str
    task: TaskIdentity | None
    waiting: bool
    waiting_seconds: float
    arrival_order: int
    held_tokens: int
    cached_tokens: int | None = None
    estimated_incremental_tokens: int | None = None
    estimated_running_growth: int | None = None


@dataclass(frozen=True)
class PolicyObservation:
    now: float
    namespace: str
    engine_epoch: str
    pools: tuple[PoolObservation, ...] = ()
    requests: tuple[RequestObservation, ...] = ()
    requests_complete: bool = False
    measured_save_seconds: float | None = None
    measured_load_seconds: float | None = None
    estimated_waiting_tokens: int | None = None


@dataclass(frozen=True)
class PolicySignal:
    payload: Signal
    scope: ResolvedTarget
    owner_key: str
    tasks: tuple[TaskIdentity, ...] = ()
    previous_scope: ResolvedTarget | None = None


@dataclass(frozen=True)
class ProposalRef:
    policy: str
    proposal: str


@dataclass(frozen=True)
class PolicyProposal:
    identity: str
    action: Primitive
    scope: ResolvedTarget
    dependencies: tuple[ProposalRef, ...] = ()
    not_before: float = 0.0
    prepare_for_admission: bool = False


@dataclass(frozen=True)
class OfferedProposal:
    policy: str
    proposal: PolicyProposal

    @property
    def ref(self):
        return ProposalRef(self.policy, self.proposal.identity)


@dataclass(frozen=True)
class PlanEntry:
    ref: ProposalRef
    proposal: PolicyProposal
    dependencies: tuple[ProposalRef, ...]
    accepted: bool
    code: str = ""


@dataclass(frozen=True)
class RequestGuidance:
    identity: str
    admit: bool = True
    score: float = 0.0
    reason: str = ""


class KVPolicy:
    """Extensions receive immutable values, never a scheduler or allocator.

    The runtime checks subscriptions before resolving history or constructing
    observations. An empty subscription has no signal consumers. Only the owner
    executes proposals; this interface grants no capacity or correctness holds.
    """

    consumed_signals: frozenset[str] = frozenset()
    observation_categories: frozenset[str] = frozenset()
    uses_guidance = False
    movement_policy = False

    def consumes(self, payload):
        return payload.signal in self.consumed_signals

    def on_signal(self, signal: PolicySignal, view: PolicyObservation):
        return None

    def guidance(self, view: PolicyObservation) -> tuple[RequestGuidance, ...]:
        return ()

    def cleanup(self, tasks: tuple[TaskIdentity, ...]):
        pass

    def on_result(self, proposal, result, *, elapsed_seconds):
        pass


def scopes_overlap(left: ResolvedTarget, right: ResolvedTarget):
    if (left.namespace, left.engine_epoch) != (right.namespace, right.engine_epoch):
        return False
    for a in left.branches:
        for b in right.branches:
            start = max(a.requested.start_token, b.requested.start_token)
            end = min(a.requested.end_token, b.requested.end_token)
            if start >= end:
                continue
            ac, bc = cache_compatibility(a.content), cache_compatibility(b.content)
            if (ac.cache_salt, ac.extra_key, ac.is_multimodal) != (
                bc.cache_salt,
                bc.extra_key,
                bc.is_multimodal,
            ):
                continue
            if (
                ac.layout_id is not None
                and bc.layout_id is not None
                and ac.layout_id != bc.layout_id
            ):
                continue
            if isinstance(a.content, TokensTarget) and isinstance(
                b.content, TokensTarget
            ):
                common = 0
                for x, y in zip(a.content.token_ids[:end], b.content.token_ids[:end]):
                    if x != y:
                        break
                    common += 1
                if common > start:
                    return True
            elif isinstance(a.content, ContentTarget) and isinstance(
                b.content, ContentTarget
            ):
                if set(a.content.components).isdisjoint(b.content.components):
                    continue
                # Unequal prefix hashes may still share an earlier physical
                # prefix. Without tokens, these scopes may overlap.
                return True
            else:
                # A descriptor cannot prove it differs from a token scope.
                # Defer competing advice rather than assuming disjoint copies.
                return True
    return False


def placement(action):
    if isinstance(action, SetPriority):
        return action.location.tier if action.value is not None else ""
    if isinstance(action, CopyTo):
        return "gpu" if action.dst.tier == "gpu" else ""
    return action.dst.tier if isinstance(action, MoveTo) else ""


class KVPlanComposer:
    """Resolve advice, validate dependencies, and leave approval to the owner.

    Compatible copies retain independent plan entries; the physical adapters
    coalesce them without coupling cancellation or results. Configured order
    breaks policy competition. Explicit intent always participates before advice.
    """

    def __init__(self, limits=Limits()):
        self.limits = limits

    def compose(
        self,
        offered: tuple[OfferedProposal, ...],
        view: PolicyObservation,
        *,
        explicit: tuple[PolicyProposal, ...] = (),
    ):
        if len(offered) + len(explicit) > self.limits.max_actions:
            raise ControlError("capacity", "Policy composition action budget exhausted")
        prepared = []
        entries, tokens, action_bytes = 0, 0, 0
        scopes = {}
        for item in offered:
            validated(item.policy, Identifier)
            validated(item.proposal.identity, Identifier)
            proposal = item.proposal
            if type(proposal.action) not in (SetPriority, CopyTo, MoveTo):
                raise ControlError(
                    "invalid_payload", "Policies may propose primitives only"
                )
            kind = {SetPriority: "set_priority", CopyTo: "copy_to", MoveTo: "move_to"}[
                type(proposal.action)
            ]
            encoded = msgspec.json.encode(proposal.action)
            action_bytes += len(encoded)
            if action_bytes > self.limits.max_envelope_bytes:
                raise ControlError("capacity", "Policy action byte budget exhausted")
            action = parse_action(
                KvHintAction(
                    action_id=proposal.identity,
                    action_type="sglang." + kind,
                    action_version="1.0",
                    payload=msgspec.json.decode(encoded),
                )
            )
            if isinstance(action.target, str):
                raise ControlError(
                    "invalid_payload", "Policy proposals require logical targets"
                )
            if len(proposal.dependencies) > self.limits.max_actions:
                raise ControlError("capacity", "Policy dependency budget exhausted")
            dependencies = []
            for dep in proposal.dependencies:
                if not isinstance(dep, ProposalRef):
                    raise ControlError(
                        "invalid_payload", "Invalid policy dependency reference"
                    )
                dependencies.append(
                    ProposalRef(
                        validated(dep.policy, Identifier),
                        validated(dep.proposal, Identifier),
                    )
                )
            if id(proposal.scope) not in scopes:
                scopes[id(proposal.scope)] = validated(proposal.scope, ResolvedTarget)
            scope = scopes[id(proposal.scope)]
            if action.target.namespace != scope.namespace:
                raise ControlError(
                    "incompatible_namespace",
                    "Policy action and snapshot namespaces differ",
                )
            if (scope.namespace, scope.engine_epoch) != (
                view.namespace,
                view.engine_epoch,
            ):
                raise ControlError(
                    "stale_epoch", "Policy proposal belongs to another owner"
                )
            if (
                type(proposal.not_before) not in (int, float)
                or not math.isfinite(proposal.not_before)
                or type(proposal.prepare_for_admission) is not bool
            ):
                raise ControlError(
                    "invalid_payload", "Invalid policy timing or preparation flag"
                )
            if proposal.prepare_for_admission and not (
                isinstance(action, CopyTo) and action.dst.tier == "gpu"
            ):
                raise ControlError(
                    "invalid_payload", "Admission preparation requires a GPU COPY"
                )
            if not any(p.proposal.scope is scope for p in prepared):
                entries += len(scope.branches) + sum(
                    len(b.spans) for b in scope.branches
                )
                tokens += sum(
                    len(b.content.token_ids)
                    for b in scope.branches
                    if isinstance(b.content, TokensTarget)
                )
            if (
                entries > self.limits.max_scope_spans
                or tokens > self.limits.max_scope_tokens
            ):
                raise ControlError("capacity", "Policy snapshot budget exhausted")
            prepared.append(
                OfferedProposal(
                    item.policy,
                    replace(
                        proposal,
                        action=action,
                        scope=scope,
                        dependencies=tuple(dependencies),
                    ),
                )
            )
        refs = {item.ref for item in prepared}
        if len(refs) != len(prepared):
            raise ControlError(
                "conflict", "Policy proposal IDs must be unique per policy"
            )
        accepted = list(explicit)
        plan = []
        for item in prepared:
            here = placement(item.proposal.action)
            conflict = bool(here) and any(
                placement(other.action)
                and placement(other.action) != here
                and scopes_overlap(item.proposal.scope, other.scope)
                for other in accepted
            )
            plan.append(
                PlanEntry(
                    item.ref,
                    item.proposal,
                    item.proposal.dependencies,
                    not conflict,
                    "policy_conflict" if conflict else "",
                )
            )
            if not conflict:
                accepted.append(item.proposal)
        # A new GPU copy must publish before a policy changes that copy's
        # priority. Action list order supplies no such dependency.
        for i, entry in enumerate(plan):
            deps = list(entry.dependencies)
            for dep in deps:
                if not isinstance(dep, ProposalRef) or dep not in refs:
                    raise ControlError("invalid_payload", "Unknown policy dependency")
            if (
                entry.accepted
                and isinstance(entry.proposal.action, SetPriority)
                and entry.proposal.action.value is not None
                and entry.proposal.action.location.tier == "gpu"
            ):
                for other in plan:
                    action = other.proposal.action
                    if (
                        other.accepted
                        and isinstance(action, CopyTo)
                        and action.dst.tier == "gpu"
                        and scopes_overlap(entry.proposal.scope, other.proposal.scope)
                        and other.ref not in deps
                    ):
                        deps.append(other.ref)
            plan[i] = replace(entry, dependencies=tuple(deps))
        by_ref = {entry.ref: entry for entry in plan}
        visiting, visited = set(), set()

        def check(ref):
            if ref in visiting:
                raise ControlError("invalid_payload", "Cyclic policy dependencies")
            if ref in visited:
                return
            visiting.add(ref)
            for dep in by_ref[ref].dependencies:
                check(dep)
            visiting.remove(ref)
            visited.add(ref)

        for ref in refs:
            check(ref)
        # Rejected predecessors reject dependents even when listed later.
        rejected = {entry.ref for entry in plan if not entry.accepted}
        while True:
            new = {
                entry.ref
                for entry in plan
                if any(dep in rejected for dep in entry.dependencies)
            } - rejected
            if not new:
                break
            rejected.update(new)
        return tuple(
            replace(entry, accepted=False, code=entry.code or "dependency_rejected")
            if entry.ref in rejected
            else entry
            for entry in plan
        )

    @staticmethod
    def combine(guidance: tuple[RequestGuidance, ...]):
        merged = {}
        for item in guidance:
            validated(item.identity, Identifier)
            if not math.isfinite(item.score) or type(item.admit) is not bool:
                raise ControlError("invalid_payload", "Invalid scheduler guidance")
            old = merged.get(item.identity)
            score = item.score + (old.score if old else 0)
            if not math.isfinite(score):
                raise ControlError("invalid_payload", "Combined policy score overflow")
            merged[item.identity] = RequestGuidance(
                item.identity,
                item.admit and (old.admit if old else True),
                score,
                (old.reason if old else "") or item.reason,
            )
        return tuple(merged.values())
