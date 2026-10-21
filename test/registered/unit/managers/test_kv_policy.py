# SPDX-License-Identifier: Apache-2.0
"""Policy composition contract, independent of model or allocator objects."""

from dataclasses import replace

import pytest

from sglang.srt.managers.kv_control.contracts import (
    ControlError,
    CopyTo,
    Limits,
    Location,
    MoveTo,
    SetPriority,
    TokensTarget,
)
from sglang.srt.managers.kv_control.index import LogicalIndex
from sglang.srt.managers.kv_control.policy import (
    KVPlanComposer,
    OfferedProposal,
    PolicyObservation,
    PolicyProposal,
    ProposalRef,
    RequestGuidance,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
VIEW = PolicyObservation(now=5, namespace="ns", engine_epoch="epoch")


def offer(
    kind="priority",
    name="p",
    *,
    tokens=(1, 2, 3, 4),
    salt=None,
    dependencies=(),
    **kwargs,
):
    target = TokensTarget(namespace="ns", token_ids=tokens, cache_salt=salt)
    scope = LogicalIndex("ns", "epoch").resolve(target)
    action = (
        SetPriority(
            target=target,
            location=Location(tier="gpu"),
            claim_id="claim",
            revision=1,
            value=50,
            ttl_seconds=10,
        )
        if kind == "priority"
        else CopyTo(target=target, src=Location(tier="host"), dst=Location(tier="gpu"))
        if kind == "copy"
        else MoveTo(target=target, src=Location(tier="gpu"), dst=Location(tier="host"))
    )
    return OfferedProposal(
        name, PolicyProposal(kind, action, scope, dependencies=dependencies, **kwargs)
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_priority_waits_for_copy_independent_of_offer_order(reverse):
    items = (offer(), offer("copy"))
    plan = KVPlanComposer().compose(items[::-1] if reverse else items, VIEW)
    priority = next(p for p in plan if p.ref.proposal == "priority")
    assert all(p.accepted for p in plan)
    assert priority.dependencies == (ProposalRef("p", "copy"),)


def test_compatible_copies_keep_independent_interests_for_physical_coalescing():
    plan = KVPlanComposer().compose((offer("copy", "a"), offer("copy", "b")), VIEW)
    assert len(plan) == 2 and all(p.accepted for p in plan)
    assert plan[0].ref != plan[1].ref


def test_explicit_intent_and_transitive_rejection_precede_policy_order():
    move = offer("move", "save")
    copy = offer("copy", "load", dependencies=(move.ref,))
    priority = offer("priority", "retain", dependencies=(copy.ref,))
    plan = KVPlanComposer().compose(
        (priority, copy, move), VIEW, explicit=(offer("copy", "explicit").proposal,)
    )
    assert [p.code for p in plan] == [
        "dependency_rejected",
        "dependency_rejected",
        "policy_conflict",
    ]
    assert not any(p.accepted for p in plan)


@pytest.mark.parametrize("other", [dict(tokens=(8, 2, 9, 4)), dict(salt="different")])
def test_unrelated_token_prefixes_and_compatibility_do_not_conflict(other):
    plan = KVPlanComposer().compose((offer("move"), offer("copy", "b", **other)), VIEW)
    assert all(p.accepted for p in plan)


def test_policy_order_resolves_competing_placement():
    plan = KVPlanComposer().compose((offer("move"), offer("copy", "b")), VIEW)
    assert plan[0].accepted and not plan[1].accepted


def test_different_branches_still_conflict_on_their_shared_prefix():
    plan = KVPlanComposer().compose(
        (offer("move"), offer("copy", "b", tokens=(1, 2, 9, 4))), VIEW
    )
    assert plan[0].accepted and not plan[1].accepted


@pytest.mark.parametrize(
    "dependencies", [(ProposalRef("missing", "copy"),), (ProposalRef("p", "priority"),)]
)
def test_unknown_and_cyclic_dependencies_fail_before_execution(dependencies):
    with pytest.raises(ControlError, match="dependency|dependencies"):
        KVPlanComposer().compose((offer(dependencies=dependencies),), VIEW)


def test_snapshot_and_preparation_constraints_are_enforced():
    with pytest.raises(ControlError, match="budget"):
        KVPlanComposer(Limits(max_scope_tokens=2)).compose((offer(),), VIEW)
    with pytest.raises(ControlError, match="GPU COPY"):
        KVPlanComposer().compose((offer(prepare_for_admission=True),), VIEW)
    old = offer()
    with pytest.raises(ControlError, match="another owner"):
        KVPlanComposer().compose(
            (
                replace(
                    old,
                    proposal=replace(
                        old.proposal,
                        scope=LogicalIndex("ns", "old").resolve(
                            old.proposal.action.target
                        ),
                    ),
                ),
            ),
            VIEW,
        )


def test_duplicate_ids_and_invalid_primitive_fail_before_composition():
    with pytest.raises(ControlError, match="unique"):
        KVPlanComposer().compose((offer(), offer()), VIEW)
    invalid = offer()
    invalid = replace(
        invalid,
        proposal=replace(
            invalid.proposal,
            action=SetPriority(
                target=invalid.proposal.action.target,
                location=Location(tier="gpu"),
                claim_id="c",
                revision=1,
                value=10,
            ),
        ),
    )
    with pytest.raises(ControlError):
        KVPlanComposer().compose((invalid,), VIEW)


def test_guidance_combines_deferrals_scores_and_rejects_overflow():
    combined = KVPlanComposer.combine(
        (
            RequestGuidance("r", score=2),
            RequestGuidance("r", admit=False, score=-1, reason="share"),
        )
    )
    assert combined == (RequestGuidance("r", admit=False, score=1, reason="share"),)
    with pytest.raises(ControlError, match="overflow"):
        KVPlanComposer.combine(
            (RequestGuidance("r", score=1e308), RequestGuidance("r", score=1e308))
        )


def test_action_bytes_and_dependency_fanout_are_bounded():
    with pytest.raises(ControlError, match="byte budget"):
        KVPlanComposer(Limits(max_envelope_bytes=16)).compose((offer(),), VIEW)
    copy = offer("copy", dependencies=(ProposalRef("p", "copy"),) * 3)
    with pytest.raises(ControlError, match="dependency budget"):
        KVPlanComposer(Limits(max_actions=2)).compose((copy,), VIEW)


def test_automatic_publication_dependency_cannot_introduce_a_cycle():
    priority = offer()
    copy = offer("copy", dependencies=(priority.ref,))
    with pytest.raises(ControlError, match="Cyclic"):
        KVPlanComposer().compose((priority, copy), VIEW)
