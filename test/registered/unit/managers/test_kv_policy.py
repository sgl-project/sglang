# SPDX-License-Identifier: Apache-2.0
"""Policy dependencies, explicit authority and bounded quota/tool advice."""

import unittest
from dataclasses import replace
from types import SimpleNamespace

import pytest

from sglang.srt.managers.kv_control.contracts import (
    Associate,
    ControlError,
    CopyTo,
    ExecutionUpdate,
    Limits,
    Location,
    MoveTo,
    SessionTarget,
    SetPriority,
    Signal,
    Status,
    TaskIdentity,
    TokensTarget,
)
from sglang.srt.managers.kv_control.index import LogicalIndex
from sglang.srt.managers.kv_control.policy import (
    KVPlanComposer,
    OfferedProposal,
    PolicyObservation,
    PolicyProposal,
    PolicySignal,
    PoolObservation,
    ProposalRef,
    RequestObservation,
)
from sglang.srt.managers.kv_control.policy_runtime import QuotaOptions, QuotaPolicy
from sglang.srt.managers.kv_control.service import parse_config
from sglang.srt.managers.kv_control.tool_wait_policy import (
    ToolWaitOptions,
    ToolWaitPolicy,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")
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


class TestKVPlanComposer(CustomTestCase):
    def test_priority_waits_for_copy_independent_of_offer_order(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                items = (offer(), offer("copy"))
                plan = KVPlanComposer().compose(items[::-1] if reverse else items, VIEW)
                priority = next((p for p in plan if p.ref.proposal == "priority"))
                assert all((p.accepted for p in plan))
                assert priority.dependencies == (ProposalRef("p", "copy"),)

    def test_explicit_intent_and_transitive_rejection_precede_policy_order(self):
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
        assert not any((p.accepted for p in plan))

    def test_different_branches_still_conflict_on_their_shared_prefix(self):
        plan = KVPlanComposer().compose(
            (offer("move"), offer("copy", "b", tokens=(1, 2, 9, 4))), VIEW
        )
        assert plan[0].accepted and (not plan[1].accepted)

    def test_unknown_and_cyclic_dependencies_fail_before_execution(self):
        for dependencies in [
            (ProposalRef("missing", "copy"),),
            (ProposalRef("p", "priority"),),
        ]:
            with self.subTest(dependencies=dependencies):
                with pytest.raises(ControlError, match="dependency|dependencies"):
                    KVPlanComposer().compose((offer(dependencies=dependencies),), VIEW)

    def test_snapshot_and_preparation_constraints_are_enforced(self):
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


A, B = (
    TaskIdentity(session_id="A", incarnation=1),
    TaskIdentity(session_id="B", incarnation=2),
)


def quota_view(*, complete=True, demand=40):
    return PolicyObservation(
        10,
        "ns",
        "epoch",
        (PoolObservation("full", "gpu", 100, 60, 40, 0),),
        (
            RequestObservation(
                "running", A, False, 0, 0, 40, estimated_running_growth=20
            ),
            RequestObservation(
                "a", A, True, 0.1, 1, 0, estimated_incremental_tokens=20
            ),
            RequestObservation(
                "b", B, True, 0.1, 2, 0, estimated_incremental_tokens=20
            ),
        ),
        complete,
        estimated_waiting_tokens=demand,
    )


class TestKVQuotaPolicy(CustomTestCase):
    def test_soft_share_defers_busy_session_but_idle_capacity_is_borrowable(self):
        assert [
            g.admit
            for g in QuotaPolicy(QuotaOptions(borrow_idle_capacity=False)).guidance(
                quota_view()
            )
        ] == [False, True]
        assert [
            g.admit for g in QuotaPolicy(QuotaOptions()).guidance(quota_view())
        ] == [True, True]

    def test_missing_observation_is_native_fallback_not_zero_demand(self):
        snapshot = quota_view()
        snapshot = replace(
            snapshot,
            requests=(
                snapshot.requests[0],
                replace(snapshot.requests[1], estimated_incremental_tokens=None),
                snapshot.requests[2],
            ),
            estimated_waiting_tokens=None,
        )
        advice = QuotaPolicy(QuotaOptions()).guidance(snapshot)
        assert all((g.admit for g in advice))

    def test_aging_lifts_share_deferral_without_claiming_physical_capacity(self):
        snapshot = quota_view(demand=1000)
        snapshot = replace(
            snapshot,
            requests=tuple((replace(r, waiting_seconds=2) for r in snapshot.requests)),
        )
        assert all(
            (
                g.admit
                for g in QuotaPolicy(QuotaOptions(borrow_idle_capacity=False)).guidance(
                    snapshot
                )
            )
        )


def tool_inputs(pause=10, *, referenced=80, kind="tool_wait"):
    index = LogicalIndex("ns", "epoch")
    record = index.associate(
        Associate(
            session_id="task",
            continuation_id="c",
            expected_revision=0,
            target=TokensTarget(namespace="ns", token_ids=(1, 2, 3, 4)),
        )
    )
    target = SessionTarget(namespace="ns", session_id="task")
    scope = index.resolve(target)
    signal = PolicySignal(
        Signal(target=target, signal=kind, resume_after_seconds=pause),
        scope,
        "owner",
        (record.task,),
    )
    view = PolicyObservation(
        100,
        "ns",
        "epoch",
        (PoolObservation("full", "gpu", 100, 10, referenced, 90 - referenced),),
    )
    return (signal, view)


class TestKVToolPolicy(CustomTestCase):
    def test_observed_cost_increases_pause_threshold_and_unknown_pool_is_not_zero(self):
        signal, view = tool_inputs(1)
        policy = ToolWaitPolicy(ToolWaitOptions())
        (proposal,) = policy.on_signal(signal, view)
        assert isinstance(proposal.action, MoveTo)
        result = SimpleNamespace(
            update=ExecutionUpdate(
                status=Status.COMPLETED,
                quiescent=True,
                completed_steps=("destination_ready",),
            )
        )
        policy.on_result(proposal, result, elapsed_seconds=2)
        assert policy.estimated_save_seconds >= 2
        assert policy.on_signal(signal, view) == ()
        assert (
            ToolWaitPolicy(ToolWaitOptions()).on_signal(signal, replace(view, pools=()))
            == ()
        )


class TestKVPolicyConfig(CustomTestCase):
    def test_policies_are_absent_by_default_and_options_are_typed(self):
        assert parse_config({"namespace": "ns"}).policies == ()
        config = parse_config(
            {
                "namespace": "ns",
                "policies": [
                    {"name": "retention", "options": {"priority": -7, "ttl_seconds": 3}}
                ],
            }
        )
        assert config.policies[0].options.priority == -7
        assert config.policies[0].options.ttl_seconds == 3

    def test_invalid_startup_policy_configuration_is_rejected(self):
        for policies in [
            [{"name": "same"}, {"name": "same"}],
            [{"name": "bad", "implementation": "unavailable"}],
            [{"name": "bad", "options": {"ttl_seconds": 0}}],
        ]:
            with self.subTest(policies=policies):
                with pytest.raises(ValueError):
                    parse_config({"namespace": "ns", "policies": policies})


if __name__ == "__main__":
    unittest.main()
