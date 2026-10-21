# SPDX-License-Identifier: Apache-2.0
"""Soft quota advice uses explicit unknowns, borrowable capacity and bounded state."""

from dataclasses import replace

import pytest

from sglang.srt.managers.kv_control.contracts import (
    ControlError,
    SessionTarget,
    Signal,
    TaskIdentity,
    TokensTarget,
)
from sglang.srt.managers.kv_control.index import LogicalIndex
from sglang.srt.managers.kv_control.policy import (
    PolicyObservation,
    PolicySignal,
    PoolObservation,
    RequestObservation,
)
from sglang.srt.managers.kv_control.policy_runtime import QuotaOptions, QuotaPolicy
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
A, B = (
    TaskIdentity(session_id="A", incarnation=1),
    TaskIdentity(session_id="B", incarnation=2),
)


def view(*, complete=True, demand=40):
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


def signal(task, *, importance=80, kind="importance"):
    payload = Signal(
        target=SessionTarget(namespace="ns", session_id=task.session_id),
        signal=kind,
        importance=importance,
    )
    return PolicySignal(
        payload,
        LogicalIndex("ns", "epoch").resolve(
            TokensTarget(namespace="ns", token_ids=(1,))
        ),
        "owner",
        (task,),
    )


def test_soft_share_defers_busy_session_but_idle_capacity_is_borrowable():
    assert [
        g.admit
        for g in QuotaPolicy(QuotaOptions(borrow_idle_capacity=False)).guidance(view())
    ] == [False, True]
    assert [g.admit for g in QuotaPolicy(QuotaOptions()).guidance(view())] == [
        True,
        True,
    ]


@pytest.mark.parametrize("unknown", ["truncated", "demand", "growth"])
def test_missing_observation_is_native_fallback_not_zero_demand(unknown):
    snapshot = view()
    if unknown == "truncated":
        snapshot = replace(
            snapshot, requests_complete=False, estimated_waiting_tokens=None
        )
    elif unknown == "demand":
        snapshot = replace(
            snapshot,
            requests=(
                snapshot.requests[0],
                replace(snapshot.requests[1], estimated_incremental_tokens=None),
                snapshot.requests[2],
            ),
            estimated_waiting_tokens=None,
        )
    else:
        snapshot = replace(
            snapshot,
            requests=(
                replace(snapshot.requests[0], estimated_running_growth=None),
                *snapshot.requests[1:],
            ),
            requests_complete=False,
        )
    advice = QuotaPolicy(QuotaOptions()).guidance(snapshot)
    assert all(g.admit for g in advice)


def test_aging_lifts_share_deferral_without_claiming_physical_capacity():
    snapshot = view(demand=1000)
    snapshot = replace(
        snapshot,
        requests=tuple(replace(r, waiting_seconds=2) for r in snapshot.requests),
    )
    assert all(
        g.admit
        for g in QuotaPolicy(QuotaOptions(borrow_idle_capacity=False)).guidance(
            snapshot
        )
    )


def test_importance_expiry_cleanup_and_incarnation_do_not_cross_task_lifetimes():
    policy = QuotaPolicy(QuotaOptions(importance_ttl=1, borrow_idle_capacity=False))
    policy.on_signal(signal(A), view())
    before = policy.guidance(view())[0].score
    policy.cleanup((B,))
    assert policy.guidance(view())[0].score == before
    policy.cleanup((A,))
    assert policy.guidance(view())[0].score < before
    policy.on_signal(signal(A), view())
    changed = replace(
        view(),
        requests=(
            view().requests[0],
            replace(
                view().requests[1], task=TaskIdentity(session_id="A", incarnation=3)
            ),
            view().requests[2],
        ),
    )
    assert policy.guidance(changed)[0].score < before
    assert policy.guidance(replace(view(), now=12))[0].score < before


def test_state_limit_refuses_new_owner_then_expiry_makes_room():
    policy = QuotaPolicy(QuotaOptions(max_states=1, importance_ttl=1))
    policy.on_signal(signal(A), view())
    with pytest.raises(ControlError, match="capacity"):
        policy.on_signal(signal(B), view())
    policy.on_signal(signal(B), replace(view(), now=12))
    assert set(policy._importance) == {B}
