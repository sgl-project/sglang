# SPDX-License-Identifier: Apache-2.0
"""Tool policy distinguishes resource facts, pause estimates and observed cost."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from sglang.srt.managers.kv_control.contracts import (
    Associate,
    CopyTo,
    ExecutionUpdate,
    MoveTo,
    SessionTarget,
    Signal,
    Status,
    TokensTarget,
)
from sglang.srt.managers.kv_control.index import LogicalIndex
from sglang.srt.managers.kv_control.policy import (
    PolicyObservation,
    PolicySignal,
    PoolObservation,
)
from sglang.srt.managers.kv_control.tool_wait_policy import (
    ToolWaitOptions,
    ToolWaitPolicy,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def inputs(pause=10, *, referenced=80, kind="tool_wait"):
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
    return signal, view


@pytest.mark.parametrize("pause,referenced", [(None, 80), (0.1, 80), (10, 0)])
def test_unknown_short_pause_or_fit_does_not_move(pause, referenced):
    signal, view = inputs(pause, referenced=referenced)
    policy = ToolWaitPolicy(ToolWaitOptions())
    assert policy.on_signal(signal, view) == ()
    resumed = replace(
        signal,
        payload=replace_signal(signal.payload, "resume"),
        previous_scope=signal.scope,
    )
    assert policy.on_signal(resumed, view) == ()


def replace_signal(payload, kind, pause=None):
    import msgspec

    return msgspec.structs.replace(payload, signal=kind, resume_after_seconds=pause)


def test_observed_cost_increases_pause_threshold_and_unknown_pool_is_not_zero():
    signal, view = inputs(1)
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


def test_prediction_uses_prior_frozen_prefix_and_waits_for_admission():
    signal, view = inputs()
    policy = ToolWaitPolicy(ToolWaitOptions(lead_seconds=0.2))
    policy.on_signal(signal, view)
    predicted = replace(
        signal,
        payload=replace_signal(signal.payload, "resume_prediction", 5),
        previous_scope=signal.scope,
    )
    (proposal,) = policy.on_signal(predicted, view)
    assert isinstance(proposal.action, CopyTo)
    assert proposal.not_before == pytest.approx(104.8)
    assert proposal.prepare_for_admission
    assert proposal.scope is signal.scope
    assert proposal.action.target.token_ids == (1, 2, 3, 4)
    assert (
        ToolWaitPolicy(ToolWaitOptions()).on_signal(
            replace(predicted, previous_scope=None), view
        )
        == ()
    )


def test_cleanup_drops_only_finished_owner_context():
    signal, view = inputs(0.1)
    policy = ToolWaitPolicy(ToolWaitOptions())
    policy.on_signal(signal, view)
    policy.cleanup(())
    assert signal.owner_key in policy._kept
    policy.cleanup(signal.tasks)
    assert signal.owner_key not in policy._kept
