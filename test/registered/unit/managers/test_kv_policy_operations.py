# SPDX-License-Identifier: Apache-2.0
"""Registered private policy execution uses the actual bounded control core."""

import msgspec
import test_kv_control as fixtures

from sglang.srt.managers.kv_control import KVControl
from sglang.srt.managers.kv_control.contracts import (
    CopyTo,
    ExecutionUpdate,
    Limits,
    Location,
    SetPriority,
    Status,
)
from sglang.srt.managers.kv_control.registry import snapshot_envelope
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def priority():
    return SetPriority(
        target=fixtures.tokens(),
        location=Location(tier="gpu"),
        claim_id="policy-claim",
        revision=1,
        value=50,
        ttl_seconds=10,
    )


def test_children_wait_for_owner_dispatch_and_cancel_independently():
    backend = fixtures.CompletionBoundary()
    core = KVControl("engine", "ns", enabled=True, executor=backend)
    load = CopyTo(
        target=fixtures.tokens(), src=Location(tier="host"), dst=Location(tier="gpu")
    )
    first, second = core._reserve_policy_operations(
        (("load", load), ("priority", priority()))
    )
    core.progress()
    assert backend.starts == 0
    assert core.registry.get(first).status == Status.QUEUED
    assert core._start_policy_operation(first).status == Status.RUNNING
    assert backend.work[first.operation_id].origin == "policy"
    assert core.cancel(second).status == Status.CANCELLED
    assert backend.starts == 1
    assert core.cancel(first).status == Status.CANCELLING
    assert core.registry.capacity()["active"] == 1
    backend.updates[first.operation_id] = ExecutionUpdate(
        status=Status.CANCELLED, quiescent=True
    )
    core.progress()
    assert core.registry.capacity()["active"] == 0


def test_public_same_caller_does_not_obtain_private_authority():
    backend = fixtures.CompletionBoundary()
    core = KVControl("engine", "ns", enabled=True, executor=backend)
    explicit = fixtures.submit(
        core,
        fixtures.action(
            "set_priority", fixtures.priority(target=fixtures.wire(fixtures.tokens()))
        ),
        context=core.context("local-policy"),
    )
    assert backend.work[explicit.results[0].ref.operation_id].origin == "explicit"
    with fixtures.fails("invalid_payload"):
        core._start_policy_operation(explicit.results[0].ref)


def test_policy_batch_capacity_and_bad_target_fail_before_reservation_or_effects():
    backend = fixtures.CompletionBoundary()
    core = KVControl(
        "engine", "ns", enabled=True, executor=backend, limits=Limits(max_active=1)
    )
    with fixtures.fails("capacity"):
        core._reserve_policy_operations((("a", priority()), ("b", priority())))
    assert core.registry.capacity()["records"] == 0 and backend.starts == 0
    invalid = msgspec.structs.replace(priority(), target="request_cached")
    with fixtures.fails("invalid_payload"):
        core._reserve_policy_operations((("a", priority()), ("b", invalid)))
    assert core.registry.capacity()["records"] == 0


def test_registered_snapshots_survive_index_change_and_finish_cancels_task_children():
    backend = fixtures.CompletionBoundary()
    core = KVControl("engine", "ns", enabled=True, executor=backend)
    record = core.index.associate(fixtures.association())
    selected = fixtures.SessionTarget(
        namespace="ns", session_id="s", continuation_id="c"
    )
    payload = msgspec.structs.replace(priority(), target=selected)
    (ref,) = core._reserve_policy_operations((("a", payload),))
    core.index.associate(
        fixtures.association(
            target=fixtures.tokens((1, 2, 3, 4, 5)), revision=record.revision
        )
    )
    core._start_policy_operation(ref)
    assert backend.work[ref.operation_id].target.branches[0].requested.end_token == 4
    fixtures.submit(
        core,
        fixtures.action(
            "finish_task",
            {"session_id": "s", "expected_revision": core.index.session_revision("s")},
        ),
        message_id="finish",
    )
    assert core.registry.get(ref).status == Status.CANCELLING
    backend.updates[ref.operation_id] = ExecutionUpdate(
        status=Status.CANCELLED, quiescent=True
    )
    core.progress()
    assert core.registry.get(ref).status == Status.CANCELLED


def test_signal_child_refs_are_typed_bounded_and_independent_of_parent_ack():
    core = KVControl(
        "engine", "ns", enabled=True, executor=fixtures.CompletionBoundary()
    )
    batch, _ = snapshot_envelope(fixtures.envelope(fixtures.action()), core.limits)
    parent = core.registry.reserve(batch, core.context("router")).results[0]
    (child,) = core._reserve_policy_operations((("a", priority()),))
    result = core.registry.update(
        parent.ref,
        ExecutionUpdate(
            status=Status.COMPLETED,
            quiescent=True,
            code="policy_context_updated",
            derived_operations=(child,),
        ),
    )
    encoded = msgspec.msgpack.encode(result)
    decoded = msgspec.msgpack.decode(encoded, type=type(result))
    assert decoded.update.derived_operations == (child,)
    core.registry.acknowledge(parent.ref.context, parent.ref.message_id)
    assert core.registry.get(child).status == Status.QUEUED
    assert core.cancel(child).status == Status.CANCELLED
    invalid_parent = core.registry.reserve(
        snapshot_envelope(
            fixtures.envelope(fixtures.action(), message_id="other"), core.limits
        )[0],
        core.context("router"),
    ).results[0]
    with fixtures.fails("capacity"):
        core.registry.update(
            invalid_parent.ref,
            ExecutionUpdate(
                status=Status.COMPLETED,
                quiescent=True,
                derived_operations=(child, child),
            ),
        )
