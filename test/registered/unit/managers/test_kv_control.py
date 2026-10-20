# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for programmable KV; no model, allocator or backend imports.

Uses the registered pure-pytest pattern documented in test/README.md. The
controlled executor is only the external completion boundary; all correlation,
resolution, mutation, cancellation and resource-retirement gating is real core
code. These tests make no cache movement or serving claim.
"""

import copy
import sys
from contextlib import contextmanager

import msgspec
import pytest

from sglang.srt.managers.kv_control import KVControl, LogicalIndex, OperationRegistry
from sglang.srt.managers.kv_control.contracts import (
    Associate,
    ContentSpan,
    ContentTarget,
    ControlContext,
    ControlError,
    ExecutionUpdate,
    Limits,
    Location,
    ParentPrefix,
    SessionTarget,
    Status,
    TargetOutcome,
    TaskReference,
    TokenRange,
    TokensTarget,
    parse_action,
)
from sglang.srt.managers.kv_control.registry import snapshot_envelope
from sglang.srt.managers.kv_hints import KvHintAction, KvHintsEnvelope
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class Clock:
    now = 100.0

    def __call__(self):
        return self.now


class CompletionBoundary:
    """A controlled external executor, without any pretend cache implementation."""

    def __init__(self):
        self.work = {}
        self.updates = {}
        self.starts = 0

    def start(self, work):
        self.starts += 1
        self.work[work.ref.operation_id] = work
        return ExecutionUpdate(status=Status.RUNNING, quiescent=False)

    def poll(self, ref):
        return self.updates.pop(ref.operation_id, None)

    def cancel(self, ref):
        return None


def action(kind="signal", payload=None, *, action_id="a", version="1.0"):
    return KvHintAction(
        action_id=action_id,
        action_type=f"sglang.{kind}",
        action_version=version,
        payload=payload if payload is not None else signal(),
    )


def envelope(*actions, message_id="m", protocol="0.1"):
    return KvHintsEnvelope(
        protocol_version=protocol, message_id=message_id, actions=list(actions)
    )


def tokens(values=(1, 2, 3, 4), **kwargs):
    return TokensTarget(namespace="ns", token_ids=tuple(values), **kwargs)


def signal():
    return {
        "target": {"kind": "session", "namespace": "ns", "session_id": "untracked"},
        "signal": "tool_wait",
    }


def span(name="h", start=0, end=4, component="full", dependency_start=0):
    return ContentSpan(
        content_id=name,
        component=component,
        start_token=start,
        end_token=end,
        dependency_start=dependency_start,
    )


def content(*spans, length=8, components=("full",), **kwargs):
    return ContentTarget(
        namespace="ns", token_count=length, components=components, spans=spans, **kwargs
    )


def association(target=None, session="s", continuation="c", revision=0, **kwargs):
    return Associate(
        session_id=session,
        continuation_id=continuation,
        expected_revision=revision,
        target=target if target is not None else tokens(),
        **kwargs,
    )


def movement(target=None, *, src="storage", dst="gpu"):
    return {
        "target": wire(target if target is not None else tokens()),
        "src": {"tier": src},
        "dst": {"tier": dst},
    }


def priority(**changes):
    result = {
        "target": "request_cached",
        "location": {"tier": "gpu"},
        "claim_id": "router/claim",
        "revision": 1,
        "value": -20,
        "ttl_seconds": 10,
    }
    result.update(changes)
    return result


def submit(core, *actions, message_id="m", context=None, protocol="0.1"):
    return core.submit(
        envelope(*actions, message_id=message_id, protocol=protocol),
        context or core.context("router"),
    )


def wire(value):
    return msgspec.json.decode(msgspec.json.encode(value))


@contextmanager
def fails(code):
    with pytest.raises(ControlError) as caught:
        yield
    assert caught.value.code == code


def test_mixed_owners_errors_and_replay_are_independent():
    core = KVControl("engine", "ns", enabled=True)
    opaque_payload = {"backend_owns": object()}
    opaque = KvHintAction(
        action_id="a",
        action_type="backend.custom",
        action_version="9",
        payload=opaque_payload,
    )
    invalid = action("set_priority", priority(value=True), action_id="bad")
    unknown = action("pin", {}, action_id="unsupported")
    assoc = action("associate", wire(association()))
    original = envelope(opaque, invalid, unknown, assoc)
    result = core.submit(original, core.context("router"))
    assert [r.status for r in result.results] == [
        Status.REJECTED,
        Status.UNSUPPORTED,
        Status.COMPLETED,
    ]
    assert result.unhandled_actions == (opaque,)
    assert result.unhandled_actions[0].payload is opaque_payload
    record = core.index.get("s", "c")
    opaque_payload["new"] = object()
    replay = core.submit(original, core.context("router"))
    assert replay.replayed and replay.results == result.results
    assert core.index.get("s", "c") == record
    assert core.index.capacity()["continuations"] == 1


def test_unconsumed_signal_does_not_resolve_or_allocate_index_state():
    core = KVControl("engine", "ns", enabled=True)
    result = submit(core, action())
    assert result.results[0].status == Status.UNCONSUMED
    assert core.index.capacity() == {
        "sessions": 0,
        "continuations": 0,
        "spans": 0,
        "tokens": 0,
    }
    assert core.registry.capacity()["active"] == 0
    assert core.index.generation == 0


def test_disabled_default_and_absent_executor_are_honest():
    disabled = KVControl("engine", "ns")
    assert (
        submit(disabled, action("associate", wire(association()))).results[0].status
        == Status.UNSUPPORTED
    )
    assert disabled.index.generation == 0
    core = KVControl("engine", "ns", enabled=True)
    result = submit(core, action("copy_to", movement()))
    assert result.results[0].status == Status.UNSUPPORTED
    assert not result.results[0].update.target_outcomes
    assert not result.results[0].update.completed_steps


@pytest.mark.parametrize(
    "changes",
    [
        {"value": True},
        {"value": 1.0},
        {"value": 1 << 31},
        {"revision": -1},
        {"revision": True},
        {"revision": 1 << 54},
        {"ttl_seconds": 0},
        {"ttl_seconds": -1},
        {"ttl_seconds": float("inf")},
        {"ttl_seconds": float("nan")},
        {"ttl_seconds": True},
        {"ttl_seconds": None},
        {"value": None},
        {"claim_id": ""},
        {"claim_id": "x" * 257},
        {"unexpected": "field"},
    ],
)
def test_invalid_native_numbers_and_fields_do_not_suppress_valid_sibling(changes):
    core = KVControl("engine", "ns", enabled=True)
    bad = action("set_priority", priority(**changes), action_id="bad")
    result = submit(core, bad, action())
    assert [r.status for r in result.results] == [Status.REJECTED, Status.UNCONSUMED]
    assert result.results[0].update.code == "invalid_payload"
    assert submit(core, bad, action()).replayed


def test_priority_clear_signed_value_and_msgspec_transport_round_trip():
    parsed = parse_action(action("set_priority", priority()))
    assert parsed.value == -20 and parsed.ttl_seconds == 10
    clear = parse_action(
        action("set_priority", priority(value=None, ttl_seconds=None, revision=2))
    )
    assert clear.value is None and clear.ttl_seconds is None
    result = submit(KVControl("engine", "ns", enabled=True), action())
    decoded = msgspec.json.decode(msgspec.json.encode(result), type=type(result))
    assert decoded == result


@pytest.mark.parametrize("protocol,version", [("future", "1.0"), ("0.1", "2.0")])
def test_unsupported_versions_and_unknown_actions_are_retained(protocol, version):
    core = KVControl("engine", "ns", enabled=True)
    result = submit(core, action(version=version), protocol=protocol)
    assert result.results[0].status == Status.UNSUPPORTED
    assert (
        submit(core, action(version=version), protocol=protocol).results
        == result.results
    )


def test_opaque_only_envelope_is_untouched_even_with_unknown_protocol():
    core = KVControl("engine", "ns")
    opaque = KvHintAction(
        action_id="x",
        action_type="backend.x",
        action_version="9",
        payload={"opaque": object()},
    )
    result = submit(core, opaque, protocol="opaque-version")
    assert result.unhandled_actions == (opaque,) and result.results == ()
    assert core.registry.capacity() == {"active": 0, "records": 0, "messages": 0}


def test_snapshot_detaches_mutable_input_and_preserves_conflict_identity():
    boundary = CompletionBoundary()
    core = KVControl("engine", "ns", enabled=True, executor=boundary)
    payload = movement(src="gpu", dst="host")
    original_payload = copy.deepcopy(payload)
    context = core.context("router", binding_id="request-1/sample-0")
    result = submit(core, action("copy_to", payload), context=context)
    ref = result.results[0].ref
    payload["target"]["token_ids"][0] = 999
    with fails("conflict"):
        submit(core, action("copy_to", payload), context=context)
    assert submit(core, action("copy_to", original_payload), context=context).replayed
    core.request_event(context, "success")
    assert boundary.work[ref.operation_id].target.branches[0].content.token_ids == (
        1,
        2,
        3,
        4,
    )
    assert core.registry.get(ref).status == Status.RUNNING
    assert boundary.starts == 1


def test_context_separates_fanout_and_callers_and_rejects_stale_generations():
    core = KVControl("engine", "ns", enabled=True)
    contexts = [
        core.context("router", binding_id=f"request/{sample}") for sample in (0, 1)
    ]
    contexts.append(core.context("other-router", binding_id="request/0"))
    refs = [submit(core, action(), context=ctx).results[0].ref for ctx in contexts]
    assert len({ref.operation_id for ref in refs}) == 3
    assert all(submit(core, action(), context=ctx).replayed for ctx in contexts)
    for field, value, code in [
        ("engine_id", "other", "wrong_engine"),
        ("engine_epoch", "old", "stale_epoch"),
        ("namespace", "weights-v2", "incompatible_namespace"),
    ]:
        with fails(code):
            submit(
                core,
                action(),
                context=msgspec.structs.replace(contexts[0], **{field: value}),
            )
    assert core.registry.capacity()["records"] == 3


@pytest.mark.parametrize(
    "limits,actions",
    [
        (Limits(max_actions=1), [action(action_id="1"), action(action_id="2")]),
        (Limits(max_envelope_bytes=120), [action()]),
        (Limits(max_json_nodes=2), [action()]),
        (Limits(max_active=1), [action(action_id="1"), action(action_id="2")]),
        (Limits(max_records=1), [action(action_id="1"), action(action_id="2")]),
    ],
)
def test_preflight_limits_do_not_partially_accept_or_mutate(limits, actions):
    core = KVControl("engine", "ns", enabled=True, limits=limits)
    with fails("capacity"):
        submit(core, *actions)
    assert core.registry.capacity() == {"active": 0, "records": 0, "messages": 0}
    assert core.index.generation == 0


def test_invalid_wrapper_is_rejected_before_valid_sibling_side_effects():
    core = KVControl("engine", "ns", enabled=True)
    assoc = action("associate", wire(association()))
    with fails("invalid_envelope"):
        submit(core, assoc, action())  # Duplicate native action ID.
    with fails("invalid_envelope"):
        submit(core, assoc, action(payload={"not_json": object()}, action_id="bad"))
    assert core.index.generation == 0 and core.registry.capacity()["messages"] == 0


def test_ack_retains_failure_and_replay_deadline_without_renewal():
    clock = Clock()
    core = KVControl(
        "engine", "ns", enabled=True, limits=Limits(retention_seconds=10), clock=clock
    )
    context = core.context("router")
    original = submit(core, action("copy_to", movement()), context=context)
    assert original.results[0].status == Status.UNSUPPORTED
    deadline = original.replay_expires_at
    clock.now += 4
    ack = core.registry.acknowledge(context, "m")
    assert ack.summaries[0].status == Status.UNSUPPORTED
    assert ack.summaries[0].reason == original.results[0].update.reason
    assert ack.replay_expires_at == deadline == 110
    assert core.registry.capacity() == {"active": 0, "records": 0, "messages": 1}
    replay = submit(core, action("copy_to", movement()), context=context)
    assert replay.replayed and replay.acknowledged and replay.summaries == ack.summaries
    with fails("conflict"):
        submit(core, action())
    clock.now = 109.999
    assert core.registry.acknowledge(context, "m") == ack
    clock.now = 110
    new = submit(core, action("copy_to", movement()), context=context)
    assert not new.replayed
    assert new.results[0].ref.operation_id != original.results[0].ref.operation_id


def test_replay_slots_are_reserved_before_effects_and_ack_never_needs_extra_slot():
    core = KVControl("engine", "ns", enabled=True, limits=Limits(max_messages=1))
    submit(core, action())
    core.registry.acknowledge(core.context("router"), "m")
    with fails("capacity"):
        submit(core, action("associate", wire(association())), message_id="new")
    assert core.index.generation == 0
    assert submit(core, action()).acknowledged


def test_active_work_does_not_expire_and_ack_does_not_cancel_it():
    clock, boundary = Clock(), CompletionBoundary()
    core = KVControl(
        "engine",
        "ns",
        enabled=True,
        executor=boundary,
        limits=Limits(retention_seconds=1, max_records=1),
        clock=clock,
    )
    result = submit(core, action("copy_to", movement())).results[0]
    clock.now += 1000
    assert core.registry.get(result.ref).status == Status.RUNNING
    with fails("busy"):
        core.registry.acknowledge(core.context("router"), "m")
    with fails("capacity"):
        submit(core, action(), message_id="new")
    assert core.registry.capacity()["active"] == 1


def test_cancellation_requires_incarnation_and_drained_worker_access():
    boundary = CompletionBoundary()
    core = KVControl("engine", "ns", enabled=True, executor=boundary)
    result = submit(core, action("copy_to", movement())).results[0]
    with fails("stale_operation"):
        core.cancel(msgspec.structs.replace(result.ref, operation_id="older-instance"))
    assert core.registry.get(result.ref).status == Status.RUNNING
    assert core.cancel(result.ref).status == Status.CANCELLING
    with fails("busy"):
        core.reset("weights-v2")
    with fails("busy"):
        core.registry.update(
            result.ref, ExecutionUpdate(status=Status.CANCELLED, quiescent=False)
        )
    boundary.updates[result.ref.operation_id] = ExecutionUpdate(
        status=Status.CANCELLED, quiescent=True
    )
    assert core.progress()[0].status == Status.CANCELLED
    assert core.registry.capacity()["active"] == 0
    assert core.cancel(result.ref).status == Status.CANCELLED
    with fails("terminal"):
        core.registry.update(
            result.ref, ExecutionUpdate(status=Status.COMPLETED, quiescent=True)
        )


def test_late_cancel_after_expiry_cannot_cancel_reused_message():
    clock, boundary = Clock(), CompletionBoundary()
    core = KVControl(
        "engine",
        "ns",
        enabled=True,
        executor=boundary,
        limits=Limits(retention_seconds=1),
        clock=clock,
    )
    context = core.context("router", binding_id="r")
    first = submit(core, action("set_priority", priority()), context=context).results[0]
    assert core.cancel(first.ref).status == Status.CANCELLED
    clock.now += 1
    second = submit(core, action("set_priority", priority()), context=context).results[
        0
    ]
    with fails("stale_operation"):
        core.cancel(first.ref)
    assert core.registry.get(second.ref).status == Status.QUEUED
    assert boundary.starts == 0


def test_request_cached_binds_only_success_and_checkpoint_never_reexecutes():
    boundary = CompletionBoundary()
    core = KVControl("engine", "ns", enabled=True, executor=boundary)
    context = core.context("router", binding_id="r")
    result = submit(core, action("set_priority", priority()), context=context).results[
        0
    ]
    assert result.status == Status.QUEUED
    assert core.request_event(context, "checkpoint", committed=tokens()) == ()
    assert core.request_event(context, "preempt") == ()
    assert boundary.starts == 0
    core.request_event(context, "success", committed=tokens((1, 2, 3, 4, 5, 6)))
    assert (
        boundary.work[result.ref.operation_id].target.branches[0].requested.end_token
        == 6
    )
    assert core.request_event(context, "success", committed=tokens((9,))) == ()
    assert submit(core, action("set_priority", priority()), context=context).replayed
    assert boundary.starts == 1


@pytest.mark.parametrize("event", ["abort", "error"])
def test_unsuccessful_request_does_not_dispatch(event):
    boundary = CompletionBoundary()
    core = KVControl("engine", "ns", enabled=True, executor=boundary)
    context = core.context("router", binding_id="r")
    result = submit(core, action("set_priority", priority()), context=context).results[
        0
    ]
    assert core.request_event(context, event)[0].status == Status.CANCELLED
    assert boundary.starts == 0 and core.registry.capacity()["active"] == 0
    assert core.registry.get(result.ref).update.code == "request_failed"


def test_historical_snapshot_does_not_grow_or_dangle_after_task_finish():
    boundary = CompletionBoundary()
    core = KVControl("engine", "ns", enabled=True, executor=boundary)
    old = core.index.associate(association())
    target = SessionTarget(
        namespace="ns",
        session_id="s",
        continuation_id="c",
        expected_revision=old.revision,
    )
    context = core.context("router", binding_id="r")
    queued = submit(
        core,
        action("copy_to", movement(target, src="gpu", dst="host")),
        context=context,
    ).results[0]
    core.index.associate(association(tokens((1, 2, 3, 4, 5, 6)), revision=old.revision))
    core.index.finish_task("s", expected_revision=core.index.session_revision("s"))
    assert core.index.capacity()["sessions"] == 0
    core.request_event(context, "success")
    snapshot = boundary.work[queued.ref.operation_id].target
    assert snapshot.branches[0].content.token_ids == (1, 2, 3, 4)
    assert snapshot.branches[0].continuation.revision == old.revision


def test_sibling_order_does_not_create_implicit_dependency():
    boundary = CompletionBoundary()
    core = KVControl("engine", "ns", enabled=True, executor=boundary)
    target = SessionTarget(namespace="ns", session_id="s", continuation_id="c")
    result = submit(
        core,
        action("associate", wire(association())),
        action("copy_to", movement(target), action_id="copy"),
    )
    assert [r.status for r in result.results] == [Status.COMPLETED, Status.MISSING]
    assert boundary.starts == 0


def test_branch_uses_explicit_parent_boundary_and_different_suffix():
    index = LogicalIndex("ns", "epoch")
    parent = index.associate(association(tokens((1, 2, 3, 4)), session="parent"))
    prefix = ParentPrefix(
        session_id="parent",
        continuation_id="c",
        expected_revision=parent.revision,
        end_token=2,
    )
    child = index.associate(
        association(tokens((1, 2, 90)), session="child", parent_prefix=prefix)
    )
    index.associate(
        association(tokens((1, 2, 3, 4, 5)), session="parent", revision=parent.revision)
    )
    grown = index.associate(
        association(tokens((1, 2, 90, 91)), session="child", revision=child.revision)
    )
    assert grown.parent_prefix == prefix and grown.incarnation == child.incarnation
    assert grown.content.token_ids == (1, 2, 90, 91)
    with fails("conflict"):
        index.associate(
            association(
                tokens((1, 8, 90, 91, 92)), session="child", revision=grown.revision
            )
        )
    assert index.get("child", "c") == grown


def test_task_tree_cleanup_preserves_external_content_and_active_snapshots():
    index = LogicalIndex("ns", "epoch")
    shared = span()
    root = index.associate(association(content(shared, length=4), session="root"))
    child = index.associate(
        association(
            content(shared, span("b", 4, 8)),
            session="child",
            task_parent=TaskReference(
                session_id="root", expected_revision=index.session_revision("root")
            ),
            parent_prefix=ParentPrefix(
                session_id="root",
                continuation_id="c",
                expected_revision=root.revision,
                end_token=4,
            ),
        )
    )
    external = index.associate(
        association(content(shared, length=4), session="external")
    )
    snapshot = index.resolve(SessionTarget(namespace="ns", session_id="child"))
    with fails("stale_revision"):
        index.finish_task("root", expected_revision=root.revision)
    cleanup = index.finish_task(
        "root", expected_revision=index.session_revision("root")
    )
    assert {task.session_id for task in cleanup.removed_tasks} == {"root", "child"}
    assert index.members(shared) == (("external", "c"),)
    assert index.get("external", "c") == external
    assert snapshot.branches[0].continuation == child
    recreated = index.associate(association(content(shared, length=4), session="root"))
    assert recreated.incarnation != root.incarnation
    with fails("stale_revision"):
        index.finish_task("root", expected_revision=root.revision)


@pytest.mark.parametrize(
    "limits",
    [Limits(max_sessions=1), Limits(max_continuations=1), Limits(max_index_tokens=4)],
)
def test_index_limits_leave_all_state_and_revisions_intact(limits):
    index = LogicalIndex("ns", "epoch", limits=limits)
    first = index.associate(association())
    before = index.capacity(), index.generation, index.query()
    with fails("capacity"):
        index.associate(association(session="other"))
    assert (index.capacity(), index.generation, index.query()) == before
    assert index.get("s", "c") == first


def test_descriptor_growth_and_failed_parent_mutation_are_atomic():
    index = LogicalIndex("ns", "epoch", limits=Limits(max_index_spans=2))
    first = index.associate(association(content(span(), length=4)))
    for target, code in [
        (content(span("different"), length=4), "conflict"),
        (content(span(), span("b", 4, 8), span("c", 8, 12), length=12), "capacity"),
    ]:
        with fails(code):
            index.associate(association(target, revision=first.revision))
        assert index.get("s", "c") == first and index.members(span()) == (("s", "c"),)
    with fails("unaligned"):
        index.associate(
            association(
                content(span(), length=4),
                session="branch",
                parent_prefix=ParentPrefix(
                    session_id="s",
                    continuation_id="c",
                    expected_revision=first.revision,
                    end_token=3,
                ),
            )
        )
    assert index.capacity()["sessions"] == 1


def test_range_resolution_reports_missing_components_and_never_extends_coverage():
    index = LogicalIndex("ns", "epoch")
    target = content(
        span(),
        span("tail", 8, 12),
        length=16,
        components=("full", "mamba"),
        start_token=2,
        end_token=10,
    )
    branch = index.resolve(target).branches[0]
    assert not branch.spans
    assert [(s.start_token, s.end_token) for s in branch.unaligned] == [(0, 4), (8, 12)]
    assert [
        (g.coverage.component, g.coverage.start_token, g.coverage.end_token)
        for g in branch.missing
    ] == [("full", 4, 8), ("mamba", 2, 10)]
    suffix = index.resolve(
        content(span(), span("suffix", 4, 8), start_token=4, end_token=8)
    ).branches[0]
    assert suffix.spans == (span("suffix", 4, 8),)
    assert suffix.dependencies == (
        TokenRange(component="full", start_token=0, end_token=4),
    )
    assert suffix.requested == TokenRange(start_token=4, end_token=8)


def test_exact_unaligned_ranges_are_unsupported_and_clip_is_explicit():
    boundary = CompletionBoundary()
    core = KVControl("engine", "ns", enabled=True, executor=boundary)
    target = content(span(), span("b", 4, 8), end_token=7)
    result = submit(core, action("copy_to", movement(target)))
    assert result.results[0].status == Status.UNSUPPORTED and boundary.starts == 0
    clipped = msgspec.structs.replace(target, allow_clip=True)
    result = submit(core, action("copy_to", movement(clipped)), message_id="clip")
    work = boundary.work[result.results[0].ref.operation_id]
    assert work.target.branches[0].spans == (span(),)
    assert work.target.branches[0].unaligned == (span("b", 4, 8),)


def test_session_wide_snapshot_covers_all_branches_and_scope_limit_is_atomic():
    index = LogicalIndex(
        "ns", "epoch", limits=Limits(max_scope_tokens=8, max_branches=2)
    )
    index.associate(association(continuation="b"))
    index.associate(association(tokens((1, 2, 9)), continuation="c"))
    target = SessionTarget(namespace="ns", session_id="s")
    snapshot = index.resolve(target)
    assert {b.continuation.continuation_id for b in snapshot.branches} == {"b", "c"}
    index.associate(association(tokens((1,)), continuation="d"))
    with fails("capacity"):
        index.resolve(target)
    assert len(snapshot.branches) == 2


def test_query_cursor_binds_epoch_namespace_generation_and_filter():
    index = LogicalIndex("ns", "epoch")
    for name in ("a", "b", "c"):
        index.associate(association(session=name))
    first = index.query(limit=1)
    second = index.query(limit=1, cursor=first.next_cursor)
    assert [first.items[0].task.session_id, second.items[0].task.session_id] == [
        "a",
        "b",
    ]
    with fails("invalid_cursor"):
        index.query(session_id="a", cursor=first.next_cursor)
    other = LogicalIndex("ns", "new-epoch")
    with fails("stale_epoch"):
        other.query(cursor=first.next_cursor)
    original = index.get("a", "c")
    index.associate(
        association(tokens((1, 2, 3, 4, 5)), session="a", revision=original.revision)
    )
    with fails("stale_revision"):
        index.query(cursor=first.next_cursor)


def test_pruning_does_not_reuse_revision_or_invalidate_snapshot():
    index = LogicalIndex("ns", "epoch")
    old = index.associate(association())
    snapshot = index.resolve(
        SessionTarget(namespace="ns", session_id="s", continuation_id="c")
    )
    index.prune("s", "c", expected_revision=old.revision)
    with fails("missing"):
        index.resolve(SessionTarget(namespace="ns", session_id="s"))
    new = index.associate(association(tokens((9, 8))))
    assert new.revision > old.revision and new.incarnation != old.incarnation
    with fails("stale_revision"):
        index.associate(association(tokens((1, 2, 3, 4, 5)), revision=old.revision))
    assert snapshot.branches[0].content.token_ids == (1, 2, 3, 4)


def test_namespace_boundaries_and_range_schema_fail_without_mutation():
    index = LogicalIndex("ns", "epoch")
    with fails("incompatible_namespace"):
        index.associate(
            association(msgspec.structs.replace(tokens(), namespace="other"))
        )
    with fails("incompatible_namespace"):
        index.resolve(msgspec.structs.replace(tokens(), namespace="other"))
    for target in [
        tokens(start_token=2, end_token=2),
        tokens(start_token=-1),
        content(span(end=6), span("overlap", 4, 8)),
        content(span(component="unknown")),
        content(span(dependency_start=1)),
    ]:
        with fails("invalid_payload"):
            index.resolve(target)
    assert index.generation == 0


def test_partial_move_stages_survive_ack_and_terminal_results_cannot_be_rewritten():
    boundary = CompletionBoundary()
    core = KVControl("engine", "ns", enabled=True, executor=boundary)
    result = submit(
        core,
        action(
            "move_to", movement(content(span(), length=4), src="gpu", dst="storage")
        ),
    ).results[0]
    update = ExecutionUpdate(
        status=Status.PARTIAL,
        quiescent=True,
        code="source_referenced",
        completed_steps=("destination_ready",),
        target_outcomes=(
            TargetOutcome(
                coverage=TokenRange(component="full", start_token=0, end_token=4),
                content_id="h",
                location=Location(tier="storage"),
                status=Status.PARTIAL,
                completed_steps=("destination_ready",),
                code="source_referenced",
            ),
        ),
    )
    boundary.updates[result.ref.operation_id] = update
    terminal = core.progress()[0]
    assert terminal.status == Status.PARTIAL and terminal.update == update
    with fails("terminal"):
        core.registry.update(
            result.ref, msgspec.structs.replace(update, status=Status.COMPLETED)
        )
    ack = core.registry.acknowledge(core.context("router"), "m")
    assert (
        ack.summaries[0].status == Status.PARTIAL
        and ack.summaries[0].code == "source_referenced"
    )


def test_executor_cannot_publish_empty_success_or_retire_on_exception():
    class BrokenBoundary(CompletionBoundary):
        def start(self, work):
            super().start(work)
            return ExecutionUpdate(status=Status.COMPLETED, quiescent=True)

    boundary = BrokenBoundary()
    core = KVControl("engine", "ns", enabled=True, executor=boundary)
    result = submit(core, action("copy_to", movement())).results[0]
    assert result.status == Status.CANCELLING and result.update.code == "executor_error"
    assert core.registry.capacity()["active"] == 1
    with fails("busy"):
        core.registry.acknowledge(core.context("router"), "m")
    boundary.updates[result.ref.operation_id] = ExecutionUpdate(
        status=Status.CANCELLED, quiescent=True
    )
    failed = core.progress()[0]
    assert failed.status == Status.FAILED
    assert failed.update.code == result.update.code
    assert failed.update.reason == result.update.reason


def test_result_size_and_transition_validation_are_atomic():
    registry = OperationRegistry("engine", "ns", limits=Limits(max_scope_spans=1))
    context = ControlContext(
        engine_id="engine",
        namespace="ns",
        engine_epoch=registry.engine_epoch,
        caller_id="router",
    )
    batch, _ = snapshot_envelope(envelope(action()), registry.limits)
    original = registry.reserve(batch, context).results[0]
    with fails("transition"):
        registry.update(
            original.ref, ExecutionUpdate(status=Status.RUNNING, quiescent=False)
        )
    outcome = TargetOutcome(
        coverage=TokenRange(start_token=0, end_token=1), status=Status.MISSING
    )
    with fails("capacity"):
        registry.update(
            original.ref,
            ExecutionUpdate(
                status=Status.MISSING,
                quiescent=True,
                target_outcomes=(outcome, outcome),
            ),
        )
    assert registry.get(original.ref) == original
    assert registry.capacity()["active"] == 1


def test_reset_isolates_old_intent_results_and_cursor_without_external_mutation():
    core = KVControl("engine", "ns", enabled=True)
    context = core.context("router")
    old = submit(core, action(), context=context).results[0]
    core.index.associate(association())
    core.reset("new-weights")
    with fails("stale_epoch"):
        core.registry.get(old.ref)
    with fails("stale_epoch"):
        core.submit(envelope(action()), context)
    assert core.index.capacity()["sessions"] == 0
    assert core.context("router").namespace == "new-weights"


def completed_outcome(
    *,
    branch=0,
    component="full",
    content_id="h",
    end=4,
    tier="gpu",
    steps=("destination_ready",),
):
    return TargetOutcome(
        branch_index=branch,
        coverage=TokenRange(component=component, start_token=0, end_token=end),
        content_id=content_id,
        location=Location(tier=tier),
        status=Status.COMPLETED,
        completed_steps=steps,
    )


def test_complete_copy_requires_every_component_and_snapshot_branch():
    boundary = CompletionBoundary()
    core = KVControl("engine", "ns", enabled=True, executor=boundary)
    target = content(
        span(), span("state", component="mamba"), length=4, components=("full", "mamba")
    )
    first = submit(core, action("copy_to", movement(target))).results[0]
    boundary.updates[first.ref.operation_id] = ExecutionUpdate(
        status=Status.COMPLETED,
        quiescent=True,
        completed_steps=("destination_ready",),
        target_outcomes=(completed_outcome(),),
    )
    assert core.progress()[0].status == Status.CANCELLING
    assert "omits required" in core.registry.get(first.ref).update.reason
    # A different token branch cannot borrow a same-position result from branch 0.
    core.index.associate(association(continuation="a"))
    core.index.associate(association(tokens((9, 8, 7, 6)), continuation="b"))
    second = submit(
        core,
        action("copy_to", movement(SessionTarget(namespace="ns", session_id="s"))),
        message_id="branches",
    ).results[0]
    boundary.updates[second.ref.operation_id] = ExecutionUpdate(
        status=Status.COMPLETED,
        quiescent=True,
        completed_steps=("destination_ready",),
        target_outcomes=(completed_outcome(),),
    )
    core.progress()
    assert core.registry.get(second.ref).status == Status.CANCELLING


@pytest.mark.parametrize(
    "outcome,steps",
    [
        (completed_outcome(end=8), ("destination_ready",)),
        (completed_outcome(tier="host"), ("destination_ready",)),
        (completed_outcome(content_id="wrong-content"), ("destination_ready",)),
        (completed_outcome(branch=1), ("destination_ready",)),
        (completed_outcome(steps=("source_reclaimed",)), ("source_reclaimed",)),
    ],
)
def test_invalid_execution_reports_do_not_publish_success(outcome, steps):
    boundary = CompletionBoundary()
    core = KVControl("engine", "ns", enabled=True, executor=boundary)
    result = submit(
        core, action("copy_to", movement(content(span(), length=4)))
    ).results[0]
    boundary.updates[result.ref.operation_id] = ExecutionUpdate(
        status=Status.COMPLETED,
        quiescent=True,
        completed_steps=steps,
        target_outcomes=(outcome,),
    )
    assert core.progress()[0].status == Status.CANCELLING
    assert core.registry.capacity()["active"] == 1


def test_completion_and_cancellation_remain_independent_for_same_content():
    boundary = CompletionBoundary()
    core = KVControl("engine", "ns", enabled=True, executor=boundary)
    first = submit(
        core, action("copy_to", movement(content(span(), length=4))), message_id="one"
    ).results[0]
    second = submit(
        core, action("copy_to", movement(content(span(), length=4))), message_id="two"
    ).results[0]
    assert core.cancel(first.ref).status == Status.CANCELLING
    assert core.registry.get(second.ref).status == Status.RUNNING
    boundary.updates[first.ref.operation_id] = ExecutionUpdate(
        status=Status.CANCELLED, quiescent=True
    )
    boundary.updates[second.ref.operation_id] = ExecutionUpdate(
        status=Status.COMPLETED,
        quiescent=True,
        completed_steps=("destination_ready",),
        target_outcomes=(completed_outcome(),),
    )
    core.progress()
    assert core.registry.get(first.ref).status == Status.CANCELLED
    assert core.registry.get(second.ref).status == Status.COMPLETED
    assert core.registry.capacity()["active"] == 0


def test_executor_exception_still_delivers_cancellation_and_progress_is_fair():
    class NeedsCancellation(CompletionBoundary):
        def start(self, work):
            super().start(work)
            raise RuntimeError("dispatch acknowledgement lost")

        def cancel(self, ref):
            return ExecutionUpdate(status=Status.CANCELLED, quiescent=True)

    boundary = NeedsCancellation()
    core = KVControl("engine", "ns", enabled=True, executor=boundary)
    refs = [
        submit(core, action("copy_to", movement()), message_id=str(n)).results[0].ref
        for n in range(3)
    ]
    assert core.registry.capacity()["active"] == 3
    for remaining in (2, 1, 0):
        failed = core.progress(limit=1)[0]
        assert failed.status == Status.FAILED
        assert failed.update.code == "executor_error"
        assert failed.update.reason == "RuntimeError: dispatch acknowledgement lost"
        assert core.registry.capacity()["active"] == remaining
    assert all(core.registry.get(ref).status == Status.FAILED for ref in refs)


def test_resource_deferral_is_terminal_and_replay_does_not_retry_approval():
    class DeferredBoundary(CompletionBoundary):
        def start(self, work):
            self.starts += 1
            return ExecutionUpdate(
                status=Status.DEFERRED, quiescent=True, code="no_capacity"
            )

    boundary = DeferredBoundary()
    core = KVControl(
        "engine", "ns", enabled=True, executor=boundary, limits=Limits(max_active=1)
    )
    original = submit(core, action("copy_to", movement()))
    assert original.results[0].status == Status.DEFERRED
    assert submit(core, action("copy_to", movement())).results == original.results
    assert core.progress() == () and boundary.starts == 1
    submit(core, action("copy_to", movement()), message_id="intentional-retry")
    assert boundary.starts == 2


def test_message_ack_waits_for_all_siblings_and_retention_starts_after_last():
    boundary, clock = CompletionBoundary(), Clock()
    core = KVControl(
        "engine",
        "ns",
        enabled=True,
        executor=boundary,
        clock=clock,
        limits=Limits(retention_seconds=5),
    )
    result = submit(core, action(), action("copy_to", movement(), action_id="copy"))
    assert result.replay_expires_at is None
    clock.now += 20
    with fails("busy"):
        core.registry.acknowledge(core.context("router"), "m")
    ref = result.results[1].ref
    boundary.updates[ref.operation_id] = ExecutionUpdate(
        status=Status.FAILED, quiescent=True, code="storage_failure"
    )
    core.progress()
    ack = core.registry.acknowledge(core.context("router"), "m")
    assert ack.replay_expires_at == 125
    assert [s.status for s in ack.summaries] == [Status.UNCONSUMED, Status.FAILED]


def test_token_reverse_lookup_uses_namespace_and_exact_prefix_equality():
    index = LogicalIndex("ns", "epoch")
    index.associate(association(tokens((1, 2, 3)), session="one"))
    index.associate(association(tokens((1, 2, 9)), session="two"))
    index.associate(association(tokens((1, 8)), session="other"))
    assert index.token_members(tokens((1, 2))) == (("one", "c"), ("two", "c"))
    index.finish_task("one", expected_revision=index.session_revision("one"))
    assert index.token_members(tokens((1, 2))) == (("two", "c"),)
    with fails("incompatible_namespace"):
        index.token_members(
            msgspec.structs.replace(tokens((1, 2)), namespace="different")
        )


def test_direct_unchecked_structs_cannot_bypass_bounds_or_epoch_checks():
    index = LogicalIndex("ns", "epoch")
    with fails("invalid_payload"):
        index.associate(association(tokens((True,))))
    with pytest.raises(ValueError):
        Limits(max_active=True)
    with pytest.raises(ValueError):
        Limits(retention_seconds=float("nan"))
    core = KVControl("engine", "ns", enabled=True)
    context = core.context("router")
    with fails("invalid_payload"):
        submit(core, action(), context=msgspec.structs.replace(context, caller_id=""))
    assert core.registry.capacity()["records"] == 0


@pytest.mark.parametrize("queued", [True, False], ids=["queued", "started"])
def test_native_finish_task_abandons_only_removed_task_interests(queued):
    """Finishing a tree must stop its work while external/shared work survives."""

    class CancellationBoundary(CompletionBoundary):
        def __init__(self):
            super().__init__()
            self.cancelled = set()

        def cancel(self, ref):
            self.cancelled.add(ref.operation_id)
            return None

    boundary = CancellationBoundary()
    core = KVControl("engine", "ns", enabled=True, executor=boundary)
    core.index.associate(association(session="root"))
    core.index.associate(
        association(
            session="child",
            task_parent=TaskReference(
                session_id="root", expected_revision=core.index.session_revision("root")
            ),
        )
    )
    core.index.associate(association(session="external"))
    targets = {
        name: SessionTarget(namespace="ns", session_id=name, continuation_id="c")
        for name in ("root", "child", "external")
    }
    targets.update(tokens=tokens(), content=content(span(), length=4))
    context = (
        core.context("router", binding_id="r") if queued else core.context("router")
    )
    actions = [
        action("set_priority", priority(target=wire(target)), action_id=name)
        if queued
        else action("copy_to", movement(target), action_id=name)
        for name, target in targets.items()
    ]
    work = submit(core, *actions, message_id="work", context=context)
    refs = {result.ref.action_id: result.ref for result in work.results}
    finished = submit(
        core,
        action(
            "finish_task",
            {
                "session_id": "root",
                "expected_revision": core.index.session_revision("root"),
            },
        ),
        message_id="finish",
    )
    assert finished.results[0].status == Status.COMPLETED
    assert {
        task.session_id
        for task in finished.results[0].update.logical_change.removed_tasks
    } == {"root", "child"}
    affected = {refs[name].operation_id for name in ("root", "child")}
    for name in ("root", "child"):
        assert core.registry.get(refs[name]).status == (
            Status.CANCELLED if queued else Status.CANCELLING
        )
    assert core.index.token_members(tokens()) == (("external", "c"),)
    if queued:
        assert boundary.starts == 0 and not boundary.cancelled
        assert core.registry.capacity()["active"] == 3
        core.request_event(context, "success")
    else:
        assert boundary.cancelled == affected
        assert core.registry.capacity()["active"] == 5
        with fails("busy"):
            core.registry.acknowledge(context, "work")
        for operation_id in affected:
            boundary.updates[operation_id] = ExecutionUpdate(
                status=Status.CANCELLED, quiescent=True
            )
        core.progress()
    assert core.registry.capacity()["active"] == 3
    for name in ("root", "child"):
        assert core.registry.get(refs[name]).status == Status.CANCELLED
    for name in ("external", "tokens", "content"):
        assert core.registry.get(refs[name]).status == Status.RUNNING
        assert refs[name].operation_id not in boundary.cancelled
    assert boundary.starts == (3 if queued else 5)


def test_native_finish_task_matches_incarnation_and_rejects_stale_cleanup():
    """Raw index cleanup keeps snapshots; finishing a reopened name selects its incarnation."""
    boundary = CompletionBoundary()
    core = KVControl("engine", "ns", enabled=True, executor=boundary)
    old = core.index.associate(association())
    target = SessionTarget(namespace="ns", session_id="s", continuation_id="c")
    old_context = core.context("router", binding_id="old-request")
    old_work = submit(
        core,
        action("set_priority", priority(target=wire(target))),
        message_id="old",
        context=old_context,
    ).results[0]
    core.index.finish_task("s", expected_revision=core.index.session_revision("s"))
    reopened = core.index.associate(association())
    assert reopened.task != old.task
    new_work = submit(
        core,
        action("set_priority", priority(target=wire(target))),
        message_id="new",
        context=core.context("router", binding_id="new-request"),
    ).results[0]
    finish = action(
        "finish_task",
        {"session_id": "s", "expected_revision": core.index.session_revision("s")},
    )
    result = submit(core, finish, message_id="finish")
    assert core.registry.get(new_work.ref).status == Status.CANCELLED
    assert core.registry.get(old_work.ref).status == Status.QUEUED
    core.index.associate(association())
    latest = submit(
        core,
        action("set_priority", priority(target=wire(target))),
        message_id="latest",
        context=core.context("router", binding_id="latest-request"),
    ).results[0]
    replay = submit(core, finish, message_id="finish")
    assert replay.replayed and replay.results == result.results
    stale = submit(core, finish, message_id="stale-cleanup")
    assert stale.results[0].status == Status.REJECTED
    assert stale.results[0].update.code == "stale_revision"
    assert core.registry.get(latest.ref).status == Status.QUEUED
    core.request_event(old_context, "success")
    assert (
        boundary.work[old_work.ref.operation_id].target.branches[0].continuation.task
        == old.task
    )
    assert core.registry.get(old_work.ref).status == Status.RUNNING


@pytest.mark.parametrize("finish_first", [True, False])
def test_native_finish_task_cancels_sibling_snapshot_before_dispatch(finish_first):
    """A snapshot prepared in the same submission must participate in cleanup."""
    boundary = CompletionBoundary()
    core = KVControl("engine", "ns", enabled=True, executor=boundary)
    record = core.index.associate(association())
    actions = [
        action(
            "finish_task",
            {"session_id": "s", "expected_revision": record.revision},
            action_id="finish",
        ),
        action(
            "set_priority",
            priority(target=wire(SessionTarget(namespace="ns", session_id="s"))),
            action_id="priority",
        ),
    ]
    if not finish_first:
        actions.reverse()
    context = core.context("router", binding_id="r")
    result = submit(core, *actions, context=context)
    assert {r.ref.action_id: r.status for r in result.results} == {
        "finish": Status.COMPLETED,
        "priority": Status.CANCELLED,
    }
    assert core.registry.capacity()["active"] == 0
    assert core.request_event(context, "success") == ()
    assert boundary.starts == 0


@pytest.mark.parametrize("failure_at", ["start", "poll", "cancel"])
def test_executor_failure_origin_survives_drain_ack_and_replay(failure_at):
    """A quiescent cancellation receipt must not relabel a backend exception."""
    observed = completed_outcome()

    class FailingBoundary(CompletionBoundary):
        def start(self, work):
            super().start(work)
            if failure_at == "start":
                raise RuntimeError("backend submit failure")
            return ExecutionUpdate(
                status=Status.RUNNING,
                quiescent=False,
                completed_steps=("destination_ready",),
                target_outcomes=(observed,),
            )

        def poll(self, ref):
            update = self.updates.pop(ref.operation_id, None)
            if isinstance(update, Exception):
                raise update
            return update

        def cancel(self, ref):
            raise RuntimeError(
                "backend cancel failure"
                if failure_at == "cancel"
                else "secondary drain failure"
            )

    boundary = FailingBoundary()
    core = KVControl("engine", "ns", enabled=True, executor=boundary)
    payload = movement(content(span(), length=4))
    original = submit(core, action("copy_to", payload)).results[0]
    if failure_at == "poll":
        boundary.updates[original.ref.operation_id] = RuntimeError(
            "backend poll failure"
        )
        core.progress()
    elif failure_at == "cancel":
        core.cancel(original.ref)
    origin = f"RuntimeError: backend {'submit' if failure_at == 'start' else failure_at} failure"
    assert core.registry.get(original.ref).update.reason == origin
    # A further failure during cleanup must not replace the originating cause.
    core.progress()
    draining = core.registry.get(original.ref)
    assert draining.status == Status.CANCELLING and not draining.update.quiescent
    assert draining.update.reason == origin
    assert core.registry.capacity()["active"] == 1
    with fails("busy"):
        core.registry.acknowledge(core.context("router"), "m")
    boundary.updates[original.ref.operation_id] = ExecutionUpdate(
        status=Status.CANCELLED, quiescent=True
    )
    terminal = core.progress()[0]
    assert terminal.status == Status.FAILED and terminal.update.quiescent
    assert terminal.update.code == "executor_error" and terminal.update.reason == origin
    assert terminal.update.completed_steps == (
        () if failure_at == "start" else ("destination_ready",)
    )
    assert terminal.update.target_outcomes == (
        () if failure_at == "start" else (observed,)
    )
    assert core.registry.capacity()["active"] == 0
    assert submit(core, action("copy_to", payload)).results == (terminal,)
    ack = core.registry.acknowledge(core.context("router"), "m")
    assert ack.summaries[0].status == Status.FAILED
    assert (
        ack.summaries[0].code == "executor_error" and ack.summaries[0].reason == origin
    )
    replay = submit(core, action("copy_to", payload))
    assert replay.replayed and replay.acknowledged and replay.summaries == ack.summaries


def test_executor_failure_retains_progress_observed_during_drain():
    """Sparse drain updates cannot erase earlier publication or the first error."""
    first = completed_outcome()
    second = msgspec.structs.replace(
        first,
        content_id="suffix",
        coverage=TokenRange(component="full", start_token=4, end_token=8),
    )

    class FailingPoll(CompletionBoundary):
        def poll(self, ref):
            update = self.updates.pop(ref.operation_id, None)
            if isinstance(update, Exception):
                raise update
            return update

    boundary = FailingPoll()
    core = KVControl("engine", "ns", enabled=True, executor=boundary)
    result = submit(core, action("copy_to", movement(tokens(range(8))))).results[0]
    boundary.updates[result.ref.operation_id] = ExecutionUpdate(
        status=Status.RUNNING,
        quiescent=False,
        completed_steps=("destination_ready",),
        target_outcomes=(first,),
    )
    core.progress()
    boundary.updates[result.ref.operation_id] = RuntimeError("backend poll failure")
    core.progress()
    boundary.updates[result.ref.operation_id] = ExecutionUpdate(
        status=Status.RUNNING, quiescent=False, target_outcomes=(second,)
    )
    progress = core.progress()[0]
    assert progress.status == Status.CANCELLING
    assert progress.update.reason == "RuntimeError: backend poll failure"
    assert progress.update.completed_steps == ("destination_ready",)
    assert progress.update.target_outcomes == (first, second)
    boundary.updates[result.ref.operation_id] = ExecutionUpdate(
        status=Status.CANCELLED, quiescent=True
    )
    terminal = core.progress()[0]
    assert terminal.status == Status.FAILED
    assert terminal.update.completed_steps == ("destination_ready",)
    assert terminal.update.target_outcomes == (first, second)
    assert terminal.update.reason == "RuntimeError: backend poll failure"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
