# SPDX-License-Identifier: Apache-2.0
"""Control protocol, bounded replay and task/request lifecycle."""

import copy
import unittest
from contextlib import contextmanager

import msgspec
import pytest

from sglang.srt.managers.kv_control import KVControl, LogicalIndex
from sglang.srt.managers.kv_control.contracts import (
    Associate,
    ContentSpan,
    ContentTarget,
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
)
from sglang.srt.managers.kv_hints import KvHintAction, KvHintsEnvelope
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


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


class TestKVControl(CustomTestCase):
    def test_mixed_owners_errors_and_replay_are_independent(self):
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

    def test_disabled_default_and_absent_executor_are_honest(self):
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

    def test_snapshot_detaches_mutable_input_and_preserves_conflict_identity(self):
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
        assert submit(
            core, action("copy_to", original_payload), context=context
        ).replayed
        core.request_event(context, "success")
        assert boundary.work[ref.operation_id].target.branches[0].content.token_ids == (
            1,
            2,
            3,
            4,
        )
        assert core.registry.get(ref).status == Status.RUNNING
        assert boundary.starts == 1

    def test_context_separates_fanout_and_callers_and_rejects_stale_generations(self):
        core = KVControl("engine", "ns", enabled=True)
        contexts = [
            core.context("router", binding_id=f"request/{sample}") for sample in (0, 1)
        ]
        contexts.append(core.context("other-router", binding_id="request/0"))
        refs = [submit(core, action(), context=ctx).results[0].ref for ctx in contexts]
        assert len({ref.operation_id for ref in refs}) == 3
        assert all((submit(core, action(), context=ctx).replayed for ctx in contexts))
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

    def test_preflight_limits_do_not_partially_accept_or_mutate(self):
        for limits, actions in [
            (Limits(max_actions=1), [action(action_id="1"), action(action_id="2")]),
            (Limits(max_active=1), [action(action_id="1"), action(action_id="2")]),
        ]:
            with self.subTest(limits=limits, actions=actions):
                core = KVControl("engine", "ns", enabled=True, limits=limits)
                with fails("capacity"):
                    submit(core, *actions)
                assert core.registry.capacity() == {
                    "active": 0,
                    "records": 0,
                    "messages": 0,
                }
                assert core.index.generation == 0

    def test_ack_retains_failure_and_replay_deadline_without_renewal(self):
        clock = Clock()
        core = KVControl(
            "engine",
            "ns",
            enabled=True,
            limits=Limits(retention_seconds=10),
            clock=clock,
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
        assert (
            replay.replayed
            and replay.acknowledged
            and (replay.summaries == ack.summaries)
        )
        with fails("conflict"):
            submit(core, action())
        clock.now = 109.999
        assert core.registry.acknowledge(context, "m") == ack
        clock.now = 110
        new = submit(core, action("copy_to", movement()), context=context)
        assert not new.replayed
        assert new.results[0].ref.operation_id != original.results[0].ref.operation_id

    def test_cancellation_requires_incarnation_and_drained_worker_access(self):
        boundary = CompletionBoundary()
        core = KVControl("engine", "ns", enabled=True, executor=boundary)
        result = submit(core, action("copy_to", movement())).results[0]
        with fails("stale_operation"):
            core.cancel(
                msgspec.structs.replace(result.ref, operation_id="older-instance")
            )
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

    def test_request_cached_binds_only_success_and_checkpoint_never_reexecutes(self):
        boundary = CompletionBoundary()
        core = KVControl("engine", "ns", enabled=True, executor=boundary)
        context = core.context("router", binding_id="r")
        result = submit(
            core, action("set_priority", priority()), context=context
        ).results[0]
        assert result.status == Status.QUEUED
        assert core.request_event(context, "checkpoint", committed=tokens()) == ()
        assert core.request_event(context, "preempt") == ()
        assert boundary.starts == 0
        core.request_event(context, "success", committed=tokens((1, 2, 3, 4, 5, 6)))
        assert (
            boundary.work[result.ref.operation_id]
            .target.branches[0]
            .requested.end_token
            == 6
        )
        assert core.request_event(context, "success", committed=tokens((9,))) == ()
        assert submit(
            core, action("set_priority", priority()), context=context
        ).replayed
        assert boundary.starts == 1

    def test_branch_uses_explicit_parent_boundary_and_different_suffix(self):
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
            association(
                tokens((1, 2, 3, 4, 5)), session="parent", revision=parent.revision
            )
        )
        grown = index.associate(
            association(
                tokens((1, 2, 90, 91)), session="child", revision=child.revision
            )
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

    def test_range_resolution_reports_missing_components_and_never_extends_coverage(
        self,
    ):
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
        assert [(s.start_token, s.end_token) for s in branch.unaligned] == [
            (0, 4),
            (8, 12),
        ]
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

    def test_complete_copy_requires_every_component_and_snapshot_branch(self):
        boundary = CompletionBoundary()
        core = KVControl("engine", "ns", enabled=True, executor=boundary)
        target = content(
            span(),
            span("state", component="mamba"),
            length=4,
            components=("full", "mamba"),
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

    def test_native_finish_task_abandons_only_removed_task_interests(self):
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
                    session_id="root",
                    expected_revision=core.index.session_revision("root"),
                ),
            )
        )
        core.index.associate(association(session="external"))
        targets = {
            name: SessionTarget(namespace="ns", session_id=name, continuation_id="c")
            for name in ("root", "child", "external")
        }
        targets.update(tokens=tokens(), content=content(span(), length=4))
        context = core.context("router")
        actions = [
            action("copy_to", movement(target), action_id=name)
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
            assert core.registry.get(refs[name]).status == Status.CANCELLING
        assert core.index.token_members(tokens()) == (("external", "c"),)
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
        assert boundary.starts == 5

    def test_executor_failure_origin_survives_drain_ack_and_replay(self):
        """A quiescent cancellation receipt must not relabel a backend exception."""
        observed = completed_outcome()

        class FailingBoundary(CompletionBoundary):
            def start(self, work):
                super().start(work)
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
                raise RuntimeError("secondary drain failure")

        boundary = FailingBoundary()
        core = KVControl("engine", "ns", enabled=True, executor=boundary)
        payload = movement(content(span(), length=4))
        original = submit(core, action("copy_to", payload)).results[0]
        boundary.updates[original.ref.operation_id] = RuntimeError(
            "backend poll failure"
        )
        core.progress()
        origin = "RuntimeError: backend poll failure"
        assert core.registry.get(original.ref).update.reason == origin
        core.progress()
        draining = core.registry.get(original.ref)
        assert draining.status == Status.CANCELLING and (not draining.update.quiescent)
        assert draining.update.reason == origin
        assert core.registry.capacity()["active"] == 1
        with fails("busy"):
            core.registry.acknowledge(core.context("router"), "m")
        boundary.updates[original.ref.operation_id] = ExecutionUpdate(
            status=Status.CANCELLED, quiescent=True
        )
        terminal = core.progress()[0]
        assert terminal.status == Status.FAILED and terminal.update.quiescent
        assert (
            terminal.update.code == "executor_error"
            and terminal.update.reason == origin
        )
        assert terminal.update.completed_steps == ("destination_ready",)
        assert terminal.update.target_outcomes == (observed,)
        assert core.registry.capacity()["active"] == 0
        assert submit(core, action("copy_to", payload)).results == (terminal,)
        ack = core.registry.acknowledge(core.context("router"), "m")
        assert ack.summaries[0].status == Status.FAILED
        assert (
            ack.summaries[0].code == "executor_error"
            and ack.summaries[0].reason == origin
        )
        replay = submit(core, action("copy_to", payload))
        assert (
            replay.replayed
            and replay.acknowledged
            and (replay.summaries == ack.summaries)
        )


if __name__ == "__main__":
    unittest.main()
