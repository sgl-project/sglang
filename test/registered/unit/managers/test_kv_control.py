# SPDX-License-Identifier: Apache-2.0
"""Control protocol, bounded replay and task/request lifecycle."""

import copy
import json
import unittest
from contextlib import contextmanager
from pathlib import Path

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
        invalid = action(
            "copy_to", movement() | {"dst": {"tier": True}}, action_id="bad"
        )
        unknown = action("future_action", {}, action_id="unsupported")
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

    def test_legacy_nonstorage_backend_namespace_is_rejected_before_execution(self):
        boundary = CompletionBoundary()
        core = KVControl("engine", "ns", enabled=True, executor=boundary)
        for kind, payload, location in (
            ("copy_to", movement(src="gpu", dst="host"), "dst"),
            ("move_to", movement(src="gpu", dst="host"), "src"),
        ):
            payload[location]["backend_namespace"] = "storage-domain"
            result = submit(core, action(kind, payload), message_id=kind).results[0]
            self.assertEqual(result.status, Status.REJECTED)
            self.assertEqual(result.update.code, "invalid_payload")
        self.assertEqual(boundary.starts, 0)

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
            core,
            action("copy_to", movement() | {"target": "request_cached"}),
            context=context,
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
            core,
            action("copy_to", movement() | {"target": "request_cached"}),
            context=context,
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


class TestPublicKVProtocol(CustomTestCase):
    fixtures = (
        Path(__file__).resolve().parents[4] / "sgl-router/sgl-kv-orchestrator/fixtures"
    )

    def test_shared_schema_and_invalid_vectors(self):
        from sglang.srt.managers.kv_control.contracts import (
            ClaimState,
            Demote,
            Pin,
            Prefetch,
            PublicResult,
            parse_action,
        )

        for name, schema in {
            "prefetch": Prefetch,
            "demote": Demote,
            "pin": Pin,
            "result": PublicResult,
            "claim": ClaimState,
        }.items():
            self.assertEqual(
                json.loads((self.fixtures / f"{name}.schema.json").read_text()),
                wire(msgspec.json.schema(schema)),
            )
        for case in json.loads((self.fixtures / "actions.json").read_text()):
            with self.subTest(case=case["name"]):
                a = msgspec.convert(case["action"], type=KvHintAction)
                if case["valid"]:
                    parse_action(a)
                else:
                    with self.assertRaises(ControlError):
                        parse_action(a)

    def test_result_and_claim_vectors_remain_separate(self):
        from sglang.srt.managers.kv_control.contracts import (
            ClaimState,
            PublicResult,
            validated,
        )

        fixtures = json.loads((self.fixtures / "results.json").read_text())
        for key, schema in (("results", PublicResult), ("claims", ClaimState)):
            for value in fixtures[key]:
                self.assertEqual(wire(validated(value, schema)), value)
        self.assertTrue(fixtures["claims"][0]["tombstone"])
        self.assertTrue(all(r["claim"] is None for r in fixtures["results"]))

    def test_salt_vectors_and_raw_salt_normalization(self):
        from sglang.srt.managers.kv_control.contracts import parse_action
        from sglang.srt.managers.kv_control.key_namespace import encode_cache_salt
        from sglang.srt.managers.kv_control.public import normalize_public

        for case in json.loads((self.fixtures / "salt.json").read_text()):
            self.assertEqual(
                encode_cache_salt(case["namespace"], case["salt"]), case["encoded"]
            )
        a = self.public_action()
        a.payload["target"]["compatibility"]["cache_salt"] = "raw"
        self.assertEqual(
            normalize_public(parse_action(a), self.caps()).target.cache_salt, "raw"
        )
        a.payload["target"]["compatibility"]["cache_salt"] = ""
        self.assertIsNone(
            normalize_public(parse_action(a), self.caps()).target.cache_salt
        )

    @classmethod
    def public_action(cls):
        case = json.loads((cls.fixtures / "actions.json").read_text())[0]
        return msgspec.convert(case["action"], type=KvHintAction)

    @staticmethod
    def caps():
        return {
            "namespace": "ns",
            "layout_id": "python:full:p4:w0",
            "components": ("full",),
            "actions": ("sglang.copy_to", "sglang.move_to"),
            "cache_generation": 7,
            "directions": {
                "copy_to": (("storage", "host"), ("host", "gpu")),
                "move_to": (("gpu", "host"),),
                "storage_backend": "file",
                "page_alignment": 4,
            },
        }

    def owner(self):
        executor = CompletionBoundary()
        executor.capabilities = self.caps
        clock = Clock()
        return (
            KVControl("engine", "ns", enabled=True, executor=executor, clock=clock),
            executor,
            clock,
        )

    def test_unsupported_capabilities_and_compatibility_never_dispatch(self):
        from sglang.srt.managers.kv_control.public import public_capabilities

        self.assertNotIn("sglang.pin", public_capabilities(self.caps())["actions"])
        self.assertFalse(public_capabilities({})["actions"])
        for field, value in (
            ("namespace", "other"),
            ("layout_id", "other"),
            ("page_size", 2),
            ("components", ["mamba"]),
            ("hash_scheme", "other"),
        ):
            control, executor, _ = self.owner()
            a = self.public_action()
            a.payload["target"]["compatibility"][field] = value
            result = control.submit(envelope(a), control.context("A")).results[0]
            self.assertIn(result.status, (Status.UNSUPPORTED, Status.REJECTED))
            self.assertEqual(executor.starts, 0)
        for mutation in ("hash_only", "mooncake", "namespace", "pin"):
            control, executor, _ = self.owner()
            a = self.public_action()
            if mutation == "hash_only":
                a.payload["target"].pop("token_ids")
            elif mutation == "mooncake":
                a.payload["source"]["backend"] = "mooncake"
            elif mutation == "namespace":
                a.payload["source"]["backend_namespace"] = "shared"
            else:
                a = action(
                    "pin",
                    {
                        "mode": "acquire",
                        "target": a.payload["target"],
                        "location": {"tier": "host"},
                        "claim_id": "c",
                        "revision": 1,
                        "ttl_seconds": 10,
                    },
                )
            result = control.submit(envelope(a), control.context("A")).results[0]
            self.assertEqual(
                result.status,
                Status.REJECTED if mutation == "namespace" else Status.UNSUPPORTED,
            )
            self.assertEqual(executor.starts, 0)
            self.assertIsNone(result.public_result.claim)

    def test_replay_versions_and_timeout_drain_keep_actual_progress(self):
        control, executor, clock = self.owner()
        a = self.public_action()
        ctx = control.context("A")
        submitted = control.submit(envelope(a), ctx)
        ref = submitted.results[0].ref
        self.assertFalse(submitted.results[0].public_result.ready_scope)
        self.assertTrue(control.submit(envelope(a), ctx).replayed)
        self.assertEqual(executor.starts, 1)
        a.payload["timeout_ms"] = 10
        with self.assertRaises(ControlError):
            control.submit(envelope(a), ctx)
        clock.now += 6
        control.progress()
        cancelling = control.registry.get(ref)
        self.assertEqual(cancelling.status, Status.CANCELLING)
        self.assertFalse(cancelling.update.quiescent)
        self.assertFalse(cancelling.public_result.ready_scope)
        executor.updates[ref.operation_id] = ExecutionUpdate(
            status=Status.CANCELLED, quiescent=True
        )
        control.progress()
        done = control.registry.get(ref)
        self.assertTrue(done.update.quiescent)
        self.assertGreater(
            done.public_result.state_version, cancelling.public_result.state_version
        )
        self.assertEqual(done.public_result.outcome_sequence, done.sequence)

    def test_host_to_storage_demote_remains_unsupported(self):
        control, executor, _ = self.owner()
        a = self.public_action()
        a = action(
            "demote",
            dict(a.payload, source={"tier": "host"}, destination={"tier": "storage"}),
        )
        result = control.submit(envelope(a), control.context("A")).results[0]
        self.assertEqual(result.status, Status.UNSUPPORTED)
        self.assertFalse(result.public_result.reclaimed_scope)
        self.assertEqual(executor.starts, 0)

    def test_attached_timeout_and_old_result_wire_contract(self):
        control, executor, clock = self.owner()
        a = self.public_action()
        a.payload["target"] = "request_cached"
        context = control.context("A", binding_id="request-incarnation")
        ref = control.submit(envelope(a), context).results[0].ref
        self.assertEqual(control.registry.get(ref).status, Status.QUEUED)
        clock.now += 6
        control.progress()
        self.assertEqual(control.registry.get(ref).status, Status.CANCELLED)
        control.request_event(context, "success", committed=tokens())
        self.assertEqual(executor.starts, 0)
        legacy = control.submit(
            envelope(action("copy_to", movement()), message_id="legacy"),
            control.context("A"),
        )
        self.assertNotIn("public_result", wire(legacy.results[0]))
        self.assertEqual(wire(legacy.results[0])["update"]["status"], "running")

    def test_missing_and_capacity_results_keep_scope_and_reason(self):
        for status, field in (
            (Status.MISSING, "missing_scope"),
            (Status.DEFERRED, "blocked_scope"),
        ):
            control, executor, _ = self.owner()
            executor.start = lambda work: ExecutionUpdate(
                status=status, quiescent=True, code="test_boundary"
            )
            result = control.submit(
                envelope(self.public_action()), control.context("A")
            ).results[0]
            scope = getattr(result.public_result, field)
            self.assertEqual(len(scope), 1)
            self.assertEqual(scope[0].coverage.end_token, 4)
            self.assertEqual(scope[0].code, "test_boundary")
            self.assertFalse(result.public_result.ready_scope)
            self.assertFalse(result.public_result.accepted_scope)

    def test_demote_projection_preservation_is_not_reclamation(self):
        control, executor, _ = self.owner()
        a = self.public_action()
        a = action(
            "demote",
            dict(a.payload, source={"tier": "gpu"}, destination={"tier": "host"}),
        )
        ref = control.submit(envelope(a), control.context("A")).results[0].ref
        coverage = TokenRange(start_token=0, end_token=4, component="full")
        outcome = TargetOutcome(
            coverage=coverage,
            status=Status.PARTIAL,
            location=Location(tier="host"),
            completed_steps=("destination_ready",),
            code="active_request_reference",
        )
        executor.updates[ref.operation_id] = ExecutionUpdate(
            status=Status.PARTIAL,
            quiescent=True,
            completed_steps=("destination_ready",),
            target_outcomes=(outcome,),
        )
        control.progress()
        result = control.registry.get(ref)
        public = result.public_result
        self.assertEqual(public.ready_scope, (outcome,))
        self.assertEqual(public.preserved_scope, (outcome,))
        self.assertEqual(public.blocked_scope, (outcome,))
        self.assertFalse(public.reclaimed_scope)
        self.assertEqual(public.cache_generation, 7)
        self.assertEqual(public.owner_epoch, ref.context.engine_epoch)
        # Existing consumers still receive their original update and steps.
        self.assertEqual(result.update.target_outcomes, (outcome,))


if __name__ == "__main__":
    unittest.main()
