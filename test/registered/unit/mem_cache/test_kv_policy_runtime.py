# SPDX-License-Identifier: Apache-2.0
"""Configured retention acts on actual CPU cache copies through native control."""

from unittest.mock import patch

import msgspec
import pytest
import test_kv_control_cache as cache_fixtures
import test_kv_control_transfers as fixtures

from sglang.srt.managers.kv_control.contracts import Associate, SessionTarget, Status
from sglang.srt.managers.kv_control.policy_runtime import (
    PolicyDefinition,
    RetentionOptions,
)
from sglang.srt.managers.kv_control.service import ServiceConfig
from sglang.srt.managers.kv_hints import KvHintAction, KvHintsEnvelope
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")
owner = fixtures.owner
CONFIG = ServiceConfig(
    namespace="ns",
    engine_id="engine",
    policies=(
        PolicyDefinition(
            name="retain", options=RetentionOptions(priority=50, ttl_seconds=10)
        ),
    ),
)
pytestmark = pytest.mark.parametrize("owner", [CONFIG], indirect=True)


def submit(service, kind="signal", *, message="signal", **payload):
    return service.control.submit(
        KvHintsEnvelope(
            protocol_version="0.1",
            message_id=message,
            actions=[
                KvHintAction(
                    action_id="a",
                    action_type="sglang." + kind,
                    action_version="1.0",
                    payload=msgspec.json.decode(msgspec.json.encode(payload)),
                )
            ],
        ),
        service.control.context("router"),
    )


def observed(service):
    return [
        item.effective_priority
        for item in service.control.executor.residency(
            service.control.index.resolve(cache_fixtures.target())
        ).items
    ]


def test_signal_creates_queryable_policy_child_and_replay_does_not_renew(owner):
    case, cache, service, dma = owner
    fixtures.seed(cache)
    first = submit(
        service, target=cache_fixtures.target(), signal="importance", importance=80
    )
    parent = first.results[0]
    assert parent.status == Status.COMPLETED
    (child,) = parent.update.derived_operations
    assert service.control.registry.get(child).status == Status.COMPLETED
    assert observed(service) == [80]
    before = service.control.registry.capacity()
    case.clock.now += 5
    assert submit(
        service, target=cache_fixtures.target(), signal="importance", importance=80
    ).replayed
    assert service.control.registry.capacity() == before
    case.clock.now += 6
    assert observed(service) == [0]
    renewed = submit(
        service, target=cache_fixtures.target(), signal="resume", message="renew"
    )
    assert renewed.results[0].status == Status.COMPLETED
    assert observed(service) == [50]
    assert not dma.pending and not service.has_pending


def test_unconsumed_signal_does_not_resolve_history_or_construct_observations(owner):
    _, cache, service, dma = owner
    target = SessionTarget(namespace="ns", session_id="untracked")
    with (
        patch.object(
            service.control.index, "resolve", side_effect=AssertionError("history read")
        ),
        patch(
            "sglang.srt.managers.kv_control.policy_runtime.PolicyObservation",
            side_effect=AssertionError("observation built"),
        ),
    ):
        result = submit(service, target=target, signal="tool_wait")
    assert result.results[0].status == Status.UNCONSUMED
    assert not result.results[0].update.derived_operations
    assert not dma.pending


def test_task_cleanup_preserves_peer_declarations_and_explicit_claims(owner):
    case, cache, service, dma = owner
    node = fixtures.seed(cache)
    for name, value in (("A", 80), ("B", 30)):
        service.control.index.associate(
            Associate(
                session_id=name,
                continuation_id="c",
                expected_revision=0,
                target=cache_fixtures.target(),
            )
        )
        submit(
            service,
            target=SessionTarget(namespace="ns", session_id=name),
            signal="importance",
            importance=value,
            message=name,
        )
    explicit = case.priority(service.control, value=-5)
    assert explicit.status == Status.COMPLETED and observed(service) == [-5]
    before = cache.token_to_kv_pool_allocator.available_size()
    finished = submit(
        service,
        "finish_task",
        message="finish",
        session_id="A",
        expected_revision=service.control.index.session_revision("A"),
    )
    assert finished.results[0].status == Status.COMPLETED
    assert observed(service) == [-5]
    case.priority(service.control, value=None, revision=2)
    assert observed(service) == [30]
    assert cache.token_to_kv_pool_allocator.available_size() == before
    assert node.component_data[ComponentType.FULL].lock_ref == 0
    service.control.index.associate(
        Associate(
            session_id="A",
            continuation_id="c",
            expected_revision=0,
            target=cache_fixtures.target(),
        )
    )
    assert submit(
        service,
        target=SessionTarget(namespace="ns", session_id="A"),
        signal="importance",
        importance=80,
        message="A",
    ).replayed
    assert observed(service) == [30]


def test_explicit_priority_beats_policy_until_its_expiry(owner):
    case, cache, service, dma = owner
    fixtures.seed(cache)
    case.priority(service.control, value=-10, ttl=3)
    submit(service, target=cache_fixtures.target(), signal="importance", importance=90)
    assert observed(service) == [-10]
    case.clock.now += 4
    assert observed(service) == [90]


def test_flush_rebuilds_policy_runtime_for_the_new_epoch(owner):
    _, cache, service, dma = owner
    fixtures.seed(cache)
    parent = submit(service, target=cache_fixtures.target(), signal="resume").results[0]
    old_runtime = service.policy_runtime
    cache.reset()
    service.reset_after_cache_flush()
    assert service.policy_runtime is not old_runtime
    assert service.control.policy_runtime is service.policy_runtime
    assert service.capabilities()["policies"] == ("retain",)
    fixtures.seed(cache)
    result = submit(service, target=cache_fixtures.target(), signal="resume")
    assert result.results[0].ref.context.engine_epoch != parent.ref.context.engine_epoch
    assert observed(service) == [50]


def test_pending_explicit_move_defers_conflicting_policy_retention(owner):
    _, cache, service, dma = owner
    node = fixtures.seed(cache)
    movement = fixtures.transfer(service, kind="move_to")
    parent = submit(service, target=cache_fixtures.target(), signal="resume").results[0]
    (child,) = parent.update.derived_operations
    result = service.control.registry.get(child)
    assert result.status == Status.DEFERRED and result.update.code == "policy_conflict"
    assert cache.tree_core.priority_claims.effective(node, ComponentType.FULL) is None
    service.progress()
    dma.finish()
    service.progress()
    assert service.control.registry.get(movement.ref).status == Status.COMPLETED


def test_unsupported_request_profile_does_not_execute_policy_and_can_finish_tasks(
    owner,
):
    _, cache, service, dma = owner
    fixtures.seed(cache)
    service.control.index.associate(
        Associate(
            session_id="task",
            continuation_id="c",
            expected_revision=0,
            target=cache_fixtures.target(),
        )
    )
    target = SessionTarget(namespace="ns", session_id="task")
    submit(service, target=target, signal="resume")
    assert observed(service) == [50]
    executor, service.control.executor = service.control.executor, None
    try:
        unsupported = submit(
            service, target=target, signal="resume", message="unsupported"
        )
        assert unsupported.results[0].status == Status.UNSUPPORTED
        assert not unsupported.results[0].update.derived_operations
        finished = submit(
            service,
            "finish_task",
            message="finish-disabled",
            session_id="task",
            expected_revision=service.control.index.session_revision("task"),
        )
        assert finished.results[0].status == Status.COMPLETED
    finally:
        service.control.executor = executor
    assert observed(service) == [0]
    assert not dma.pending
