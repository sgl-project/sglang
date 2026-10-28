# SPDX-License-Identifier: Apache-2.0
"""Configured policies act on actual copies and hand off to native prefill."""

import unittest
from array import array
from types import SimpleNamespace

import msgspec
import test_kv_control_cache as cache_fixtures
import test_kv_control_transfers as transfers

from sglang.srt.managers.kv_control.contracts import Associate, SessionTarget, Status
from sglang.srt.managers.kv_control.policy_runtime import (
    PolicyDefinition,
    QuotaDefinition,
    QuotaOptions,
    RetentionOptions,
    ToolWaitDefinition,
)
from sglang.srt.managers.kv_control.service import ServiceConfig
from sglang.srt.managers.kv_control.tool_wait_policy import ToolWaitOptions
from sglang.srt.managers.kv_hints import KvHintAction, KvHintsEnvelope
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.common import match_kv_cache
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def policy_submit(service, kind="signal", *, message="signal", **payload):
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


RETENTION = ServiceConfig(
    namespace="ns",
    engine_id="engine",
    policies=(
        PolicyDefinition(
            name="retain", options=RetentionOptions(priority=50, ttl_seconds=10)
        ),
    ),
)


class TestKVRetention(cache_fixtures.KVCacheTestCase):
    def test_signal_creates_queryable_policy_child_and_replay_does_not_renew(self):
        with transfers.hicache(self, RETENTION) as owner:
            case, cache, service, dma = owner
            transfers.seed(cache)
            first = policy_submit(
                service,
                target=cache_fixtures.target(),
                signal="importance",
                importance=80,
            )
            parent = first.results[0]
            assert parent.status == Status.COMPLETED
            (child,) = parent.update.derived_operations
            assert service.control.registry.get(child).status == Status.COMPLETED
            assert observed(service) == [80]
            before = service.control.registry.capacity()
            case.clock.now += 5
            assert policy_submit(
                service,
                target=cache_fixtures.target(),
                signal="importance",
                importance=80,
            ).replayed
            assert service.control.registry.capacity() == before
            case.clock.now += 6
            assert observed(service) == [0]
            renewed = policy_submit(
                service,
                target=cache_fixtures.target(),
                signal="resume",
                message="renew",
            )
            assert renewed.results[0].status == Status.COMPLETED
            assert observed(service) == [50]
            assert not dma.pending and (not service.has_pending)

    def test_task_cleanup_preserves_peer_declarations_and_explicit_claims(self):
        with transfers.hicache(self, RETENTION) as owner:
            case, cache, service, dma = owner
            node = transfers.seed(cache)
            for name, value in (("A", 80), ("B", 30)):
                service.control.index.associate(
                    Associate(
                        session_id=name,
                        continuation_id="c",
                        expected_revision=0,
                        target=cache_fixtures.target(),
                    )
                )
                policy_submit(
                    service,
                    target=SessionTarget(namespace="ns", session_id=name),
                    signal="importance",
                    importance=value,
                    message=name,
                )
            explicit = case.priority(service.control, value=-5)
            assert explicit.status == Status.COMPLETED and observed(service) == [-5]
            before = cache.token_to_kv_pool_allocator.available_size()
            finished = policy_submit(
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
            assert policy_submit(
                service,
                target=SessionTarget(namespace="ns", session_id="A"),
                signal="importance",
                importance=80,
                message="A",
            ).replayed
            assert observed(service) == [30]


def task(service, name):
    service.control.index.associate(
        Associate(
            session_id=name,
            continuation_id="c",
            expected_revision=0,
            target=cache_fixtures.target(),
        )
    )


def policy_request(service, name, tokens=(1, 2, 3, 4, 9), *, output=1):
    req = Req(
        rid=name,
        origin_input_text=None,
        origin_input_ids=array("q", tokens),
        sampling_params=SamplingParams(max_new_tokens=output, ignore_eos=True),
        vocab_size=128,
    )
    service.prepare_request(req)
    envelope = KvHintsEnvelope(
        protocol_version="0.1",
        message_id="bind-" + name,
        actions=[
            KvHintAction(
                action_id="resume",
                action_type="sglang.signal",
                action_version="1.0",
                payload={
                    "target": {
                        "kind": "session",
                        "namespace": "ns",
                        "session_id": name,
                    },
                    "signal": "resume",
                },
            )
        ],
    )
    service.attach(req, envelope)
    return req


def observe(cache, service, requests, running=()):
    for req in requests:
        match_kv_cache(req, cache)
    batch = SimpleNamespace(reqs=list(running))
    scheduler = SimpleNamespace(
        waiting_queue=list(requests),
        chunked_req=None,
        new_token_ratio_tracker=SimpleNamespace(current=1),
    )
    service.update_policy_guidance(scheduler, batch)
    return (service.policy_runtime._view, scheduler, batch)


QUOTA = ServiceConfig(
    namespace="ns",
    engine_id="engine",
    policies=(
        QuotaDefinition(name="quota", options=QuotaOptions(borrow_idle_capacity=False)),
    ),
)


class TestKVQuotaGuidance(cache_fixtures.KVCacheTestCase):
    def test_policy_deferral_aging_and_native_priority_order(self):
        with transfers.hicache(self, QUOTA) as owner:
            case, cache, service, dma = owner
            transfers.seed(cache)
            for name in ("A", "B"):
                task(service, name)
            a, b = (
                policy_request(service, "A", output=30),
                policy_request(service, "B", (7, 8, 9), output=1),
            )
            observe(cache, service, (a, b))
            assert not service.policy_allows_request(
                a
            ) and service.policy_allows_request(b)
            assert service.policy_request_key(b, 1, 0) < service.policy_request_key(
                a, 0, 0
            )
            a.priority, b.priority = (10, 0)
            assert service.policy_request_key(a, 0, -1) < service.policy_request_key(
                b, 1, -1
            )
            case.clock.now += 2
            assert service.policy_allows_request(a)
            assert service.policy_request_key(a, 0, 0) < service.policy_request_key(
                b, 1, 0
            )

    def test_advisory_allowance_cannot_spend_stale_space_before_real_prefill(self):
        with transfers.hicache(self, QUOTA) as owner:
            _, cache, service, dma = owner
            transfers.seed(cache)
            task(service, "A")
            req = policy_request(service, "A")
            _, _, batch = observe(cache, service, (req,))
            assert service.policy_allows_request(req)
            other = cache.token_to_kv_pool_allocator.alloc(60)
            try:
                req.init_next_round_input(cache)
                adder = PrefillAdder(
                    1, cache, cache.token_to_kv_pool_allocator, batch, 1, 64, None
                )
                result = adder.add_one_req(
                    req, has_chunked_req=False, truncation_align_size=None
                )
                assert result in (AddReqResult.NO_TOKEN, AddReqResult.OTHER)
                assert not adder.can_run_list
                assert cache.token_to_kv_pool_allocator.available_size() == 0
            finally:
                cache.token_to_kv_pool_allocator.free(other)


FORCED = ToolWaitOptions(
    min_pressure=0, min_pause_seconds=0.001, initial_estimated_transfer_seconds=0.001
)


def config(*, tier="host", forced=True):
    options = (
        msgspec.structs.replace(FORCED, tier=tier)
        if forced
        else ToolWaitOptions(tier=tier)
    )
    return ServiceConfig(
        namespace="ns",
        engine_id="engine",
        policies=(
            PolicyDefinition(
                name="retain", options=RetentionOptions(priority=40, ttl_seconds=10)
            ),
            ToolWaitDefinition(name="tools", options=options),
        ),
    )


def initialize(owner):
    _, cache, service, dma = owner
    node = transfers.seed(cache)
    task(service, "task")
    return (node, SessionTarget(namespace="ns", session_id="task"))


def signal(service, target, kind, *, message, pause=None):
    parent = policy_submit(
        service,
        target=target,
        signal=kind,
        message=message,
        **{"resume_after_seconds": pause} if pause is not None else {},
    ).results[0]
    assert parent.status == Status.COMPLETED, parent
    return (
        parent,
        tuple(
            (
                service.control.registry.get(ref)
                for ref in parent.update.derived_operations
            )
        ),
    )


def save(owner, target):
    _, cache, service, dma = owner
    _, children = signal(service, target, "tool_wait", message="pause", pause=10)
    (child,) = children
    service.progress()
    dma.finish()
    done = transfers.wait(service, child)
    assert done.status == Status.COMPLETED, done
    return done


def begin_resume(owner):
    _, cache, service, dma = owner
    req = policy_request(service, "task")
    observe(cache, service, (req,))
    assert not service.prepare_admission(req)
    transfers.wait_for(service, lambda: bool(dma.pending))
    return req


class TestKVToolCoordination(cache_fixtures.KVCacheTestCase):
    def test_save_restore_priority_dependency_and_actual_prefill_handoff(self):
        for tier in ("host", "storage"):
            with self.subTest(tier=tier):
                with transfers.hicache(
                    self,
                    config(tier=tier),
                    storage=tier == "storage",
                    namespace_keys=tier == "storage",
                ) as owner:
                    _, cache, service, dma = owner
                    node, target = initialize(owner)
                    save(owner, target)
                    assert node.component_data[FULL].value is None
                    if cache.enable_storage:
                        assert cache.evict_host(4) == 4
                    req = begin_resume(owner)
                    parent = service.control.registry.query(
                        req.kv_control_context, "bind-task"
                    ).results[0]
                    retention = next(
                        (
                            service.control.registry.get(ref)
                            for ref in parent.update.derived_operations
                            if service.control.registry.get(ref).action_type
                            == "sglang.set_priority"
                        )
                    )
                    assert retention.status == Status.QUEUED
                    dma.finish()
                    transfers.wait_for(
                        service,
                        lambda: any(
                            (
                                w.ready
                                for w in service.policy_runtime.execution.works.values()
                            )
                        ),
                    )
                    observe(cache, service, (req,))
                    assert service.prepare_admission(req)
                    current = service.control.executor.residency(
                        service.control.index.resolve(cache_fixtures.target())
                    )
                    assert [item.effective_priority for item in current.items] == [40]
                    req.init_next_round_input(cache)
                    adder = PrefillAdder(
                        1,
                        cache,
                        cache.token_to_kv_pool_allocator,
                        SimpleNamespace(reqs=[]),
                        1,
                        64,
                        None,
                    )
                    adder.add_one_req(
                        req, has_chunked_req=False, truncation_align_size=None
                    )
                    assert req in adder.can_run_list and req.lock is not None
                    try:
                        service.policy_request_admitted(req)
                        assert not service.control.executor.preparations
                        assert (
                            service.policy_runtime.execution.stats[
                                "preparations_consumed"
                            ]
                            == 1
                        )
                        assert (
                            service.policy_runtime.execution.stats[
                                "consumed_prefix_tokens"
                            ]
                            == 4
                        )
                        cache.evict(EvictParams(num_tokens=4))
                        assert cache.token_to_kv_pool_allocator.available_size() == 60
                    finally:
                        cache.unlock(req.lock)
                        req.lock = None
                    service.progress()
                    assert not service.has_pending

    def test_fit_short_and_unknown_pause_do_not_move(self):
        for forced, pause in ((False, 10), (True, 0.001), (True, None)):
            with self.subTest(forced=forced, pause=pause):
                with transfers.hicache(self, config(forced=forced)) as owner:
                    _, cache, service, dma = owner
                    node, target = initialize(owner)
                    parent, children = signal(
                        service, target, "tool_wait", message="keep", pause=pause
                    )
                    assert not children and (not dma.pending)
                    _, children = signal(service, target, "resume", message="resume")
                    assert all(
                        (
                            child.action_type == "sglang.set_priority"
                            for child in children
                        )
                    )
                    assert node.component_data[FULL].value is not None
                    assert not service.has_pending

    def test_abort_during_h2d_drains_before_retiring_and_does_not_apply_dependent_priority(
        self,
    ):
        with transfers.hicache(self, config()) as owner:
            _, cache, service, dma = owner
            _, target = initialize(owner)
            save(owner, target)
            req = begin_resume(owner)
            copy_work = next(
                (
                    w
                    for w in service.policy_runtime.execution.works.values()
                    if w.entry.proposal.prepare_for_admission
                )
            )
            copy_ref = copy_work.ref
            service.finish_request(req, aborted=True)
            assert service.control.registry.get(copy_ref).status == Status.CANCELLING
            assert service.has_pending and service.blocks_prefill
            dma.finish()
            transfers.wait_for(service, lambda: not service.has_pending)
            assert service.control.registry.get(copy_ref).status == Status.CANCELLED
            assert not service.policy_runtime.execution.works and (
                not service.control.executor.preparations
            )

    def test_whole_request_headroom_is_checked_before_prefix_copy(self):
        with transfers.hicache(self, config()) as owner:
            _, cache, service, dma = owner
            _, target = initialize(owner)
            save(owner, target)
            other = cache.token_to_kv_pool_allocator.alloc(60)
            req = policy_request(service, "task")
            observe(cache, service, (req,))
            try:
                assert not service.prepare_admission(req)
                service.progress()
                assert not dma.pending and (not service.control.executor.preparations)
                assert cache.token_to_kv_pool_allocator.available_size() == 4
                cache.token_to_kv_pool_allocator.free(other[:3])
                assert not service.prepare_admission(req)
                transfers.wait_for(service, lambda: bool(dma.pending))
                dma.finish()
                transfers.wait_for(
                    service,
                    lambda: any(
                        (
                            w.ready
                            for w in service.policy_runtime.execution.works.values()
                        )
                    ),
                )
                assert cache.token_to_kv_pool_allocator.available_size() == 3
                service.finish_request(req, aborted=True)
                transfers.wait_for(service, lambda: not service.has_pending)
            finally:
                cache.token_to_kv_pool_allocator.free(other[3:])


FULL = ComponentType.FULL
if __name__ == "__main__":
    unittest.main()
