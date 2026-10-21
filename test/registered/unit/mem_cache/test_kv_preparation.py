# SPDX-License-Identifier: Apache-2.0
"""Real FULL pool references between a completed copy and request admission."""

from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import pytest
import test_kv_control_cache as cache_fixtures
import test_kv_control_storage_reads as reads
import test_kv_control_transfers as transfers

from sglang.srt.managers.kv_control.contracts import (
    Associate,
    ControlError,
    CopyTo,
    Location,
    SessionTarget,
    Status,
)
from sglang.srt.managers.kv_control.service import CacheLimits, ServiceConfig
from sglang.srt.managers.kv_hints import KvHintAction, KvHintsEnvelope
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")
owner = transfers.owner
FULL = ComponentType.FULL


def host_scope(owner):
    _, cache, service, dma = owner
    node = transfers.seed(cache)
    moved = transfers.transfer(service, kind="move_to")
    service.progress()
    dma.finish()
    service.progress()
    assert transfers.current(service, moved).status == Status.COMPLETED
    return node


def register(service, *, source="host", count=1):
    payload = CopyTo(
        target=cache_fixtures.target(),
        src=Location(tier=source),
        dst=Location(tier="gpu"),
    )
    refs = service.control._reserve_policy_operations(
        tuple((str(i), payload) for i in range(count))
    )
    return refs, [service.control.executor.prepare_prefix(ref) for ref in refs]


def test_publication_handoff_has_no_unlock_gap_and_survives_until_request_lock(owner):
    _, cache, service, dma = owner
    node = host_scope(owner)
    (ref,), (lease,) = register(service)
    try:
        assert not lease.ready and not lease.capture()
        service.control._start_policy_operation(ref)
        service.progress()
        assert dma.pending and not lease.ready
        dma.finish()
        unlock = cache.unlock

        def try_evict(lock):
            unlock(lock)
            cache.evict(EvictParams(num_tokens=4))

        with patch.object(cache, "unlock", side_effect=try_evict):
            service.progress()
        assert service.control.registry.get(ref).status == Status.COMPLETED
        assert lease.ready and lease.locked_tokens == 4
        assert service.control.registry.capacity()["active"] == 0
        assert service.has_pending  # Separate scheduling-owner obligation.
        with pytest.raises(ControlError, match="preparation"):
            service.reset_after_cache_flush()
        # PrefillAdder commits the same real cache lock for an admitted request.
        request_lock = cache.lock(node.id)
        try:
            lease.release()
            assert not service.has_pending
            cache.evict(EvictParams(num_tokens=4))
            assert node.component_data[FULL].value is not None
        finally:
            cache.unlock(request_lock)
        cache.evict(EvictParams(num_tokens=4))
        assert node.component_data[FULL].value is None
        assert cache.token_to_kv_pool_allocator.available_size() == 64
    finally:
        lease.release()


def test_shared_copy_has_independent_preparation_owners_without_double_allocation(
    owner,
):
    _, cache, service, dma = owner
    node = host_scope(owner)
    refs, leases = register(service, count=2)
    try:
        for ref in refs:
            service.control._start_policy_operation(ref)
        service.progress()
        assert len(dma.pending) == 1
        dma.finish()
        service.progress()
        assert all(
            service.control.registry.get(ref).status == Status.COMPLETED for ref in refs
        )
        assert all(lease.ready for lease in leases)
        assert node.component_data[FULL].lock_ref == 2
        assert cache.tree_core.full_protected_size() == 4
        assert cache.token_to_kv_pool_allocator.available_size() == 60
        leases[0].release()
        cache.evict(EvictParams(num_tokens=4))
        assert node.component_data[FULL].value is not None
        leases[1].release()
        cache.evict(EvictParams(num_tokens=4))
        assert cache.token_to_kv_pool_allocator.available_size() == 64
    finally:
        for lease in leases:
            lease.release()


@pytest.mark.parametrize(
    "owner",
    [
        ServiceConfig(
            namespace="ns",
            engine_id="engine",
            cache_limits=CacheLimits(max_prepared_prefixes=1),
        )
    ],
    indirect=True,
)
def test_registration_capacity_and_abort_release_without_cancelling_worker_holds(owner):
    _, cache, service, dma = owner
    node = host_scope(owner)
    payload = CopyTo(
        target=cache_fixtures.target(),
        src=Location(tier="host"),
        dst=Location(tier="gpu"),
    )
    a, b = service.control._reserve_policy_operations((("a", payload), ("b", payload)))
    lease = service.control.executor.prepare_prefix(a)
    try:
        with pytest.raises(ControlError, match="capacity"):
            service.control.executor.prepare_prefix(b)
        service.control.cancel(b)
        service.control._start_policy_operation(a)
        service.progress()
        assert dma.pending
        lease.release()
        assert service.control.cancel(a).status == Status.CANCELLING
        assert node.component_data[FULL].lock_ref > 0
        dma.finish()
        service.progress()
        assert service.control.registry.get(a).status == Status.CANCELLED
        assert not lease.capture()
        assert node.component_data[FULL].lock_ref == 0 and not service.has_pending
    finally:
        lease.release()


@pytest.mark.parametrize("owner", ["file_namespaced"], indirect=True)
def test_file_restore_hands_off_before_its_host_and_gpu_holds_retire(owner):
    cache, service, dma = reads.cold(owner)
    (ref,), (lease,) = register(service, source="storage")
    try:
        service.control._start_policy_operation(ref)
        reads.wait_for(service, lambda: service.blocks_prefill)
        service.progress()
        dma.finish()
        reads.wait_for(
            service, lambda: service.control.registry.get(ref).status.terminal
        )
        assert service.control.registry.get(ref).status == Status.COMPLETED
        assert lease.ready and lease.locked_tokens == 4
        cache.evict(EvictParams(num_tokens=4))
        assert cache.token_to_kv_pool_allocator.available_size() == 60
        lease.release()
        assert not service.has_pending
        reads.assert_retired(cache, service)
    finally:
        lease.release()


def test_headroom_loss_during_dma_does_not_grant_a_preparation_hold(owner):
    _, cache, service, dma = owner
    node = host_scope(owner)
    capacity = {"remaining": 4}
    service.control.executor.gpu_copy_capacity = lambda: capacity["remaining"]
    (ref,), (lease,) = register(service)
    try:
        service.control._start_policy_operation(ref)
        service.progress()
        capacity["remaining"] = -1
        dma.finish()
        service.progress()
        assert service.control.registry.get(ref).status == Status.COMPLETED
        assert not lease.ready and lease.code == "preparation_capacity"
        assert node.component_data[FULL].lock_ref == 0
        cache.evict(EvictParams(num_tokens=4))
        assert node.component_data[FULL].value is None
        capacity["remaining"] = 4
        assert not lease.capture()  # Historical COPY success is not readiness.
    finally:
        lease.release()


def test_capture_and_release_follow_ordinary_radix_splits(owner):
    _, cache, service, dma = owner
    host_scope(owner)
    (ref,), (lease,) = register(service)
    try:
        service.control._start_policy_operation(ref)
        service.progress()
        cache.match_prefix(
            cache_fixtures.MatchPrefixParams(
                key=cache_fixtures.RadixKey(cache_fixtures.array("q", [1, 2, 8]))
            )
        )
        dma.finish()
        service.progress()
        assert service.control.registry.get(ref).status == Status.COMPLETED
        assert lease.ready and lease.locked_tokens == 4
        lease.release()
        assert cache.tree_core.full_protected_size() == 0
        cache.evict(EvictParams(num_tokens=4))
        assert cache.token_to_kv_pool_allocator.available_size() == 64
        cache.sanity_check()
    finally:
        lease.release()


def test_explicit_control_cannot_register_internal_preparation(owner):
    _, cache, service, dma = owner
    host_scope(owner)
    public = transfers.transfer(service, src="host", dst="gpu", message="public")
    with pytest.raises(ControlError, match="private"):
        service.control.executor.prepare_prefix(public.ref)
    assert not service.control.executor.preparations
    service.progress()
    dma.finish()
    service.progress()
    assert service.control.registry.get(public.ref).status == Status.COMPLETED


def test_unsupported_request_task_finish_still_cancels_the_real_transfer_owner(owner):
    _, cache, service, dma = owner
    node = transfers.seed(cache)
    service.control.index.associate(
        Associate(
            session_id="task",
            continuation_id="c",
            expected_revision=0,
            target=cache_fixtures.target(),
        )
    )
    movement = transfers.transfer(
        service,
        kind="move_to",
        selection=SessionTarget(namespace="ns", session_id="task"),
    )
    envelope = KvHintsEnvelope(
        protocol_version="0.1",
        message_id="unsupported-finish",
        actions=[
            KvHintAction(
                action_id="finish",
                action_type="sglang.finish_task",
                action_version="1.0",
                payload={
                    "session_id": "task",
                    "expected_revision": service.control.index.session_revision("task"),
                },
            ),
            KvHintAction(
                action_id="priority",
                action_type="sglang.set_priority",
                action_version="1.0",
                payload={
                    "target": msgspec.json.decode(
                        msgspec.json.encode(cache_fixtures.target())
                    ),
                    "location": {"tier": "gpu"},
                    "claim_id": "unsupported",
                    "revision": 1,
                    "value": 90,
                    "ttl_seconds": 10,
                },
            ),
        ],
    )
    request = SimpleNamespace()
    executor = service.control.executor
    service.attach(request, envelope, unsupported_request=True)
    results = service.control.registry.query(
        request.kv_control_context, envelope.message_id
    ).results
    assert [r.status for r in results] == [Status.COMPLETED, Status.UNSUPPORTED]
    assert service.control.executor is executor
    service.progress()
    dma.finish()
    service.progress()
    assert service.control.registry.get(movement.ref).status == Status.CANCELLED
    assert node.component_data[FULL].value is not None
    assert node.component_data[FULL].lock_ref == 0


def test_task_finish_releases_only_its_preparation_while_shared_data_stays_held(owner):
    _, cache, service, dma = owner
    node = transfers.seed(cache)
    copied = transfers.transfer(service)
    service.progress()
    dma.finish()
    service.progress()
    assert service.control.registry.get(copied.ref).status == Status.COMPLETED
    leases, refs = [], []
    try:
        for name in ("A", "B"):
            service.control.index.associate(
                Associate(
                    session_id=name,
                    continuation_id="c",
                    expected_revision=0,
                    target=cache_fixtures.target(),
                )
            )
            payload = CopyTo(
                target=SessionTarget(namespace="ns", session_id=name),
                src=Location(tier="host"),
                dst=Location(tier="gpu"),
            )
            (ref,) = service.control._reserve_policy_operations(((name, payload),))
            refs.append(ref)
            lease = service.control.executor.prepare_prefix(ref)
            leases.append(lease)
            assert lease.capture()
        for i, name in enumerate(("A", "B")):
            envelope = KvHintsEnvelope(
                protocol_version="0.1",
                message_id="finish-" + name,
                actions=[
                    KvHintAction(
                        action_id="finish",
                        action_type="sglang.finish_task",
                        action_version="1.0",
                        payload={
                            "session_id": name,
                            "expected_revision": service.control.index.session_revision(
                                name
                            ),
                        },
                    )
                ],
            )
            result = service.control.submit(envelope, service.control.context("router"))
            assert result.results[0].status == Status.COMPLETED
            assert leases[i].released
            assert service.control.registry.get(refs[i]).status == Status.CANCELLED
            assert node.component_data[FULL].lock_ref == 1 - i
            if i == 0:
                cache.evict(EvictParams(num_tokens=4))
                assert node.component_data[FULL].value is not None
        assert not service.has_pending
        cache.evict(EvictParams(num_tokens=4))
        assert cache.token_to_kv_pool_allocator.available_size() == 64
    finally:
        for lease in leases:
            lease.release()
        for ref in refs:
            service.control.cancel(ref)
