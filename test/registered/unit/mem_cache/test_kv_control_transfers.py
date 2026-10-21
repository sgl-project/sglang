# SPDX-License-Identifier: Apache-2.0
"""Real CPU HiCache pools/controller; only DMA and its fence are controlled."""

import atexit
import json
from unittest.mock import patch

import msgspec
import pytest
import test_kv_control_cache as fixtures
import torch

from sglang.srt.managers.kv_control.contracts import (
    Associate,
    Limits,
    Location,
    SessionTarget,
    Status,
)
from sglang.srt.managers.kv_control.service import (
    CacheLimits,
    KVControlService,
    ServiceConfig,
)
from sglang.srt.managers.kv_hints import KvHintAction, KvHintsEnvelope
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.l2_transfer import TransferCompletion
from sglang.srt.mem_cache.pool_host import common as host_memory
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.runtime_context import get_context, get_parallel, get_server_args
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class Fence:
    def __init__(self, ready=False):
        self.ready = ready

    def query(self):
        return self.ready

    def synchronize(self):
        assert self.ready, "Prematurely waited for an unfinished transfer"


class ControlledDMA:
    def __init__(self):
        self.pending = []
        self.submitted = []

    def submit(self, direction, transfers, on_layer_done=None):
        event = Fence()
        self.pending.append((direction, transfers, event, on_layer_done))
        self.submitted.append(
            (direction, sum(len(t.device_indices) for t in transfers))
        )
        return TransferCompletion(Fence(True), event, False)

    def submit_device_to_host(self, transfers):
        return self.submit("save", transfers)

    def submit_host_to_device(
        self, transfers, *, transfer_layer_id_max, start_event=None, on_layer_done=None
    ):
        return self.submit("load", transfers, on_layer_done)

    def finish(self):
        direction, transfers, event, on_layer_done = self.pending.pop(0)
        for transfer in transfers:
            host, device = transfer.host_pool, transfer.device_pool
            for layer in range(device.layer_num):
                for h, d in (
                    (host.k_buffer[layer], device.get_key_buffer(layer)),
                    (host.v_buffer[layer], device.get_value_buffer(layer)),
                ):
                    if direction == "save":
                        h[transfer.host_indices] = d[transfer.device_indices]
                    else:
                        d[transfer.device_indices] = h[transfer.host_indices]
                if on_layer_done is not None:
                    on_layer_done(layer)
        event.ready = True


@pytest.fixture
def owner(request):
    assert torch.cuda.device_count() == 0
    case = fixtures.TestKVControlCache()
    case.setUp()
    fixture_kind = getattr(request, "param", None)
    storage = fixture_kind in ("file", "file_namespaced")
    storage_args = {}
    if storage:
        storage_args = dict(
            hicache_storage_backend="file",
            hicache_storage_backend_extra_config=json.dumps(
                {"file_storage_path": str(request.getfixturevalue("tmp_path"))}
            ),
        )
    scope = get_context().override_server_args(
        device="cpu",
        enable_hierarchical_cache=True,
        hicache_ratio=2,
        hicache_write_policy="write_back",
        hicache_mem_layout="layer_first",
        disable_overlap_schedule=True,
        **storage_args,
    )
    scope.install()
    parallel = get_parallel().override(tp_rank=0, tp_size=1, pp_rank=0, pp_size=1)
    parallel.__enter__()
    try:
        config = (
            None if storage else getattr(request, "param", None)
        ) or ServiceConfig(namespace="ns", engine_id="engine")
        cache, _ = case.make(
            session=False, config=config.cache_config(), limits=config.limits
        )
        cache.reset()
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=cache.req_to_token_pool,
            token_to_kv_pool_allocator=cache.token_to_kv_pool_allocator,
            page_size=1,
            tree_components=(ComponentType.FULL,),
            kv_control_config=config.cache_config(),
        )
        with patch.object(host_memory, "_cuda_host_register"):
            cache.init_hicache(get_server_args(), params)
        if storage:
            # This fixture has ordinary CPU tensors and no CUDA allocator.
            # File-read scratch pages must use the same unpinned CPU allocation.
            cache.host_pool_group.anchor_entry.host_pool.pin_memory = False
        atexit.unregister(cache.shutdown)
        dma = ControlledDMA()
        cache.cache_controller.l2_transfer_engine = dma
        service = KVControlService(
            cache,
            config,
            clock=case.clock,
            namespace_keys=fixture_kind == "file_namespaced",
            idle_check=(lambda: True) if fixture_kind == "file_namespaced" else None,
        )
        try:
            yield case, cache, service, dma
            assert not service.has_pending, "Test must retire every operation"
            assert not cache.ongoing_write_through and not cache.ongoing_load_back
        finally:
            # Retire controlled CPU workers even when a failing assertion left
            # them outstanding. This cleanup is not part of the passing proof.
            for pending in tuple(service.control._pending.values()):
                service.control.cancel(pending.ref)
            service.progress()
            while dma.pending:
                dma.finish()
            service.progress()
            if storage:
                cache._storage_attachment.shutdown()
            cache.release_host_resources()
    finally:
        parallel.__exit__(None, None, None)
        scope.restore()
        case.doCleanups()


def transfer(
    service,
    *,
    kind="copy_to",
    src="gpu",
    dst="host",
    caller="A",
    message="copy",
    tokens=(1, 2, 3, 4),
    selection=None,
    submission=False,
):
    result = service.control.submit(
        KvHintsEnvelope(
            protocol_version="0.1",
            message_id=message,
            actions=[
                KvHintAction(
                    action_id="a",
                    action_type="sglang." + kind,
                    action_version="1.0",
                    payload={
                        "target": msgspec.json.decode(
                            msgspec.json.encode(selection or fixtures.target(tokens))
                        ),
                        "src": {"tier": src},
                        "dst": {"tier": dst},
                    },
                )
            ],
        ),
        service.control.context(caller),
    )
    return result if submission else result.results[0]


def current(service, result):
    return service.control.registry.get(result.ref)


def seed(cache, tokens=(1, 2, 3, 4), value=7):
    slots = cache.token_to_kv_pool_allocator.alloc(len(tokens))
    assert slots is not None
    pool = cache.token_to_kv_pool_allocator.get_kvcache()
    pool.k_buffer[0][slots] = value
    pool.v_buffer[0][slots] = value + 1
    executor = cache.kv_control_service.control.executor
    compatibility = executor._compatibility(fixtures.target(tokens))
    # Simulate computing the same canonical prefix when a new branch overlaps
    # resident or host-only content. Only the new suffix gets the new marker.
    for node, start, _, matched in executor._walk(tokens, compatibility):
        cd = node.component_data[ComponentType.FULL]
        old = cd.value if cd.value is not None else cd.host_value
        if old is not None:
            source = (
                pool
                if cd.value is not None
                else cache.host_pool_group.anchor_entry.host_pool
            )
            length = matched - start
            pool.k_buffer[0][slots[start:matched]] = source.k_buffer[0][old[:length]]
            pool.v_buffer[0][slots[start:matched]] = source.v_buffer[0][old[:length]]
    result = cache.insert(
        fixtures.InsertParams(
            key=fixtures.RadixKey(
                fixtures.array("q", tokens), cache_salt=compatibility.cache_salt
            ),
            value=slots.to(torch.int64),
        )
    )
    return cache.tree_core.node_by_id(result.last_device_node)


def test_copy_waits_for_fence_preserves_bytes_and_pool_ownership(owner):
    _, cache, service, dma = owner
    node = seed(cache)
    initial = cache.token_to_kv_pool_allocator.available_size()
    result = transfer(service)
    assert result.status == Status.RUNNING
    service.progress()
    assert current(service, result).status == Status.RUNNING
    assert node.component_data[ComponentType.FULL].lock_ref > 0
    assert node not in cache.tree_core.evictable_device_leaves
    assert cache.tree_core.full_protected_size() == 4
    assert len(dma.pending) == 1
    dma.finish()
    service.progress()
    assert current(service, result).status == Status.COMPLETED, current(service, result)
    cd = node.component_data[ComponentType.FULL]
    host = cache.host_pool_group.anchor_entry.host_pool
    assert torch.all(host.k_buffer[0][cd.host_value] == 7)
    assert cd.value is not None and cd.lock_ref == cd.host_lock_ref == 0
    assert cache.token_to_kv_pool_allocator.available_size() == initial
    assert not service.control.executor.transfers.interests
    snapshot = service.control.executor.residency(
        service.control.index.resolve(fixtures.target()), location=Location(tier="host")
    )
    assert snapshot.items[0].location.tier == "host"
    assert snapshot.items[0].lock_ref == 0
    cache.sanity_check()


def test_move_stores_before_reclaim_and_reports_shared_request_source(owner):
    _, cache, service, dma = owner
    node = seed(cache)
    request_lock = cache.lock(node.id)
    result = transfer(service, kind="move_to")
    service.progress()
    assert node.component_data[ComponentType.FULL].value is not None
    dma.finish()
    service.progress()
    status = current(service, result)
    assert status.status == Status.PARTIAL, status
    assert status.update.target_outcomes[0].completed_steps == ("destination_ready",)
    assert node.component_data[ComponentType.FULL].value is not None
    cache.unlock(request_lock)
    retry = transfer(service, kind="move_to", message="move-retry")
    service.progress()
    assert current(service, retry).status == Status.COMPLETED, current(service, retry)
    assert node.component_data[ComponentType.FULL].value is None
    assert cache.token_to_kv_pool_allocator.available_size() == 64
    assert len(dma.submitted) == 1
    cache.sanity_check()


def test_shared_waiter_cancel_does_not_cancel_other_interest_or_release_dma(owner):
    _, cache, service, dma = owner
    node = seed(cache)
    first = transfer(service, caller="A")
    second = transfer(service, caller="B")
    service.progress()
    service.control.cancel(first.ref)
    assert current(service, first).status == Status.CANCELLED
    assert node.component_data[ComponentType.FULL].lock_ref > 0
    assert len(dma.submitted) == 1
    service.control.cancel(second.ref)
    assert current(service, second).status == Status.CANCELLING
    assert node.component_data[ComponentType.FULL].lock_ref > 0
    dma.finish()
    service.progress()
    assert current(service, second).status == Status.CANCELLED
    assert node.component_data[ComponentType.FULL].lock_ref == 0
    assert not service.control.executor.transfers.jobs
    cache.sanity_check()


def test_host_load_holds_whole_prefix_and_blocks_prefill_until_fence(owner):
    _, cache, service, dma = owner
    node = seed(cache)
    saved = transfer(service, kind="move_to")
    service.progress()
    dma.finish()
    service.progress()
    assert current(service, saved).status == Status.COMPLETED
    pool = cache.token_to_kv_pool_allocator.get_kvcache()
    pool.k_buffer[0].fill_(-7)
    pool.v_buffer[0].fill_(-8)
    loaded = transfer(service, src="host", dst="gpu", message="load")
    assert service.blocks_prefill
    service.progress()
    assert current(service, loaded).status == Status.RUNNING
    assert node.component_data[ComponentType.FULL].value is not None
    assert node.component_data[ComponentType.FULL].lock_ref > 0
    assert service.blocks_prefill
    dma.finish()
    service.progress()
    assert not service.blocks_prefill
    assert current(service, loaded).status == Status.COMPLETED, current(service, loaded)
    slots = node.component_data[ComponentType.FULL].value
    pool = cache.token_to_kv_pool_allocator.get_kvcache()
    assert torch.all(pool.k_buffer[0][slots] == 7)
    assert torch.all(pool.v_buffer[0][slots] == 8)
    assert cache.load_back_threshold == 10
    cache.sanity_check()


def branched_scope(cache, service):
    seed(cache, (1, 2, 3, 4))
    seed(cache, (1, 2, 5, 6), value=11)
    for name, tokens in (("a", (1, 2, 3, 4)), ("b", (1, 2, 5, 6))):
        service.control.index.associate(
            Associate(
                session_id="task",
                continuation_id=name,
                expected_revision=0,
                target=fixtures.target(tokens),
            )
        )
    return SessionTarget(namespace="ns", session_id="task")


def test_shared_prefix_whole_scope_capacity_is_charged_once(owner):
    _, cache, service, dma = owner
    selection = branched_scope(cache, service)
    assert cache.token_to_kv_pool_allocator.available_size() == 58
    saved = transfer(service, kind="move_to", selection=selection)
    service.progress()
    dma.finish()
    service.progress()
    assert current(service, saved).status == Status.COMPLETED, current(service, saved)
    assert cache.token_to_kv_pool_allocator.available_size() == 64
    other = cache.token_to_kv_pool_allocator.alloc(58)
    loaded = transfer(
        service, src="host", dst="gpu", selection=selection, message="load"
    )
    assert loaded.status == Status.RUNNING, loaded
    assert cache.token_to_kv_pool_allocator.available_size() == 0
    service.progress()
    assert dma.submitted == [("save", 6), ("load", 6)]
    assert cache.tree_core.full_protected_size() == 6
    dma.finish()
    service.progress()
    assert current(service, loaded).status == Status.COMPLETED, current(service, loaded)
    assert cache.tree_core.full_protected_size() == 0
    cache.token_to_kv_pool_allocator.free(other)
    cache.sanity_check()


def test_insufficient_whole_scope_defers_without_reserving_or_queuing(owner):
    _, cache, service, dma = owner
    selection = branched_scope(cache, service)
    saved = transfer(service, kind="move_to", selection=selection)
    service.progress()
    dma.finish()
    service.progress()
    assert current(service, saved).status == Status.COMPLETED
    other = cache.token_to_kv_pool_allocator.alloc(60)
    loaded = transfer(
        service, src="host", dst="gpu", selection=selection, message="load"
    )
    assert loaded.status == Status.QUEUED, loaded
    service.progress()
    assert not cache.cache_controller.load_queue and not dma.pending
    assert cache.token_to_kv_pool_allocator.available_size() == 4
    assert cache.tree_core.full_protected_size() == 0
    for node in cache.tree_core._node_arena.values():
        if node is not cache.tree_core.root_node:
            cd = node.component_data[ComponentType.FULL]
            assert cd.lock_ref == cd.host_lock_ref == 0
    assert service.control.cancel(loaded.ref).status == Status.CANCELLED
    cache.token_to_kv_pool_allocator.free(other)


def test_partial_submission_failure_keeps_cause_after_cancel_ack_and_retry(owner):
    _, cache, service, dma = owner
    seed(cache, (1, 2), value=7)
    seed(cache, (1, 2, 3, 4), value=11)
    controller = cache.cache_controller
    original = controller.write
    calls = 0

    def write(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs) if calls == 1 else None

    with patch.object(controller, "write", side_effect=write):
        failed = transfer(service)
    assert failed.status == Status.CANCELLING, failed
    assert failed.update.code == "capacity"
    service.control.cancel(failed.ref)
    service.progress()
    assert current(service, failed).status == Status.CANCELLING
    assert cache.tree_core.full_protected_size() == 4
    dma.finish()
    service.progress()
    result = current(service, failed)
    assert result.status == Status.FAILED and result.update.code == "capacity", result
    assert any(
        o.completed_steps == ("destination_ready",)
        for o in result.update.target_outcomes
    )
    acknowledgement = service.control.registry.acknowledge(
        failed.ref.context, failed.ref.message_id
    )
    assert acknowledgement.summaries[0].status == Status.FAILED
    replay = transfer(service, submission=True)
    assert replay.replayed and replay.acknowledged
    assert (
        replay.summaries[0].status == Status.FAILED
        and replay.summaries[0].code == "capacity"
    )
    retried = transfer(service, message="retry")
    service.progress()
    dma.finish()
    service.progress()
    assert current(service, retried).status == Status.COMPLETED, current(
        service, retried
    )
    assert cache.tree_core.full_protected_size() == 0
    cache.sanity_check()


def test_split_during_dma_preserves_holds_and_late_cancel_keeps_observed_copy(owner):
    _, cache, service, dma = owner
    tokens = tuple(range(8))
    seed(cache, tokens, value=13)
    copied = transfer(service, tokens=tokens)
    service.progress()
    cache.match_prefix(
        MatchPrefixParams(key=fixtures.RadixKey(fixtures.array("q", tokens[:4])))
    )
    assert cache.tree_core.full_protected_size() == 8
    dma.finish()
    service.control.executor.begin_poll()
    assert current(service, copied).status == Status.RUNNING
    cancelled = service.control.cancel(copied.ref)
    assert cancelled.status == Status.CANCELLED
    assert cancelled.update.completed_steps == ("destination_ready",)
    assert [
        (o.coverage.start_token, o.coverage.end_token)
        for o in cancelled.update.target_outcomes
    ] == [(0, 8)]
    assert cache.tree_core.full_protected_size() == 0
    for node in cache.tree_core._node_arena.values():
        if node is not cache.tree_core.root_node:
            cd = node.component_data[ComponentType.FULL]
            assert cd.lock_ref == cd.host_lock_ref == 0
    cache.sanity_check()


@pytest.mark.parametrize(
    "owner",
    [
        ServiceConfig(
            namespace="ns", engine_id="engine", limits=Limits(max_scope_spans=1)
        ),
        ServiceConfig(
            namespace="ns",
            engine_id="engine",
            cache_limits=CacheLimits(max_query_nodes=3),
        ),
    ],
    indirect=True,
)
def test_result_and_split_traversal_budgets_reject_before_any_transfer(owner):
    _, cache, service, dma = owner
    node = seed(cache)
    host_available = cache.cache_controller.mem_pool_host.available_size()
    result = transfer(service)
    assert result.status == Status.REJECTED and result.update.code == "capacity", result
    assert not dma.pending and not dma.submitted
    assert not cache.cache_controller.write_queue and not cache.ongoing_write_through
    assert node.component_data[ComponentType.FULL].host_value is None
    assert node.component_data[ComponentType.FULL].lock_ref == 0
    assert cache.cache_controller.mem_pool_host.available_size() == host_available
    assert not service.control.executor.transfers.jobs


def test_native_profile_rejects_weight_mutation_before_touching_runners(owner):
    from types import SimpleNamespace

    from sglang.srt.managers.scheduler_components.weight_updater import (
        SchedulerWeightUpdaterManager,
    )

    _, cache, service, dma = owner
    manager = object.__new__(SchedulerWeightUpdaterManager)
    manager.scheduler = SimpleNamespace(kv_control_service=service)
    # Runner and communication attributes are deliberately absent: each typed
    # refusal must occur before accessing a runner, collective or tensor input.
    for name in (
        "begin_weight_update",
        "update_weights_from_disk",
        "update_weights_from_tensor",
        "update_weights_from_distributed",
        "update_weights_from_ipc",
    ):
        result = getattr(manager, name)(SimpleNamespace())
        assert not result.success and "weight-namespace isolation" in result.message


def test_native_load_preserves_existing_running_growth_budget_without_double_charge(
    owner,
):
    from array import array
    from types import SimpleNamespace

    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.sampling.sampling_params import SamplingParams

    _, cache, service, dma = owner
    selection = branched_scope(cache, service)
    saved = transfer(service, kind="move_to", selection=selection)
    service.progress()
    dma.finish()
    service.progress()
    assert current(service, saved).status == Status.COMPLETED
    other = cache.token_to_kv_pool_allocator.alloc(56)
    running = Req(
        rid="running",
        origin_input_text=None,
        origin_input_ids=array("q", [99]),
        sampling_params=SamplingParams(max_new_tokens=8, ignore_eos=True),
        vocab_size=128,
    )
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.tree_cache = cache
    scheduler.token_to_kv_pool_allocator = cache.token_to_kv_pool_allocator
    scheduler.running_batch = SimpleNamespace(reqs=[running])
    scheduler.chunked_req = running
    scheduler.new_token_ratio_tracker = SimpleNamespace(current=0.01)
    service.control.executor.gpu_copy_capacity = scheduler._kv_control_gpu_copy_capacity
    assert scheduler._kv_control_gpu_copy_capacity() == 0
    loaded = transfer(
        service, src="host", dst="gpu", selection=selection, message="load"
    )
    assert (
        loaded.status == Status.QUEUED and loaded.update.code == "whole_scope_capacity"
    )
    # Another request releases six physical slots. The remaining eight slots
    # of running decode growth stay protected, and the shared six-token prefix
    # fits exactly once rather than charging eight logical branch tokens.
    cache.token_to_kv_pool_allocator.free(other[:6])
    service.progress()
    assert current(service, loaded).status == Status.RUNNING, current(service, loaded)
    assert cache.token_to_kv_pool_allocator.available_size() == 8
    dma.finish()
    service.progress()
    assert current(service, loaded).status == Status.COMPLETED, current(service, loaded)
    cache.token_to_kv_pool_allocator.free(other[6:])
    cache.sanity_check()


def test_scope_larger_than_device_pool_defers_terminally_without_idle_leak(owner):
    _, cache, service, dma = owner
    branches = [
        tuple([1, 2] + list(range(10, 48))),
        tuple([1, 2] + list(range(100, 138))),
    ]
    for i, tokens in enumerate(branches):
        seed(cache, tokens)
        saved = transfer(service, kind="move_to", tokens=tokens, message=f"save-{i}")
        service.progress()
        dma.finish()
        service.progress()
        assert current(service, saved).status == Status.COMPLETED, current(
            service, saved
        )
        service.control.index.associate(
            Associate(
                session_id="large",
                continuation_id=str(i),
                expected_revision=0,
                target=fixtures.target(tokens),
            )
        )
    loaded = transfer(
        service,
        src="host",
        dst="gpu",
        selection=SessionTarget(namespace="ns", session_id="large"),
        message="load",
    )
    assert loaded.status == Status.DEFERRED and loaded.update.quiescent, loaded
    assert loaded.update.code == "scope_exceeds_destination_capacity"
    assert not service.has_pending
    assert (
        not service.control.executor.transfers.jobs
        and not service.control.executor.transfers.interests
    )
    assert cache.token_to_kv_pool_allocator.available_size() == 64
    assert not dma.pending
    cache.sanity_check()


def test_cache_flush_rebind_preserves_scheduler_resource_approval(owner):
    _, cache, service, _ = owner

    def budget():
        return 17

    service.control.executor.gpu_copy_capacity = budget
    previous = service.control.registry.engine_epoch
    cache.reset()
    service.reset_after_cache_flush()
    assert service.control.registry.engine_epoch != previous
    assert service.control.executor.gpu_copy_capacity is budget
    assert service.control.executor.gpu_copy_capacity() == 17
    assert (
        service.control.executor.capabilities()["directions"]["gpu_copy_approval"]
        == "scheduler_prefill_budget"
    )
