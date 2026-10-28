# SPDX-License-Identifier: Apache-2.0
"""KV copy/move with real CPU pools and file storage; controlled DMA fences."""

import atexit
import json
import threading
import time
import unittest
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from unittest.mock import patch

import msgspec
import test_kv_control_cache as cache_fixtures
import torch

from sglang.srt.managers.kv_control.contracts import (
    Associate,
    Location,
    SessionTarget,
    Status,
)
from sglang.srt.managers.kv_control.service import KVControlService, ServiceConfig
from sglang.srt.managers.kv_hints import KvHintAction, KvHintsEnvelope
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.l2_transfer import TransferCompletion
from sglang.srt.mem_cache.pool_host import common as host_memory
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.runtime_context import get_context, get_parallel, get_server_args
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")
FULL = ComponentType.FULL


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
            (direction, sum((len(t.device_indices) for t in transfers)))
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
            host, device = (transfer.host_pool, transfer.device_pool)
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
                            msgspec.json.encode(
                                selection or cache_fixtures.target(tokens)
                            )
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
    compatibility = executor._compatibility(cache_fixtures.target(tokens))
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
        cache_fixtures.InsertParams(
            key=cache_fixtures.RadixKey(
                cache_fixtures.array("q", tokens), cache_salt=compatibility.cache_salt
            ),
            value=slots.to(torch.int64),
        )
    )
    return cache.tree_core.node_by_id(result.last_device_node)


def branched_scope(cache, service):
    seed(cache, (1, 2, 3, 4))
    seed(cache, (1, 2, 5, 6), value=11)
    for name, tokens in (("a", (1, 2, 3, 4)), ("b", (1, 2, 5, 6))):
        service.control.index.associate(
            Associate(
                session_id="task",
                continuation_id=name,
                expected_revision=0,
                target=cache_fixtures.target(tokens),
            )
        )
    return SessionTarget(namespace="ns", session_id="task")


@contextmanager
def hicache(case, config=None, *, storage=False, namespace_keys=False):
    config = config or ServiceConfig(namespace="ns", engine_id="engine")
    storage = storage or namespace_keys
    storage_dir = TemporaryDirectory(prefix="sglang-kv-test-")
    case.addCleanup(storage_dir.cleanup)
    storage_args = (
        {
            "hicache_storage_backend": "file",
            "hicache_storage_backend_extra_config": json.dumps(
                {"file_storage_path": storage_dir.name}
            ),
        }
        if storage
        else {}
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
            cache.host_pool_group.anchor_entry.host_pool.pin_memory = False
        atexit.unregister(cache.shutdown)
        dma = ControlledDMA()
        cache.cache_controller.l2_transfer_engine = dma
        service = KVControlService(
            cache,
            config,
            clock=case.clock,
            namespace_keys=namespace_keys,
            idle_check=(lambda: True) if namespace_keys else None,
        )
        try:
            yield (case, cache, service, dma)
            assert not service.has_pending, "Test must retire every operation"
            assert not cache.ongoing_write_through and (not cache.ongoing_load_back)
        finally:
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


class TestKVControlTransfers(cache_fixtures.KVCacheTestCase):
    def test_copy_waits_for_fence_preserves_bytes_and_pool_ownership(self):
        with hicache(self) as owner:
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
            assert current(service, result).status == Status.COMPLETED, current(
                service, result
            )
            cd = node.component_data[ComponentType.FULL]
            host = cache.host_pool_group.anchor_entry.host_pool
            assert torch.all(host.k_buffer[0][cd.host_value] == 7)
            assert cd.value is not None and cd.lock_ref == cd.host_lock_ref == 0
            assert cache.token_to_kv_pool_allocator.available_size() == initial
            assert not service.control.executor.transfers.interests
            snapshot = service.control.executor.residency(
                service.control.index.resolve(cache_fixtures.target()),
                location=Location(tier="host"),
            )
            assert snapshot.items[0].location.tier == "host"
            assert snapshot.items[0].lock_ref == 0
            cache.sanity_check()

    def test_shared_waiter_cancel_does_not_cancel_other_interest_or_release_dma(self):
        with hicache(self) as owner:
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

    def test_host_load_holds_whole_prefix_and_blocks_prefill_until_fence(self):
        with hicache(self) as owner:
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
            assert current(service, loaded).status == Status.COMPLETED, current(
                service, loaded
            )
            slots = node.component_data[ComponentType.FULL].value
            pool = cache.token_to_kv_pool_allocator.get_kvcache()
            assert torch.all(pool.k_buffer[0][slots] == 7)
            assert torch.all(pool.v_buffer[0][slots] == 8)
            assert cache.load_back_threshold == 10
            cache.sanity_check()

    def test_shared_prefix_whole_scope_capacity_is_charged_once(self):
        with hicache(self) as owner:
            _, cache, service, dma = owner
            selection = branched_scope(cache, service)
            assert cache.token_to_kv_pool_allocator.available_size() == 58
            saved = transfer(service, kind="move_to", selection=selection)
            service.progress()
            dma.finish()
            service.progress()
            assert current(service, saved).status == Status.COMPLETED, current(
                service, saved
            )
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
            assert current(service, loaded).status == Status.COMPLETED, current(
                service, loaded
            )
            assert cache.tree_core.full_protected_size() == 0
            cache.token_to_kv_pool_allocator.free(other)
            cache.sanity_check()

    def test_insufficient_whole_scope_defers_without_reserving_or_queuing(self):
        with hicache(self) as owner:
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
            assert not cache.cache_controller.load_queue and (not dma.pending)
            assert cache.token_to_kv_pool_allocator.available_size() == 4
            assert cache.tree_core.full_protected_size() == 0
            for node in cache.tree_core._node_arena.values():
                if node is not cache.tree_core.root_node:
                    cd = node.component_data[ComponentType.FULL]
                    assert cd.lock_ref == cd.host_lock_ref == 0
            assert service.control.cancel(loaded.ref).status == Status.CANCELLED
            cache.token_to_kv_pool_allocator.free(other)

    def test_partial_submission_failure_keeps_cause_after_cancel_ack_and_retry(self):
        with hicache(self) as owner:
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
            assert (
                result.status == Status.FAILED and result.update.code == "capacity"
            ), result
            assert any(
                (
                    o.completed_steps == ("destination_ready",)
                    for o in result.update.target_outcomes
                )
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

    def test_native_load_preserves_existing_running_growth_budget_without_double_charge(
        self,
    ):
        with hicache(self) as owner:
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
            service.control.executor.gpu_copy_capacity = (
                scheduler._kv_control_gpu_copy_capacity
            )
            assert scheduler._kv_control_gpu_copy_capacity() == 0
            loaded = transfer(
                service, src="host", dst="gpu", selection=selection, message="load"
            )
            assert (
                loaded.status == Status.QUEUED
                and loaded.update.code == "whole_scope_capacity"
            )
            cache.token_to_kv_pool_allocator.free(other[:6])
            service.progress()
            assert current(service, loaded).status == Status.RUNNING, current(
                service, loaded
            )
            assert cache.token_to_kv_pool_allocator.available_size() == 8
            dma.finish()
            service.progress()
            assert current(service, loaded).status == Status.COMPLETED, current(
                service, loaded
            )
            cache.token_to_kv_pool_allocator.free(other[6:])
            cache.sanity_check()


def wait(service, result):
    deadline = time.monotonic() + 5
    while not current(service, result).status.terminal:
        service.progress()
        assert time.monotonic() < deadline, current(service, result)
        time.sleep(0.005)
    return current(service, result)


def stage(service, dma):
    service.progress()
    if dma.pending:
        dma.finish()
    service.progress()


def assert_files(cache, node):
    cc = cache.cache_controller
    pool = cache.host_pool_group.anchor_entry.host_pool
    indices = node.component_data[FULL].host_value
    hashes = cache.tree_core.get_hash_values(node.id)
    for i, key in enumerate(hashes):
        expected = pool.get_data_page(indices[i]).clone()
        actual = torch.zeros_like(expected)
        assert cc.storage_backend.get(key, actual) is not None
        assert torch.equal(actual, expected)


class TestKVControlStorage(cache_fixtures.KVCacheTestCase):
    def test_copy_and_move_wait_for_storage_worker_not_enqueue(self):
        with hicache(self, storage=True) as owner:
            _, cache, service, dma = owner
            node = seed(cache)
            cc = cache.cache_controller
            entered, release = (threading.Event(), threading.Event())
            original = cc.page_set_func

            def blocked(*args, **kwargs):
                entered.set()
                assert release.wait(5)
                return original(*args, **kwargs)

            try:
                with patch.object(cc, "page_set_func", blocked):
                    result = transfer(service, dst="storage", kind="move_to")
                    stage(service, dma)
                    assert entered.wait(5)
                    for _ in range(3):
                        service.progress()
                    assert current(service, result).status == Status.RUNNING
                    cd = node.component_data[FULL]
                    assert (
                        cd.value is not None
                        and cd.lock_ref > 0
                        and (cd.host_lock_ref > 0)
                    )
                    assert not current(service, result).update.completed_steps
                    assert cache._storage_attachment.detach()[0] is False
                    release.set()
                    done = wait(service, result)
            finally:
                release.set()
            assert done.status == Status.COMPLETED, done
            assert done.update.completed_steps == (
                "destination_ready",
                "source_reclaimed",
            )
            assert node.component_data[FULL].value is None
            assert_files(cache, node)
            assert not service.control.executor.storage.jobs
            assert not cache._native_storage_write_results
            assert cd.lock_ref == cd.host_lock_ref == 0

    def test_short_ack_preserves_source_and_actual_prefix_and_failure_on_ack(self):
        with hicache(self, storage=True) as owner:
            _, cache, service, dma = owner
            node = seed(cache)
            cc = cache.cache_controller
            original = cc.page_set_func
            count = 0

            def fail_second(*args, **kwargs):
                nonlocal count
                count += 1
                return original(*args, **kwargs) if count == 1 else False

            with (
                patch("sglang.srt.managers.cache_controller.STORAGE_BATCH_SIZE", 2),
                patch.object(cc, "page_set_func", fail_second),
            ):
                result = transfer(service, dst="storage", kind="move_to")
                stage(service, dma)
                done = wait(service, result)
            assert (
                done.status == Status.FAILED
                and done.update.code == "incomplete_storage_write"
            ), done
            assert node.component_data[FULL].value is not None
            assert [
                (o.coverage.start_token, o.coverage.end_token, o.completed_steps)
                for o in done.update.target_outcomes
            ] == [(0, 2, ("destination_ready",)), (2, 4, ())]
            acknowledged = service.control.registry.acknowledge(
                result.ref.context, result.ref.message_id
            )
            assert acknowledged.summaries[0].status == Status.FAILED
            assert acknowledged.summaries[0].code == done.update.code
            retry = transfer(service, dst="storage", kind="move_to", message="retry")
            assert wait(service, retry).status == Status.COMPLETED
            assert_files(cache, node)


def cold_prefix(owner):
    _, cache, service, dma = owner
    seed(cache)
    result = transfer(service, kind="move_to", dst="storage")
    stage(service, dma)
    assert wait(service, result).status == Status.COMPLETED
    assert cache.evict_host(4) == 4
    assert not service.control.executor._walk(
        (1, 2, 3, 4), service.control.executor._compatibility(cache_fixtures.target())
    )
    return (cache, service, dma)


def wait_for(service, condition):
    deadline = time.monotonic() + 5
    while not condition():
        service.progress()
        assert time.monotonic() < deadline
        time.sleep(0.005)


def prefix_path(cache, service):
    return service.control.executor._walk(
        (1, 2, 3, 4), service.control.executor._compatibility(cache_fixtures.target())
    )


def assert_retired(cache, service):
    assert not service.control.executor.storage_reads.jobs
    assert not service.control.executor.storage_reads.interests
    assert not cache.ongoing_prefetch
    assert not cache.prefetch_loaded_tokens_by_reqid
    assert not cache.prefetch_loaded_storage_start_by_reqid
    assert not cache._storage_prefetch_hit_remaining_by_reqid
    assert cache.cache_controller.prefetch_tokens_occupied == 0
    assert cache.cache_controller.host_mem_release_queue.empty()


class TestKVControlStorageReads(cache_fixtures.KVCacheTestCase):
    def test_restore_from_actual_file_without_resident_radix_nodes(self):
        with hicache(self, namespace_keys=True) as owner:
            destination = "gpu"
            cache, service, dma = cold_prefix(owner)
            pool = cache.token_to_kv_pool_allocator.get_kvcache()
            pool.k_buffer[0].fill_(-100)
            pool.v_buffer[0].fill_(-101)
            threshold = cache.prefetch_threshold
            assert threshold > 4
            result = transfer(
                service, src="storage", dst=destination, message="restore"
            )
            if destination == "gpu":
                wait_for(service, lambda: service.blocks_prefill)
                assert current(service, result).status == Status.RUNNING
                assert not current(service, result).update.completed_steps
                service.progress()
                assert len(dma.pending) == 1
                dma.finish()
            done = wait(service, result)
            assert done.status == Status.COMPLETED, done
            assert done.update.completed_steps == ("destination_ready",)
            assert cache.prefetch_threshold == threshold
            for node, _, _, _ in prefix_path(cache, service):
                cd = node.component_data[FULL]
                source = (
                    cache.host_pool_group.anchor_entry.host_pool
                    if destination == "host"
                    else pool
                )
                indices = cd.host_value if destination == "host" else cd.value
                assert torch.all(source.k_buffer[0][indices] == 7)
                assert torch.all(source.v_buffer[0][indices] == 8)
                assert cd.lock_ref == cd.host_lock_ref == 0
            assert not service.blocks_prefill
            assert_retired(cache, service)

    def test_storage_read_cancel_waits_for_actual_worker_exit(self):
        with hicache(self, namespace_keys=True) as owner:
            cache, service, dma = cold_prefix(owner)
            cc = cache.cache_controller
            entered, release = (threading.Event(), threading.Event())
            original = cc.page_get_func

            def blocked(*args, **kwargs):
                entered.set()
                assert release.wait(5)
                return original(*args, **kwargs)

            try:
                with patch.object(cc, "page_get_func", blocked):
                    result = transfer(
                        service, src="storage", dst="gpu", message="restore"
                    )
                    wait_for(service, entered.is_set)
                    assert (
                        service.control.cancel(result.ref).status == Status.CANCELLING
                    )
                    for _ in range(3):
                        service.progress()
                    assert service.has_pending
                    assert cache.ongoing_prefetch
                    release.set()
                    done = wait(service, result)
            finally:
                release.set()
            assert done.status == Status.CANCELLED, done
            assert not dma.pending
            assert_retired(cache, service)


if __name__ == "__main__":
    unittest.main()
