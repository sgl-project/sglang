# SPDX-License-Identifier: Apache-2.0
"""Actual file-to-CPU KV restoration, with explicit worker completion barriers."""

import threading
import time
from unittest.mock import patch

import pytest
import test_kv_control_cache as cache_fixtures
import test_kv_control_storage as writes
import test_kv_control_transfers as transfers
import torch

from sglang.srt.managers.kv_control.contracts import Status, TokensTarget
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")
pytestmark = pytest.mark.parametrize("owner", ["file_namespaced"], indirect=True)
owner = transfers.owner
FULL = ComponentType.FULL


def cold(owner):
    _, cache, service, dma = owner
    transfers.seed(cache)
    result = transfers.transfer(service, kind="move_to", dst="storage")
    writes.stage(service, dma)
    assert writes.wait(service, result).status == Status.COMPLETED
    assert cache.evict_host(4) == 4
    assert not service.control.executor._walk(
        (1, 2, 3, 4), service.control.executor._compatibility(cache_fixtures.target())
    )
    return cache, service, dma


def wait_for(service, condition):
    deadline = time.monotonic() + 5
    while not condition():
        service.progress()
        assert time.monotonic() < deadline
        time.sleep(0.005)


def path(cache, service):
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


@pytest.mark.parametrize("destination", ["host", "gpu"])
def test_restore_from_actual_file_without_resident_radix_nodes(owner, destination):
    cache, service, dma = cold(owner)
    pool = cache.token_to_kv_pool_allocator.get_kvcache()
    pool.k_buffer[0].fill_(-100)
    pool.v_buffer[0].fill_(-101)
    # Ordinary threshold remains large; explicit restoration supports 4 tokens
    # without mutating normal prefetch policy.
    threshold = cache.prefetch_threshold
    assert threshold > 4
    result = transfers.transfer(
        service, src="storage", dst=destination, message="restore"
    )
    if destination == "gpu":
        wait_for(service, lambda: service.blocks_prefill)
        assert transfers.current(service, result).status == Status.RUNNING
        assert not transfers.current(service, result).update.completed_steps
        service.progress()  # Submit the queued native H2D batch.
        assert len(dma.pending) == 1
        dma.finish()
    done = writes.wait(service, result)
    assert done.status == Status.COMPLETED, done
    assert done.update.completed_steps == ("destination_ready",)
    assert cache.prefetch_threshold == threshold
    for node, _, _, _ in path(cache, service):
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


def test_storage_read_cancel_waits_for_actual_worker_exit(owner):
    cache, service, dma = cold(owner)
    cc = cache.cache_controller
    entered, release = threading.Event(), threading.Event()
    original = cc.page_get_func

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    try:
        with patch.object(cc, "page_get_func", blocked):
            result = transfers.transfer(
                service, src="storage", dst="gpu", message="restore"
            )
            wait_for(service, entered.is_set)
            assert service.control.cancel(result.ref).status == Status.CANCELLING
            for _ in range(3):
                service.progress()
            assert service.has_pending
            assert cache.ongoing_prefetch
            release.set()
            done = writes.wait(service, result)
    finally:
        release.set()
    assert done.status == Status.CANCELLED, done
    assert not dma.pending
    assert_retired(cache, service)


def test_shared_read_one_cancel_keeps_other_reader(owner):
    cache, service, dma = cold(owner)
    cc = cache.cache_controller
    entered, release = threading.Event(), threading.Event()
    original = cc.page_get_func
    calls = []

    def blocked(*args, **kwargs):
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    try:
        with patch.object(cc, "page_get_func", blocked):
            a = transfers.transfer(
                service, src="storage", dst="host", message="restore"
            )
            b = transfers.transfer(
                service, src="storage", dst="host", caller="B", message="restore"
            )
            wait_for(service, entered.is_set)
            assert service.control.cancel(a.ref).status == Status.CANCELLED
            assert service.has_pending
            release.set()
            assert writes.wait(service, b).status == Status.COMPLETED
    finally:
        release.set()
    assert len(calls) == 1
    assert_retired(cache, service)


def test_read_shortfall_reports_real_host_prefix_and_does_not_load_gpu(owner):
    cache, service, dma = cold(owner)
    cc = cache.cache_controller
    original = cc.page_get_func
    count = 0

    def short(*args, **kwargs):
        nonlocal count
        count += 1
        return original(*args, **kwargs) if count == 1 else 0

    with (
        patch("sglang.srt.managers.cache_controller.STORAGE_BATCH_SIZE", 2),
        patch.object(cc, "page_get_func", short),
    ):
        result = transfers.transfer(
            service, src="storage", dst="host", message="restore"
        )
        done = writes.wait(service, result)
    assert (
        done.status == Status.FAILED and done.update.code == "incomplete_storage_read"
    ), done
    assert [
        (o.coverage.start_token, o.coverage.end_token, o.completed_steps)
        for o in done.update.target_outcomes
    ] == [(0, 2, ("destination_ready",)), (2, 4, ())]
    assert len(dma.submitted) == 1
    assert_retired(cache, service)


def test_missing_cold_prefix_and_early_query_cancel_retire_without_worker_leaks(owner):
    _, cache, service, dma = owner
    result = transfers.transfer(service, src="storage", dst="host")
    assert writes.wait(service, result).status == Status.MISSING
    assert_retired(cache, service)
    result = transfers.transfer(
        service, src="storage", dst="host", message="query-cancel"
    )
    service.control.cancel(result.ref)
    assert writes.wait(service, result).status == Status.CANCELLED
    assert not dma.pending
    assert_retired(cache, service)


def test_existing_host_prefix_is_charged_once_before_suffix_read(owner):
    cache, service, dma = cold(owner)
    prefix = transfers.transfer(
        service, tokens=(1, 2), src="storage", dst="host", message="prefix"
    )
    assert writes.wait(service, prefix).status == Status.COMPLETED
    pool = cache.host_pool_group.anchor_entry.host_pool
    unrelated = pool.alloc(pool.available_size() - 2)
    try:
        assert pool.available_size() == 2
        result = transfers.transfer(
            service, src="storage", dst="host", message="suffix"
        )
        assert writes.wait(service, result).status == Status.COMPLETED
        assert pool.available_size() == 0
        assert len(path(cache, service)) == 2
        assert_retired(cache, service)
    finally:
        pool.free(unrelated)


def test_gpu_capacity_waits_without_pinning_or_starting_disk_io(owner):
    cache, service, dma = cold(owner)
    executor = service.control.executor
    executor.gpu_copy_capacity = lambda: 0
    result = transfers.transfer(
        service, src="storage", dst="gpu", message="no-headroom"
    )
    assert result.status == Status.QUEUED and result.update.quiescent
    service.progress()
    assert not cache.ongoing_prefetch and not dma.pending
    job = next(iter(executor.storage_reads.jobs.values()))
    assert not job.gpu_locks and not job.host_locks
    assert service.control.cancel(result.ref).status == Status.CANCELLED
    assert_retired(cache, service)


def test_cancel_after_storage_publication_drains_gpu_dma(owner):
    cache, service, dma = cold(owner)
    result = transfers.transfer(service, src="storage", dst="gpu", message="gpu-cancel")
    wait_for(service, lambda: service.blocks_prefill)
    service.progress()
    assert dma.pending
    assert service.control.cancel(result.ref).status == Status.CANCELLING
    assert service.blocks_prefill
    dma.finish()
    done = writes.wait(service, result)
    assert done.status == Status.CANCELLED
    assert not service.blocks_prefill
    assert_retired(cache, service)


def test_namespace_change_cannot_load_old_generation_file_pages(owner):
    cache, service, dma = cold(owner)
    service.rebind_namespace("different-weights")
    target = TokensTarget(namespace="different-weights", token_ids=(1, 2, 3, 4))
    result = transfers.transfer(
        service, src="storage", dst="host", selection=target, message="new-weights"
    )
    assert writes.wait(service, result).status == Status.MISSING
    assert not service.control.executor._walk(
        target.token_ids, service.control.executor._compatibility(target)
    )
    assert not dma.pending
    assert_retired(cache, service)


def test_late_cancel_preserves_observed_failed_read_ack(owner):
    cache, service, dma = cold(owner)
    with patch.object(cache.cache_controller, "page_get_func", return_value=0):
        result = transfers.transfer(
            service, src="storage", dst="host", message="failed-ack"
        )
        job = next(iter(service.control.executor.storage_reads.jobs.values()))
        deadline = time.monotonic() + 5
        while not getattr(job.operation, "native_quiescent", False):
            # The real cache owner publishes the ACK; don't poll the logical
            # interest until cancellation races that observed failed result.
            cache.drain_storage_control_queues()
            assert time.monotonic() < deadline
            time.sleep(0.005)
        done = service.control.cancel(result.ref)
    assert done.status == Status.FAILED, done
    assert done.update.code == "incomplete_storage_read"
    assert_retired(cache, service)
