# SPDX-License-Identifier: Apache-2.0
"""Real file storage and CPU KV pools, controlled only at worker/DMA boundaries."""

import threading
import time
from array import array
from pathlib import Path
from unittest.mock import patch

import pytest
import test_kv_control_cache as cache_fixtures
import test_kv_control_transfers as transfer_fixtures
import torch
from test_kv_control_transfers import (
    current,
    seed,
    transfer,
)

from sglang.srt.managers.kv_control.contracts import Status
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")
pytestmark = pytest.mark.parametrize("owner", ["file"], indirect=True)
FULL = ComponentType.FULL
owner = transfer_fixtures.owner


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


def test_copy_and_move_wait_for_storage_worker_not_enqueue(owner):
    _, cache, service, dma = owner
    node = seed(cache)
    cc = cache.cache_controller
    entered, release = threading.Event(), threading.Event()
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
            assert cd.value is not None and cd.lock_ref > 0 and cd.host_lock_ref > 0
            assert not current(service, result).update.completed_steps
            assert cache._storage_attachment.detach()[0] is False
            release.set()
            done = wait(service, result)
    finally:
        release.set()
    assert done.status == Status.COMPLETED, done
    assert done.update.completed_steps == ("destination_ready", "source_reclaimed")
    assert node.component_data[FULL].value is None
    assert_files(cache, node)
    assert not service.control.executor.storage.jobs
    assert not cache._native_storage_write_results
    assert cd.lock_ref == cd.host_lock_ref == 0


def test_explicit_host_copy_does_not_implicitly_write_storage(owner):
    _, cache, service, dma = owner
    node = seed(cache)
    result = transfer(service)
    stage(service, dma)
    assert wait(service, result).status == Status.COMPLETED
    backend = cache.cache_controller.storage_backend
    assert backend.batch_exists(cache.tree_core.get_hash_values(node.id)) == 0
    result = transfer(service, src="host", dst="storage", message="cold")
    assert wait(service, result).status == Status.COMPLETED
    assert len(dma.submitted) == 1  # No H2D for host-source writes.
    assert_files(cache, node)


@pytest.mark.parametrize("cancel_all", [False, True])
def test_shared_write_cancel_has_independent_lifetimes(owner, cancel_all):
    _, cache, service, dma = owner
    node = seed(cache)
    cc = cache.cache_controller
    entered, release = threading.Event(), threading.Event()
    original = cc.page_set_func
    calls = []

    def blocked(*args, **kwargs):
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    try:
        with patch.object(cc, "page_set_func", blocked):
            a = transfer(service, dst="storage", kind="move_to")
            b = transfer(service, dst="storage", kind="move_to", caller="B")
            stage(service, dma)
            assert entered.wait(5)
            assert service.control.cancel(a.ref).status == Status.CANCELLED
            if cancel_all:
                assert service.control.cancel(b.ref).status == Status.CANCELLING
                assert node.component_data[FULL].host_lock_ref > 0
                assert service.has_pending
            release.set()
            done = wait(service, b)
    finally:
        release.set()
    assert len(calls) == 1
    assert done.status == (Status.CANCELLED if cancel_all else Status.COMPLETED), done
    assert (node.component_data[FULL].value is not None) == cancel_all
    assert node.component_data[FULL].host_lock_ref == 0
    assert_files(cache, node)


def test_short_ack_preserves_source_and_actual_prefix_and_failure_on_ack(owner):
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
        done.status == Status.FAILED and done.update.code == "incomplete_storage_write"
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


def test_request_hold_yields_partial_move_after_successful_store(owner):
    _, cache, service, dma = owner
    node = seed(cache)
    lock = cache.lock(node.id)
    try:
        result = transfer(service, dst="storage", kind="move_to")
        stage(service, dma)
        done = wait(service, result)
        assert done.status == Status.PARTIAL, done
        assert done.update.completed_steps == ("destination_ready",)
        assert node.component_data[FULL].value is not None
        assert_files(cache, node)
    finally:
        cache.unlock(lock)


def test_cancel_during_gpu_staging_does_not_start_cold_write(owner):
    _, cache, service, dma = owner
    node = seed(cache)
    result = transfer(service, dst="storage")
    service.progress()
    assert dma.pending
    assert service.control.cancel(result.ref).status == Status.CANCELLING
    dma.finish()
    done = wait(service, result)
    assert done.status == Status.CANCELLED
    assert (
        cache.cache_controller.storage_backend.batch_exists(
            cache.tree_core.get_hash_values(node.id)
        )
        == 0
    )
    assert (
        node.component_data[FULL].lock_ref
        == node.component_data[FULL].host_lock_ref
        == 0
    )


def test_write_geometry_survives_ordinary_radix_split(owner):
    _, cache, service, dma = owner
    seed(cache)
    cc = cache.cache_controller
    entered, release = threading.Event(), threading.Event()
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
            cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", (1, 2, 8)))))
            release.set()
            done = wait(service, result)
    finally:
        release.set()
    assert done.status == Status.COMPLETED, done
    assert [
        (o.coverage.start_token, o.coverage.end_token)
        for o in done.update.target_outcomes
    ] == [(0, 4)]
    path = service.control.executor._walk(
        (1, 2, 3, 4), service.control.executor._compatibility(cache_fixtures.target())
    )
    assert len(path) == 2
    for fragment, _, _, _ in path:
        assert_files(cache, fragment)
        cd = fragment.component_data[FULL]
        assert cd.value is None and cd.lock_ref == cd.host_lock_ref == 0


def test_invalid_and_excessive_scopes_refuse_before_io(owner):
    _, cache, service, dma = owner
    seed(cache)
    result = transfer(service, src="storage", dst="gpu")
    assert result.status == Status.UNSUPPORTED
    result = transfer(
        service, kind="move_to", src="host", dst="storage", message="host-move"
    )
    assert result.status == Status.UNSUPPORTED
    assert not dma.pending and not cache._native_storage_write_results


def test_receipts_do_not_certify_a_prefix_evicted_during_write(owner):
    _, cache, service, dma = owner
    node = seed(cache)
    cc = cache.cache_controller
    original = cc.page_set_func

    def evicted(*args, **kwargs):
        success = original(*args, **kwargs)
        # Real bytes were written, then a backend eviction removed them before
        # the whole-prefix completion boundary.
        for path in Path(cc.storage_backend.file_path).glob("*.bin"):
            path.unlink()
        return success

    with patch.object(cc, "page_set_func", evicted):
        result = transfer(service, dst="storage", kind="move_to")
        stage(service, dma)
        done = wait(service, result)
    assert done.status == Status.FAILED, done
    assert done.update.code == "incomplete_storage_write"
    assert not done.update.completed_steps
    assert node.component_data[FULL].value is not None
