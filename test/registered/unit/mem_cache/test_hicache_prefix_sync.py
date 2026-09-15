"""CPU/Gloo coverage for hybrid storage prefix agreement across ranks."""

import multiprocessing
import tempfile
import threading
import time
import unittest
from datetime import timedelta
from pathlib import Path
from queue import Queue
from unittest.mock import patch

import torch.distributed as dist

from sglang.srt.managers.cache_controller import (
    HiCacheController,
    PrefetchOperation,
    StorageOperation,
)
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    PrefetchOperation as HybridPrefetchOperation,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

_PAGE_SIZE = 4
_TOKENS = list(range(4 * _PAGE_SIZE))
_HASHES = [f"{page:064x}" for page in range(1, 5)]
_COLLECTIVE_TIMEOUT = timedelta(seconds=30)
_WORKER_TIMEOUT = 90


def _controller(controller_cls, groups):
    controller = controller_cls.__new__(controller_cls)
    controller.page_size = _PAGE_SIZE
    controller.prefetch_hits_sync_groups = groups
    return controller


def _check_file_prefetch(rank, groups, scenario):
    """Exercise real file queries and the full hit-queue handoff, without GPUs."""
    kv_pages = 4
    checkpoints = (4,)
    controller_cls = HybridCacheController
    terminated = False
    if scenario == "aligned":
        expected = 4
    elif scenario == "disjoint":
        checkpoints = (4,) if rank % 2 == 0 else (3,)
        expected = 0
    elif scenario == "earlier_common":
        checkpoints = (1, 4) if rank % 2 == 0 else (1, 3)
        expected = 1
    elif scenario == "kv_bound":
        kv_pages = 4 if rank % 2 == 0 else 3
        checkpoints = (1, 3, 4) if rank % 2 == 0 else (1, 2, 4)
        expected = 1
    elif scenario == "zero_hit":
        kv_pages = 0 if rank == 0 else 4
        expected = 0
    elif scenario == "terminated":
        terminated = rank == 0
        expected = 0
    elif scenario in ("mixed_controllers", "hybrid_kv_only"):
        if rank % 2 == 0:
            if scenario == "mixed_controllers":
                controller_cls = HiCacheController
            kv_pages = 3
        checkpoints = (1, 4)
        expected = 1
    else:
        raise AssertionError(f"Unknown scenario: {scenario}")

    hashes = _HASHES
    # Native hashing is Linux-only and independent of prefix agreement. Use
    # fixed page keys; file lookup, controller logic and collectives stay real.
    with (
        tempfile.TemporaryDirectory() as directory,
        patch("sglang.srt.mem_cache.utils.get_native_hash", return_value=hashes),
    ):
        backend = HiCacheFile.__new__(HiCacheFile)
        backend.file_path = directory
        backend.config_suffix = f"_rank{rank}"
        backend.metadata_cache = None
        for name, pages in (
            (PoolName.KV, range(1, kv_pages + 1)),
            (PoolName.MAMBA, checkpoints),
        ):
            for page in pages:
                Path(backend._get_component_path(hashes[page - 1], name)).touch()

        controller = _controller(controller_cls, groups)
        controller.storage_backend = backend
        if controller_cls is HybridCacheController:
            operation = HybridPrefetchOperation(
                "request",
                _TOKENS,
                pool_transfers=[
                    PoolTransfer(
                        name=PoolName.MAMBA,
                        keys=hashes[-1:],
                        hit_policy=PoolHitPolicy.TRAILING_PAGES,
                    )
                ],
            )
            if scenario == "hybrid_kv_only" and rank % 2 == 0:
                operation.pool_transfers = None
        else:
            operation = PrefetchOperation("request", _TOKENS)
        if terminated:
            operation.mark_terminate()

        controller.prefetch_queue = Queue()
        controller.prefetch_hit_queue = Queue()
        controller.prefetch_queue.put(operation)
        # The production loop drains pending work after stop is requested.
        controller.storage_stop_event = threading.Event()
        controller.storage_stop_event.set()
        controller.prefetch_thread_func()

        result = controller.prefetch_hit_queue.get_nowait()
        assert result is operation
        assert result.storage_hit_count == expected * _PAGE_SIZE, (
            rank,
            scenario,
            result.storage_hit_count,
        )
        assert result.hash_value == hashes[:expected], (rank, scenario)
        assert controller.prefetch_hit_queue.empty()
        if (
            isinstance(operation, HybridPrefetchOperation)
            and operation.pool_transfers
            and expected
        ):
            assert expected in operation.restorable_prefix_pages
            controller._sync_trailing_keys(
                operation.pool_transfers, operation.all_hash_values, expected
            )
            assert operation.pool_transfers[0].keys == hashes[expected - 1 : expected]
            assert Path(
                backend._get_component_path(hashes[expected - 1], PoolName.MAMBA)
            ).is_file()


def _prefix_sync_worker(rank, world_size, rendezvous):
    dist.init_process_group(
        backend="gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=_COLLECTIVE_TIMEOUT,
    )
    try:
        groups = [dist.group.WORLD]
        for scenario in (
            "aligned",
            "disjoint",
            "earlier_common",
            "kv_bound",
            "zero_hit",
            "terminated",
            "mixed_controllers",
            "hybrid_kv_only",
        ):
            _check_file_prefetch(rank, groups, scenario)

        # Backends without metadata retain the legacy dense-prefix assumption.
        # Explicit [] is different: it declares no restorable positive prefix.
        for candidates, hit_pages, expected in (
            (None, 3 if rank == 0 else 4, 3),
            (None if rank == 0 else [1, 4], 3 if rank == 0 else 4, 1),
            ([] if rank == 0 else None, 4, 0),
        ):
            operation = StorageOperation(None, _TOKENS)
            operation.restorable_prefix_pages = candidates
            controller = _controller(HiCacheController, groups)
            actual = controller._sync_storage_hit_count(
                operation, hit_pages * _PAGE_SIZE
            )
            assert actual == expected * _PAGE_SIZE, (rank, candidates, actual)

        if world_size == 4:
            # Orthogonal groups model the sequential CP/TP (or TP/PP) sync.
            # Each row has an earlier endpoint absent from the other row.
            groups = []
            created_groups = []
            for ranks in ([0, 1], [2, 3], [0, 2], [1, 3]):
                group = dist.new_group(ranks, timeout=_COLLECTIVE_TIMEOUT)
                if rank in ranks:
                    groups.append(group)
                    created_groups.append(group)
            try:
                operation = StorageOperation(None, _TOKENS)
                operation.restorable_prefix_pages = (
                    [1, 2, 4],
                    [1, 2, 3],
                    [1, 3, 4],
                    [1, 3],
                )[rank]
                controller = _controller(HiCacheController, groups)
                actual = controller._sync_storage_hit_count(
                    operation, max(operation.restorable_prefix_pages) * _PAGE_SIZE
                )
                assert actual == _PAGE_SIZE, (rank, actual)
            finally:
                for group in created_groups:
                    dist.destroy_process_group(group)
    finally:
        dist.destroy_process_group()


class TestHiCachePrefixSync(CustomTestCase):
    def test_single_rank_metadata_contract(self):
        controller = _controller(HiCacheController, [])
        for candidates, hit_pages, expected in (
            (None, 3, 3),
            ([], 3, 0),
            ([1, 3], 3, 3),
            ([1, 4], 3, 1),
            ([4], 3, 0),
            ([1, 4], 0, 0),
        ):
            with self.subTest(candidates=candidates, hit_pages=hit_pages):
                operation = StorageOperation(None, _TOKENS)
                operation.restorable_prefix_pages = candidates
                self.assertEqual(
                    controller._sync_storage_hit_count(
                        operation, hit_pages * _PAGE_SIZE
                    ),
                    expected * _PAGE_SIZE,
                )

    def test_trailing_windows_follow_selected_endpoint(self):
        controller = _controller(HybridCacheController, [])
        hashes = _HASHES
        for endpoint in (0, 1, 3):
            with self.subTest(endpoint=endpoint):
                transfers = [
                    PoolTransfer(
                        name=pool,
                        keys=hashes[-window:],
                        hit_policy=PoolHitPolicy.TRAILING_PAGES,
                    )
                    for pool, window in ((PoolName.MAMBA, 1), (PoolName.SWA, 2))
                ]
                controller._sync_trailing_keys(transfers, hashes, endpoint)
                self.assertEqual(
                    transfers[0].keys, hashes[max(0, endpoint - 1) : endpoint]
                )
                self.assertEqual(
                    transfers[1].keys, hashes[max(0, endpoint - 2) : endpoint]
                )

    def _run_distributed(self, world_size):
        if not dist.is_available() or not dist.is_gloo_available():
            self.skipTest("Gloo is required for CPU prefix synchronization")
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = Path(directory, "rendezvous").as_uri()
            workers = [
                context.Process(
                    target=_prefix_sync_worker,
                    args=(rank, world_size, rendezvous),
                )
                for rank in range(world_size)
            ]
            try:
                for worker in workers:
                    worker.start()
                deadline = time.monotonic() + _WORKER_TIMEOUT
                for rank, worker in enumerate(workers):
                    worker.join(max(0, deadline - time.monotonic()))
                    self.assertFalse(worker.is_alive(), f"Rank {rank} timed out")
                    self.assertEqual(worker.exitcode, 0, f"Rank {rank} failed")
            finally:
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                    if worker.pid is not None:
                        worker.join(timeout=5)

    def test_two_ranks(self):
        self._run_distributed(2)

    def test_four_ranks(self):
        self._run_distributed(4)


if __name__ == "__main__":
    unittest.main()
