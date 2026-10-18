"""Four real Gloo ranks exercise DCP missing shards and worker completion."""

import json
import tempfile
import threading
import time
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_hicache_dcp_storage_controller import _controller

from sglang.srt.environ import envs
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
    PrefetchOperation,
)
from sglang.srt.mem_cache.pool_host.group import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

CASES = (
    "healthy",
    "missing",
    "evicted_after_lookup",
    "short_read_after_lookup",
    "backup_delayed",
    "best_effort",
    "timeout",
    "best_effort_partial",
    "timeout_partial",
    "timeout_not_expired",
)


def _drain(cache, prefetch=0, backup=0):
    cache._drain_storage_control_queues_impl(
        n_storage_hit=0,
        n_ack_prefetch=prefetch,
        n_backup=backup,
        n_release=None,
        extra_release_counts=None,
        log_metrics=False,
    )


def _failure_controller(rank):
    # Reuse the real CPU pool/config fixture, then exercise the concrete
    # controller used by the unified cache, including its backup worker.
    base = _controller(rank)
    cc = HybridCacheController.__new__(HybridCacheController)
    cc.__dict__.update(base.__dict__)
    cc.mem_pool_host = HostPoolGroup(
        [
            PoolEntry(
                name=PoolName.KV,
                host_pool=base.mem_pool_host,
                device_pool=base.mem_pool_device,
                layer_mapper=lambda layer: layer,
                is_primary_index_anchor=True,
            )
        ]
    )
    cc.storage_backend._evictor._eviction_enabled = False
    cc.storage_backend._evictor._eviction_configured = False
    cc.page_get_func = cc._generic_page_get
    cc.tp_group = cc.attn_tp_group = dist.group.WORLD
    cc.attn_cp_group = cc.pp_group = None
    # The factory normally needs SGLang's initialized GPU world. Only
    # replace that factory; group selection and reductions stay real.
    with mock.patch(
        "sglang.srt.distributed.parallel_state.create_custom_parallel_group",
        side_effect=lambda group_ranks, backend: dist.new_group(
            ranks=group_ranks,
            backend=backend,
            timeout=timedelta(seconds=30),
        ),
    ):
        cc.prefetch_hits_sync_groups = cc._create_sync_groups()
        cc.prefetch_completion_sync_groups = cc._create_sync_groups()
    assert all(
        dist.get_process_group_ranks(g) == [0, 1, 2, 3]
        for g in cc.prefetch_hits_sync_groups + cc.prefetch_completion_sync_groups
    )
    cc.enable_storage = True
    return cc


def _failure_cache(cc):
    pool = cc.storage_host_pool
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.cache_controller = cc
    cache.host_memory_mode = "cache"
    cache.ongoing_backup = {}
    cache.ongoing_prefetch = {}
    cache.dec_host_lock_ref = lambda node, slots: pool.free(slots)
    # A successful prefix would belong to the tree. Release it here
    # after completion; the real ACK drain releases the incomplete tail.
    cache._handle_prefetch_result = lambda op: pool.free(
        op.host_indices[: op.completed_tokens]
    )
    return cache


def _delayed_backup(*, rank, cc, cache, tokens):
    pool = cc.storage_host_pool
    entered, release = threading.Event(), threading.Event()
    original_set = cc.page_set_func

    def controlled_write(*args):
        if rank == 1:
            entered.set()
            assert release.wait(10)
        return original_set(*args)

    cc.page_set_func = controlled_write
    slots = pool.alloc(128)
    ident = cc.write_storage(slots, tokens[:128], ["new-page"])
    cache.ongoing_backup[ident] = (0, slots)
    if rank == 1:
        assert entered.wait(10)
        assert cc.ack_backup_queue.empty()
        assert pool.slot_used[slots].all()
        release.set()
    ack = cc.ack_backup_queue.get(timeout=20)
    assert ack.completed_tokens == (0 if rank >= 2 else 128)
    cc.ack_backup_queue.put(ack)
    _drain(cache, backup=1)
    assert int(pool.slot_used.sum()) == 0
    assert cc.backup_thread.is_alive()


def _run_failure_case(*, rank, directory, case):
    reports = []
    storage_dir = Path(directory) / case
    storage_dir.mkdir(exist_ok=True)
    with envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.override(str(storage_dir)):
        cc = _failure_controller(rank)
        pool = cc.storage_host_pool
        cache = _failure_cache(cc)
        tokens = list(range(384))
        hashes = get_storage_hash_str(tokens, None, page_size=128)
        pool.kv_buffer.fill_(rank % 2 + 1)
        source = pool.alloc(384)
        if rank < 2:
            assert cc._generic_page_set(hashes, source)
        pool.free(source)
        dist.barrier()
        bad_path = storage_dir / (
            cc.storage_backend._get_suffixed_key(hashes[1]) + ".bin"
        )
        if rank == 1 and case == "missing":
            bad_path.unlink()
        dist.barrier()
        partial = case.endswith("_partial")
        entered, release = threading.Event(), threading.Event()
        original_get = cc.page_get_func

        def controlled_read(operation, keys, *args):
            if partial and rank == 1 and keys == [hashes[1]]:
                entered.set()
                assert release.wait(10)
            return original_get(operation, keys, *args)

        cc.page_get_func = controlled_read
        # One page per batch checks every ACK when a shard disappears
        # after lookup, including batches after the first missing page.
        with mock.patch("sglang.srt.managers.cache_controller.STORAGE_BATCH_SIZE", 1):
            HiCacheController._start_storage_threads(cc)
            try:
                op = PrefetchOperation(CacheRequestHandle(case, 0), tokens)
                cc.prefetch_queue.put(op)
                queried = cc.prefetch_hit_queue.get(timeout=20)
                assert queried is op
                expected_lookup = 128 if case == "missing" else 384
                assert op.storage_hit_count == expected_lookup
                if case == "evicted_after_lookup" and rank == 1:
                    bad_path.unlink()
                if case == "short_read_after_lookup" and rank == 1:
                    bad_path.write_bytes(b"truncated")
                dist.barrier()
                op.host_indices = pool.alloc(op.storage_hit_count)
                cache.ongoing_prefetch[op.handle] = SimpleNamespace(operation=op)
                cache.prefetch_stop_policy = (
                    "best_effort"
                    if case.startswith("best_effort")
                    else "timeout"
                    if case.startswith("timeout")
                    else "wait_complete"
                )
                cache.pp_rank = 0
                # Only one rank's deadline expires; all ranks must stop together.
                cache.prefetch_timeout_base = (
                    0 if rank == 1 and case != "timeout_not_expired" else 3600
                )
                cache.prefetch_timeout_per_page = 0
                cache._all_reduce = lambda tensor, reduce_op: dist.all_reduce(
                    tensor, op=reduce_op
                )
                if case in ("best_effort", "timeout"):
                    assert cache._can_terminate_prefetch(op)
                    cc.terminate_prefetch(op)
                elif not partial:
                    assert not cache._can_terminate_prefetch(op)
                cc.prefetch_buffer.put(op)
                acks = []
                if partial:
                    first = cc.ack_prefetch_queue.get(timeout=20)
                    assert first.completed_tokens == 128
                    acks.append(first)
                    if rank == 1:
                        assert entered.wait(10)
                    assert cache._can_terminate_prefetch(op)
                    cc.terminate_prefetch(op)
                    release.set()
                while True:
                    ack = cc.ack_prefetch_queue.get(timeout=20)
                    acks.append(ack)
                    if ack.completed_req:
                        break
                for ack in acks:
                    cc.ack_prefetch_queue.put(ack)
                _drain(cache, prefetch=len(acks))
                expected = (
                    128
                    if partial
                    or case
                    in ("missing", "evicted_after_lookup", "short_read_after_lookup")
                    else 384
                )
                if case in ("best_effort", "timeout"):
                    expected = 0
                assert op.completed_tokens == expected, (
                    case,
                    rank,
                    op.completed_tokens,
                )
                assert len(acks) == expected_lookup // 128 + 1
                assert int(pool.slot_used.sum()) == 0
                assert cc.prefetch_io_aux_thread.is_alive()

                # A delayed shard writer must retain its host allocation
                # until its write finishes, while replicas can acknowledge.
                if case == "backup_delayed":
                    _delayed_backup(rank=rank, cc=cc, cache=cache, tokens=tokens)
                reports.append(
                    dict(
                        case=case,
                        rank=rank,
                        tokens=op.completed_tokens,
                        acknowledgments=len(acks),
                        remaining_slots=int(pool.slot_used.sum()),
                    )
                )
            finally:
                release.set()
                HiCacheController._stop_storage_threads(cc)
                cc._destroy_sync_groups(cc.prefetch_hits_sync_groups)
                cc._destroy_sync_groups(cc.prefetch_completion_sync_groups)
        dist.barrier()
    return reports[0]


def _worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/rendezvous",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    reports = []
    try:
        for case in CASES:
            reports.append(_run_failure_case(rank=rank, directory=directory, case=case))
        Path(directory, f"rank-{rank}.json").write_text(json.dumps(reports))
    finally:
        dist.destroy_process_group()


class TestDcpStorageFailures(CustomTestCase):
    def test_four_rank_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            context = mp.spawn(_worker, args=(directory,), nprocs=4, join=False)
            deadline = time.monotonic() + 180
            try:
                while not context.join(timeout=1):
                    self.assertLess(time.monotonic(), deadline, "DCP workers hung")
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)
            reports = [
                json.loads(Path(directory, f"rank-{rank}.json").read_text())
                for rank in range(4)
            ]
            self.assertTrue(all(len(rank) == len(CASES) for rank in reports))
            for case_index in range(len(CASES)):
                rows = [rank[case_index] for rank in reports]
                self.assertEqual(len({row["tokens"] for row in rows}), 1)
                self.assertTrue(all(row["remaining_slots"] == 0 for row in rows))


if __name__ == "__main__":
    unittest.main()
