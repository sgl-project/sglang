"""Mooncake faults must preserve rank agreement and host-buffer ownership.

Four CPU/Gloo ranks run the real query, backup, IO and completion workers.
Only the native client and the radix publication/lock boundary are replaced.
The same worker can use native TCP Mooncake from the manual integration test.
"""

import json
import tempfile
import threading
import time
import traceback
import unittest
from datetime import timedelta
from pathlib import Path
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_hicache_dcp_storage_controller import _controller
from test_hicache_dcp_storage_failures import _drain
from test_mooncake_dcp_storage import _config, _page_segments, _store

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
    PrefetchOperation,
)
from sglang.srt.mem_cache.pool_host.group import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.srt.mem_cache.storage_prefetch import StoragePrefetchRetries
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedRadixCache,
    _OngoingPrefetch,
)
from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=35, suite="base-a-test-cpu")

CASES = (
    "healthy",
    "missing",
    "evicted_after_lookup",
    "short_read",
    "lookup_exception",
    "lookup_short_batch",
    "get_exception",
    "put_exception",
    "put_short_batch",
    "cancel_inflight",
    "backup_delayed",
)


class _FaultClient:
    """Faults occur at the synchronous native-call boundary, after registration."""

    def __init__(self, client, rank, hashes):
        self.client = client
        self.rank = rank
        self.hashes = hashes
        self.mode = None
        self.put_keys = []
        self.entered = threading.Event()
        self.release = threading.Event()

    def __getattr__(self, name):
        method = getattr(self.client, name)
        if not name.startswith(("batch_get_into", "batch_put_from", "batch_is_exist")):
            return method

        def call(keys, *args):
            kind = "lookup" if name == "batch_is_exist" else name.split("_")[1]
            faulty_page = any(self.hashes[1] in key for key in keys)
            if self.rank == 1:
                if self.mode == f"{kind}_exception" and faulty_page:
                    raise RuntimeError(f"injected Mooncake {kind} failure")
                if self.mode == f"{kind}_short_batch" and faulty_page:
                    return []
                if (self.mode, kind) in (
                    ("cancel_inflight", "get"),
                    ("backup_delayed", "put"),
                ):
                    self.entered.set()
                    assert self.release.wait(15), "test did not release native IO"
            result = method(keys, *args)
            if kind == "put":
                self.put_keys.extend(keys)
            if (
                self.rank == 1
                and self.mode == "short_read"
                and kind == "get"
                and faulty_page
            ):
                return [n - 1 for n in result]
            return result

        return call

    def own(self, pool, indices):
        if hasattr(self.client, "own"):
            self.client.own(
                pool,
                (indices[:: pool.logical_page_size] // pool.logical_page_size).tolist(),
            )

    def remove(self, key):
        if hasattr(self.client, "objects"):
            del self.client.objects[key]
        else:
            # The isolated test service owns these objects. Bypass its lookup
            # lease to deterministically simulate loss after a successful query.
            assert self.client.remove(key, force=True) == 0


def _make_controller(rank, objects, tag, address):
    def factory(config, pool):
        config.extra_config = dict(_config().extra_config, extra_backend_tag=tag)
        if address is None:
            return _store(config, pool, objects)
        config.extra_config.update(
            master_server_address=address,
            metadata_server="P2PHANDSHAKE",
            local_hostname="127.0.0.1",
            global_segment_size=0,
        )
        store = MooncakeStore(config)
        store.register_mem_pool_host(pool)
        store.registered_pools[PoolName.KV] = pool
        return store

    base = _controller(rank, storage_factory=factory)
    cc = HybridCacheController.__new__(HybridCacheController)
    cc.__dict__.update(base.__dict__)
    cc.mem_pool_host = HostPoolGroup(
        [
            PoolEntry(
                name=PoolName.KV,
                host_pool=cc.storage_host_pool,
                device_pool=cc.mem_pool_device,
                layer_mapper=lambda layer: layer,
                is_primary_index_anchor=True,
            )
        ]
    )
    cc.page_get_func = cc._page_get_zero_copy
    cc.page_set_func = cc._page_set_zero_copy
    cc.tp_group = cc.attn_tp_group = dist.group.WORLD
    cc.attn_cp_group = cc.pp_group = None
    with mock.patch(
        "sglang.srt.distributed.parallel_state.create_custom_parallel_group",
        side_effect=lambda group_ranks, backend: dist.new_group(
            ranks=group_ranks, backend=backend, timeout=timedelta(seconds=30)
        ),
    ):
        cc.prefetch_hits_sync_groups = cc._create_sync_groups()
        cc.prefetch_completion_sync_groups = cc._create_sync_groups()
    cc.enable_storage = True
    cc.prefetch_tokens_occupied = 0
    return cc


def _cache(cc):
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.cache_controller = cc
    cache.host_memory_mode = "cache"
    cache.buffer_pipeline = cache.linker = None
    cache.ongoing_prefetch = {}
    cache.ongoing_backup = {}
    cache.storage_prefetch_retries = StoragePrefetchRetries()
    cache.prefetch_loaded_tokens_by_reqid = {}
    cache.prefetch_loaded_storage_start_by_reqid = {}
    cache._storage_prefetch_hit_remaining_by_reqid = {}
    cache.dec_host_lock_ref = lambda node, slots: cc.storage_host_pool.free(slots)
    return cache


def _backup(cc, cache, indices, tokens, hashes):
    ident = cc.write_storage(indices, tokens, hashes)
    cache.ongoing_backup[ident] = (0, indices)
    return ident


def _copy_source(pool, source, indices):
    for src, dst in enumerate((indices[::128] // 128).tolist()):
        for reference, target in zip(
            _page_segments(pool, source, src),
            _page_segments(pool, pool.kv_buffer, dst),
        ):
            target.copy_(reference)


def _backup_done(cc, cache, expected):
    ack = cc.ack_backup_queue.get(timeout=10)
    assert ack.completed_tokens == expected, (ack.completed_tokens, expected)
    assert cc.storage_host_pool.slot_used[ack.host_indices].all()
    cc.ack_backup_queue.put(ack)
    _drain(cache, backup=1)
    assert cc.backup_thread.is_alive()


def _restore(cc, cache, client, case, tokens, source, expected_lookup, expected_tokens):
    pool = cc.storage_host_pool
    op = PrefetchOperation(CacheRequestHandle(case, 0), tokens)
    cc.prefetch_queue.put(op)
    assert cc.prefetch_hit_queue.get(timeout=10) is op
    assert op.storage_hit_count == expected_lookup
    if case == "evicted_after_lookup" and client.rank == 1:
        client.remove(
            f"{cc.storage_backend.config_prefix}_{client.hashes[1]}_{cc.storage_backend.mla_suffix}_k"
        )
    dist.barrier()
    # Keep the source addresses occupied; restored pages must be unrelated.
    blocker = pool.alloc(384)
    op.host_indices = (
        pool.alloc(expected_lookup)
        if expected_lookup
        else torch.empty(0, dtype=torch.int64)
    )
    client.own(pool, op.host_indices)
    pool.kv_buffer.fill_(165)
    before = pool.kv_buffer.clone()
    cache.ongoing_prefetch[op.handle] = _OngoingPrefetch(
        0, RadixKey(tokens), op.host_indices, op, None, {}
    )
    cc.prefetch_tokens_occupied = len(tokens)
    published = []

    def publish(operation):
        assert operation is op
        published.append(operation.completed_tokens)
        # Compare only the agreed prefix: bytes beyond it may have been written
        # by faster ranks but must never be advertised as a usable cache hit.
        for page in range(operation.completed_tokens // 128):
            dst = int(operation.host_indices[page * 128]) // 128
            for actual, reference in zip(
                _page_segments(pool, pool.kv_buffer, dst),
                _page_segments(pool, source, page),
            ):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        pool.free(operation.host_indices[: operation.completed_tokens])
        del cache.ongoing_prefetch[operation.handle]

    cache._handle_prefetch_result = publish
    cc.prefetch_buffer.put(op)
    replacement = None
    if case == "cancel_inflight":
        if client.rank == 1:
            assert client.entered.wait(10)
        dist.barrier()
        cache.release_aborted_request(op.handle)
        _drain(cache)
        assert pool.slot_used[op.host_indices].all(), (
            "cancel freed an in-flight destination"
        )
        # Reuse the textual request ID with a new generation before old ACKs.
        replacement = PrefetchOperation(CacheRequestHandle(case, 1), tokens)
        replacement.host_indices = pool.alloc(128)
        cache.ongoing_prefetch[replacement.handle] = _OngoingPrefetch(
            0, RadixKey(tokens), replacement.host_indices, replacement, None, {}
        )
        assert not torch.isin(replacement.host_indices, op.host_indices).any()
        dist.barrier()
        client.release.set()

    acks = []
    while True:
        ack = cc.ack_prefetch_queue.get(timeout=10)
        acks.append(ack)
        if ack.completed_req:
            break
    assert len(acks) == expected_lookup // 128 + 1
    for ack in acks:
        cc.ack_prefetch_queue.put(ack)
    _drain(cache, prefetch=len(acks))
    if replacement is None:
        assert op.completed_tokens == expected_tokens
        assert published == [expected_tokens]
    else:
        assert not published, "canceled IO was published"
        assert replacement.completed_tokens == 0
        assert cache.ongoing_prefetch[replacement.handle].operation is replacement
        assert pool.slot_used[replacement.host_indices].all()
        pool.free(replacement.host_indices)
        del cache.ongoing_prefetch[replacement.handle]
    # No read can touch allocations outside its assigned destination pages.
    untouched = before.clone()
    for dst in (op.host_indices[::128] // 128).tolist():
        for actual, expected in zip(
            _page_segments(pool, pool.kv_buffer, dst),
            _page_segments(pool, untouched, dst),
        ):
            expected.copy_(actual)
    torch.testing.assert_close(pool.kv_buffer, untouched, rtol=0, atol=0)
    pool.free(blocker)
    assert int(pool.slot_used.sum()) == 0
    assert cc.prefetch_thread.is_alive() and cc.prefetch_io_aux_thread.is_alive()
    return dict(
        case=case,
        rank=client.rank,
        lookup=expected_lookup,
        tokens=op.completed_tokens,
        acknowledgments=len(acks),
        remaining_slots=0,
    )


def _exercise_backup_failure(*, cc, cache, client, case, source, hashes, rank):
    pool = cc.storage_host_pool
    backup_tokens = list(range(384, 768))
    backup_hashes = get_storage_hash_str(backup_tokens, None, page_size=128)
    client.hashes = backup_hashes
    indices = pool.alloc(384)
    _copy_source(pool, source, indices)
    client.own(pool, indices)
    _backup(cc, cache, indices, backup_tokens, backup_hashes)
    if case == "backup_delayed" and rank == 1:
        assert client.entered.wait(10)
        assert cc.ack_backup_queue.empty()
        assert pool.slot_used[indices].all()
        client.release.set()
    expected = 0 if rank >= 2 else 128 if rank == 1 and case.startswith("put_") else 384
    _backup_done(cc, cache, expected)
    client.mode = None
    # A later backup on the same worker must still finish.
    indices = pool.alloc(384)
    _copy_source(pool, source, indices)
    client.own(pool, indices)
    _backup(cc, cache, indices, backup_tokens, backup_hashes)
    _backup_done(cc, cache, 384 if rank < 2 else 0)
    client.hashes = hashes
    dist.barrier()


def _worker(rank, directory, objects, cases=CASES, address=None):
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
        for case in cases:
            cc = _make_controller(rank, objects, case, address)
            cache = _cache(cc)
            pool = cc.storage_host_pool
            tokens = list(range(384))
            hashes = get_storage_hash_str(tokens, None, page_size=128)
            client = _FaultClient(cc.storage_backend.store, rank, hashes)
            cc.storage_backend.store = client
            with mock.patch(
                "sglang.srt.managers.cache_controller.STORAGE_BATCH_SIZE", 1
            ):
                HiCacheController._start_storage_threads(cc)
                try:
                    # Shard-specific multibyte tags are independent of the
                    # adapter's address calculation; TP replicas share bytes.
                    words = pool.kv_buffer.view(torch.int32)
                    words.copy_(
                        (
                            torch.arange(words.numel(), dtype=torch.int32)
                            + (rank % 2 + 1) * 100003
                        ).reshape(words.shape)
                    )
                    source = pool.kv_buffer.clone()
                    indices = pool.alloc(384)
                    client.own(pool, indices)
                    _backup(cc, cache, indices, tokens, hashes)
                    _backup_done(cc, cache, 384 if rank < 2 else 0)
                    dist.barrier()
                    client.mode = case
                    if case.startswith("put_") or case == "backup_delayed":
                        _exercise_backup_failure(
                            cc=cc,
                            cache=cache,
                            client=client,
                            case=case,
                            source=source,
                            hashes=hashes,
                            rank=rank,
                        )
                    if case == "missing" and rank == 1:
                        client.remove(
                            f"{cc.storage_backend.config_prefix}_{hashes[1]}_{cc.storage_backend.mla_suffix}_k"
                        )
                    dist.barrier()
                    lookup = (
                        0
                        if case.startswith("lookup_")
                        else 128
                        if case == "missing"
                        else 384
                    )
                    completed = (
                        0
                        if case == "cancel_inflight"
                        else (
                            128
                            if case
                            in ("evicted_after_lookup", "short_read", "get_exception")
                            else lookup
                        )
                    )
                    reports.append(
                        _restore(
                            cc, cache, client, case, tokens, source, lookup, completed
                        )
                    )
                    if case.startswith("lookup_") or case == "get_exception":
                        client.mode = None
                        reports.append(
                            _restore(
                                cc,
                                cache,
                                client,
                                case + "_recovery",
                                tokens,
                                source,
                                384,
                                384,
                            )
                        )
                finally:
                    client.release.set()
                    HiCacheController._stop_storage_threads(cc)
                    cc._destroy_sync_groups(cc.prefetch_hits_sync_groups)
                    cc._destroy_sync_groups(cc.prefetch_completion_sync_groups)
                    if address is not None:
                        client.client.close()
            dist.barrier()
        Path(directory, f"rank-{rank}.json").write_text(json.dumps(reports))
    except Exception:
        print(f"FAILED Mooncake controller case={case} rank={rank}", flush=True)
        traceback.print_exc()
        raise
    finally:
        dist.destroy_process_group()


def run_workers(directory, objects, cases=CASES, address=None, worker=_worker):
    context = mp.spawn(
        worker, args=(directory, objects, cases, address), nprocs=4, join=False
    )
    deadline = time.monotonic() + 180
    try:
        while not context.join(timeout=1):
            assert time.monotonic() < deadline, "Mooncake controller ranks hung"
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    reports = [
        json.loads(Path(directory, f"rank-{rank}.json").read_text())
        for rank in range(4)
    ]
    for rows in zip(*reports):
        assert len({row["tokens"] for row in rows}) == 1, rows
        assert all(row["remaining_slots"] == 0 for row in rows)
    return reports


class TestMooncakeDcpStorageController(CustomTestCase):
    def test_faults_and_cancellation_on_four_gloo_ranks(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            mp.get_context("spawn").Manager() as manager,
        ):
            reports = run_workers(directory, manager.dict())
            self.assertTrue(all(len(rows) == len(CASES) + 3 for rows in reports))


if __name__ == "__main__":
    unittest.main()
