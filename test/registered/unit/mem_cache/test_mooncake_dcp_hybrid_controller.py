"""Four-rank MLA/KDA checkpoint contracts using real host pools.

The fixture assembles real host pools below the public capability guard. Workers,
collectives, abort handling and the all-or-nothing acceptance check are real.
CPU CI substitutes a byte-copying native client; the manual test uses TCP. The
final radix publication/lock boundary is supplied by the fixture.
"""

import json
import tempfile
import threading
import traceback
import unittest
from datetime import timedelta
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_mooncake_dcp_storage import _mamba_pool, _own_mamba, _page_segments
from test_mooncake_dcp_storage_controller import (
    _backup_done,
    _cache,
    _FaultClient,
    _make_controller,
    run_workers,
)

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.buffer_mode.storage_existence_cache import (
    StorageExistenceCache,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
    PrefetchOperation,
)
from sglang.srt.mem_cache.pool_host.group import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_radix_cache import _OngoingPrefetch
from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=35, suite="base-a-test-cpu")

CASES = (
    "kv_only_rank",
    "sparse",
    "disjoint",
    "complete",
    "missing_state",
    "missing_component",
    "query_cancelled",
    "query_failure",
    "evicted_state",
    "cancel_inflight",
)


def _state_sets(case):
    if case in ("sparse", "kv_only_rank"):
        return [{1, 4}, {1, 3}, {1, 2, 4}, {1, 3, 4}]
    if case == "disjoint":
        return [{2, 4}, {1, 3}, {2, 4}, {1, 3}]
    if case == "missing_state":
        return [{1, 4}, set(), {1, 4}, {1, 4}]
    return [{1, 4} for _ in range(4)]


def _drain(cache, count=0):
    cache._drain_storage_control_queues_impl(
        n_storage_hit=0,
        n_ack_prefetch=count,
        n_backup=0,
        n_release=None,
        extra_release_counts={PoolName.MAMBA: None},
        log_metrics=False,
    )


def _state_tag(rank, component, boundary):
    return 17 * rank + 5 * component + boundary


class _HybridCase:
    """One distributed checkpoint scenario with owned host-pool allocations."""

    def __init__(self, *, rank, objects, case, address):
        self.rank, self.case, self.address = rank, case, address
        self.objects = objects
        self.cc = _make_controller(
            self.rank, self.objects, "hybrid-" + self.case, self.address
        )
        self.kv, self.state = self.cc.storage_host_pool, _mamba_pool()
        self.cc.mem_pool_host = HostPoolGroup(
            [
                *self.cc.mem_pool_host.entries,
                PoolEntry(
                    PoolName.MAMBA,
                    self.state,
                    self.state.device_pool,
                    lambda layer: layer,
                ),
            ]
        )
        self.cc.extra_host_mem_release_queues = {PoolName.MAMBA: Queue()}
        self.cc.storage_backend.register_mem_host_pool_v2(self.state, PoolName.MAMBA)
        self.cache = _cache(self.cc)
        self.cache.tree_core = SimpleNamespace(page_size=128)
        self.cache.storage_existence_cache = StorageExistenceCache()
        self.tokens = list(range(512))
        self.hashes = get_storage_hash_str(self.tokens, None, page_size=128)
        self.client = _FaultClient(
            self.cc.storage_backend.store, self.rank, self.hashes
        )
        self.cc.storage_backend.store = self.client
        self.legal = _state_sets(self.case)
        self.boundaries = sorted(self.legal[self.rank])
        self.kv_source = self.kv.alloc(512)
        self.kv.kv_buffer.fill_(self.rank % 2 + 1)
        self.kv_oracle = self.kv.kv_buffer.clone()
        # Recurrent slots are independent of token pages and DCP divisibility.
        self.state_guard = self.state.alloc(3)
        self.state_source = self.state.alloc(len(self.boundaries))
        for component, buffer in enumerate(self.state.get_hybrid_pool_buffer()):
            for slot, boundary in zip(self.state_source.tolist(), self.boundaries):
                buffer[slot].view(torch.uint8).fill_(
                    _state_tag(self.rank, component, boundary)
                )
        self.client.own(self.kv, self.kv_source)
        if self.address is None:
            _own_mamba(self.client.client, self.state, self.state_source.tolist())
        self.outgoing = (
            [
                PoolTransfer(
                    PoolName.MAMBA,
                    host_indices=self.state_source,
                    keys=[self.hashes[p - 1] for p in self.boundaries],
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            ]
            if self.boundaries
            else None
        )
        # Retain seed allocations so restore cannot accidentally reuse them.
        self.cache.dec_host_lock_ref = lambda node, indices: None

    def _backup(self):
        ident = self.cc.write_storage(
            self.kv_source, self.tokens, self.hashes, extra_pools=self.outgoing
        )
        self.cache.ongoing_backup[ident] = (0, self.kv_source)
        _backup_done(
            self.cc, self.cache, 512 if self.rank < 2 or self.boundaries else 0
        )
        assert sum(key.endswith("_k") for key in self.client.put_keys) == (
            4 if self.rank < 2 else 0
        )
        assert sum(not key.endswith("_k") for key in self.client.put_keys) == 3 * len(
            self.boundaries
        )
        dist.barrier()

    def _remove_state_component(self):
        keys, _ = self.cc.storage_backend._get_hybrid_page_component_keys(
            [self.hashes[3]], self.incoming
        )
        self.client.remove(self.cc.storage_backend._tag_keys(keys)[1])

    def _query(self):
        self.incoming = PoolTransfer(
            PoolName.MAMBA,
            keys=["__placeholder__"],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        if self.case == "missing_component" and self.rank == 1:
            self._remove_state_component()
        dist.barrier()
        candidates = [set(pages) for pages in self.legal]
        if self.case == "missing_component":
            candidates[1].remove(4)
        self.expected = max(set.intersection(*candidates), default=0)
        if self.case in ("query_cancelled", "query_failure"):
            self.expected = 0
        self.op = PrefetchOperation(
            CacheRequestHandle(self.case, 0),
            self.tokens,
            pool_transfers=[self.incoming],
        )
        if self.case == "kv_only_rank" and self.rank == 3:
            # A rank needing no state still joins the checkpoint collective.
            self.op.pool_transfers = None
        if self.case == "query_cancelled" and self.rank == 1:
            self.op.mark_terminate()
        if self.case == "query_failure":
            self.client.mode = "lookup_exception"
        self.cc.prefetch_queue.put(self.op)
        assert self.cc.prefetch_hit_queue.get(timeout=10) is self.op
        assert self.op.storage_hit_count == self.expected * 128, (
            self.case,
            self.rank,
            self.op.storage_hit_count,
            self.expected * 128,
        )
        assert self.op.hash_value == self.hashes[: self.expected]
        self.published = []

    def _prepare_restore(self):
        if self.case == "evicted_state" and self.rank == 1:
            self._remove_state_component()
        dist.barrier()
        self.op.host_indices = self.kv.alloc(self.expected * 128)
        self.incoming.host_indices = self.state.alloc(1)
        self.client.own(self.kv, self.op.host_indices)
        if self.address is None:
            _own_mamba(
                self.client.client, self.state, self.incoming.host_indices.tolist()
            )
        for buffer in self.state.get_hybrid_pool_buffer():
            buffer.view(torch.uint8).fill_(165)
        self.state_before = [
            b.view(torch.uint8).clone() for b in self.state.get_hybrid_pool_buffer()
        ]
        self.cache.ongoing_prefetch[self.op.handle] = _OngoingPrefetch(
            0,
            RadixKey(self.tokens),
            self.op.host_indices,
            self.op,
            None,
            {PoolName.MAMBA: [self.incoming]},
        )
        self.cc.prefetch_tokens_occupied = len(self.tokens)

    def _publish(self, operation):
        if not self.cache._check_hybrid_prefetch_result(
            self.op.handle,
            operation,
            operation.completed_tokens,
            operation.hash_value,
            operation.host_indices,
            0,
            None,
            RadixKey(self.tokens),
        ):
            return
        assert self.incoming.keys == [self.hashes[self.expected - 1]]
        for page, dst in enumerate((self.op.host_indices[::128] // 128).tolist()):
            for actual, reference in zip(
                _page_segments(self.kv, self.kv.kv_buffer, dst),
                _page_segments(self.kv, self.kv_oracle, page),
            ):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        for component, (buffer, oracle) in enumerate(
            zip(self.state.get_hybrid_pool_buffer(), self.state_before)
        ):
            oracle[self.incoming.host_indices] = _state_tag(
                self.rank, component, self.expected
            )
            torch.testing.assert_close(buffer.view(torch.uint8), oracle, rtol=0, atol=0)
        self.published.append(operation.completed_tokens)
        self.cc.append_host_mem_release(operation.host_indices, [self.incoming])
        del self.cache.ongoing_prefetch[operation.handle]

    def _restore(self):
        self.cache._handle_prefetch_result = self._publish
        if self.case == "cancel_inflight":
            self.client.mode = self.case
        self.cc.prefetch_buffer.put(self.op)
        if self.case == "cancel_inflight":
            if self.rank == 1:
                assert self.client.entered.wait(10)
            dist.barrier()
            before_available = self.state.available_size()
            self.cache.release_aborted_request(self.op.handle)
            _drain(self.cache)
            assert self.kv.slot_used[self.op.host_indices].all()
            assert self.state.available_size() == before_available, (
                "state released before IO completion"
            )
            dist.barrier()
            self.client.release.set()
        acks = []
        while True:
            ack = self.cc.ack_prefetch_queue.get(timeout=10)
            acks.append(ack)
            if ack.completed_req:
                break
        assert len(acks) == self.expected + 2, (
            "KV progress, state result and final ACK must all arrive"
        )
        for ack in acks:
            self.cc.ack_prefetch_queue.put(ack)
        _drain(self.cache, len(acks))
        assert self.published == (
            []
            if self.case in ("evicted_state", "cancel_inflight")
            else [self.expected * 128]
        )
        assert not self.cache.ongoing_prefetch

    def _finish(self):
        self.kv.free(self.kv_source)
        self.state.free(self.state_source)
        self.state.free(self.state_guard)
        assert int(self.kv.slot_used.sum()) == 0
        assert self.state.available_size() == self.state.size
        free = torch.cat([self.state.free_slots, *self.state.release_slots])
        assert free.unique().numel() == self.state.size, "duplicate state release"
        assert (
            self.cc.prefetch_thread.is_alive()
            and self.cc.prefetch_io_aux_thread.is_alive()
        )
        return dict(
            case=self.case,
            rank=self.rank,
            tokens=self.published[0] if self.published else 0,
            lookup=self.expected * 128,
            remaining_slots=0,
        )

    def _close(self):
        self.client.release.set()
        HiCacheController._stop_storage_threads(self.cc)
        self.cc._destroy_sync_groups(self.cc.prefetch_hits_sync_groups)
        self.cc._destroy_sync_groups(self.cc.prefetch_completion_sync_groups)
        if self.address is not None:
            self.client.client.close()

    def run(self):
        with mock.patch("sglang.srt.managers.cache_controller.STORAGE_BATCH_SIZE", 1):
            HiCacheController._start_storage_threads(self.cc)
            try:
                self._backup()
                self._query()
                if self.expected and self.case != "kv_only_rank":
                    self._prepare_restore()
                    self._restore()
                return self._finish()
            finally:
                self._close()


def _run_case(rank, directory, objects, case, address):
    return _HybridCase(rank=rank, objects=objects, case=case, address=address).run()


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
            reports.append(_run_case(rank, directory, objects, case, address))
            dist.barrier()
        Path(directory, f"rank-{rank}.json").write_text(json.dumps(reports))
    except Exception:
        print(f"FAILED hybrid case={case} rank={rank}", flush=True)
        traceback.print_exc()
        raise
    finally:
        dist.destroy_process_group()


class TestMooncakeDcpHybridController(CustomTestCase):
    def test_trailing_maximum_does_not_prove_earlier_checkpoints(self):
        """Legacy results and assume-stored hints prove only the trailing endpoint."""
        for policy, assume_stored, expected in (
            (PoolHitPolicy.TRAILING_PAGES, False, 0),
            (PoolHitPolicy.ALL_PAGES, False, 8),
            (PoolHitPolicy.TRAILING_PAGES, True, 0),
        ):
            with self.subTest(policy=policy, assume_stored=assume_stored):
                cc = HybridCacheController.__new__(HybridCacheController)
                cc.page_size = 4
                cc.prefetch_queue, cc.prefetch_hit_queue = Queue(), Queue()
                cc.storage_stop_event = threading.Event()
                cc.prefetch_hits_sync_groups = []
                cc.storage_backend = mock.Mock()
                cc.storage_backend.batch_exists_v2.return_value = PoolTransferResult(
                    4, {}
                )
                if assume_stored:
                    cc.storage_backend.batch_exists_v2.side_effect = AssertionError(
                        "hint should skip lookup"
                    )

                def peer_reduction(tensor, reduce_op, groups):
                    # External collective boundary: the peer can restore only
                    # page 2. The production query/worker chooses the payload.
                    if tensor.ndim == 0:
                        tensor.fill_(min(int(tensor), 8))
                    else:
                        peer = torch.zeros_like(tensor)
                        peer[2] = 1
                        tensor.mul_(peer)
                    cc.storage_stop_event.set()

                cc._all_reduce = peer_reduction
                transfer = PoolTransfer(
                    PoolName.MAMBA, keys=["__placeholder__"], hit_policy=policy
                )
                operation = PrefetchOperation(
                    CacheRequestHandle("hint", 0),
                    list(range(16)),
                    pool_transfers=[transfer],
                    assume_stored=assume_stored,
                )
                cc.prefetch_queue.put(operation)
                cc.prefetch_thread_func()
                self.assertIs(cc.prefetch_hit_queue.get_nowait(), operation)
                self.assertEqual(operation.storage_hit_count, expected)

    def test_sparse_checkpoints_and_state_lifetime_on_four_ranks(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            mp.get_context("spawn").Manager() as manager,
        ):
            reports = run_workers(
                directory, manager.dict(), cases=CASES, worker=_worker
            )
            self.assertTrue(all(len(rows) == len(CASES) for rows in reports))


if __name__ == "__main__":
    unittest.main()
