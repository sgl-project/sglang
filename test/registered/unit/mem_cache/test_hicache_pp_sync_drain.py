"""Unit tests for HiCache PP synchronization."""

import socket
import tempfile
import threading
import time
import unittest
from array import array
from collections import deque
from datetime import timedelta
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.distributed
import torch.multiprocessing

from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.hicache_storage import PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.pool_host.group import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage_prefetch import StoragePrefetchRetries
from sglang.srt.mem_cache.unified_cache.cache_action import BackupKV
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedRadixCache,
    _OngoingPrefetch,
    _StoragePrefetchWait,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


class _FakeWork:
    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True


class _Holder:
    pass  # carries only the attributes the PP-sync window touches


def _holder(*, window: int) -> _Holder:
    holder = _Holder()
    holder.pp_sync_window = window
    holder.work_rounds = deque([[]])
    return holder


def _round(holder: _Holder, *, sends: int) -> list[_FakeWork]:
    # one scheduler round: bound the outstanding sends, then issue this round's
    UnifiedRadixCache._bound_async_work(holder)
    works = [_FakeWork() for _ in range(sends)]
    holder.work_rounds[-1].extend(works)
    return works


class TestPPSyncWindow(CustomTestCase):
    """Sends inside the window are never waited on, so a rank is not tied to the next
    rank's progress; sends older than the window are, so the list stays bounded."""

    def test_rounds_within_window_are_not_waited(self):
        holder = _holder(window=3)
        rounds = [_round(holder, sends=2) for _ in range(3)]
        self.assertFalse(any(w.waited for works in rounds for w in works))

        _round(holder, sends=1)
        self.assertTrue(all(w.waited for w in rounds[0]))
        self.assertFalse(any(w.waited for works in rounds[1:] for w in works))
        self.assertEqual(len(holder.work_rounds), 3)

    def test_window_one_is_the_single_round_drain(self):
        holder = _holder(window=1)
        first = _round(holder, sends=2)
        self.assertFalse(any(w.waited for w in first))

        _round(holder, sends=1)
        self.assertTrue(all(w.waited for w in first))
        self.assertEqual(len(holder.work_rounds), 1)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _gloo_pp_rank(rank: int, port: int, result: dict) -> None:
    torch.distributed.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=f"tcp://127.0.0.1:{port}"
    )
    cache = _holder(window=1)
    cache.pp_rank = rank
    cache.pp_size = 2
    cache.pp_group = torch.distributed.group.WORLD
    data = torch.zeros(2, dtype=torch.int64)
    elapsed = {}
    for phase, window, rounds in (("drain", 1, 2), ("window", 4, 4)):
        cache.pp_sync_window = window
        if rank == 1:
            time.sleep(1.0)
        start = time.perf_counter()
        for _ in range(rounds):
            UnifiedRadixCache._bound_async_work(cache)
            UnifiedRadixCache._pp_sync(cache, data)
        elapsed[phase] = time.perf_counter() - start
        for works in cache.work_rounds:
            for work in works:
                work.wait()
        cache.work_rounds = deque([[]])
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    if rank == 0:
        result.update(elapsed)


def _write_allocation_miss_rank(rank, store, failed_stage):
    dist = torch.distributed
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=store,
        timeout=timedelta(seconds=15),
    )
    try:
        cache = TestUnifiedPPSyncBatching()._make_cache(rank, [], [])
        del cache._all_reduce, cache.writing_check
        cache.attn_tp_group = cache.attn_cp_group = None
        cache.tp_world_size = 1
        cache.pp_group = dist.group.WORLD
        cache.metrics_collector = None
        cache._build_backup_sidecar = MagicMock(return_value=[])
        cache.dec_lock_ref = MagicMock()
        cache.tree_core.build_backup_kv_action = lambda node: BackupKV(node_ids=[node])
        cache.tree_core.collect_path_info = lambda nodes: [
            [SimpleNamespace(node_id=node, parent_id=0, path_start=0, path_end=1)]
            for node in nodes
        ]
        cache.tree_core.build_backup_spec = lambda node: (torch.tensor([node]), {})
        cache.tree_core.rotation_phase_of = lambda node: None
        published = {}
        cache.tree_core.commit_backup = lambda node, slots, _: published.update(
            {node: slots}
        )
        cache.evict_host = lambda needed: 0
        cache.inc_lock_ref = MagicMock()
        cache._track_write_through_node = MagicMock()
        completed = []
        cache._finish_write_through_ack = completed.append

        cc = cache.cache_controller
        # Exercise the real free list and double-free checks without a device mirror.
        pool = object.__new__(MLATokenToKVPoolHost)
        pool.size, pool.page_size, pool.device = 3, 1, "cpu"
        pool.layout, pool.can_use_write_back_jit = "layer_first", False
        pool.lock = threading.Lock()
        pool.clear()
        cc.mem_pool_host = HostPoolGroup(
            [PoolEntry("kv", pool, device_pool=None, layer_mapper=lambda i: i)]
        )
        pool.shared_allocation_domain = object() if rank == failed_stage else None
        assert not cache.enable_pp_write_admission(lambda: None)
        assert cache._pp_pending_writes is None
        pool.shared_allocation_domain = None
        assert cache.enable_pp_write_admission(lambda: None)
        cc.write_queue = []
        cc.write = HybridCacheController.write.__get__(cc)
        cc.allocate_host_transfers = (
            HybridCacheController.allocate_host_transfers.__get__(cc)
        )
        cc._uses_shared_host_domain = (
            HybridCacheController._uses_shared_host_domain.__get__(cc)
        )

        def flush():
            if cc.write_queue:
                ids = [node for op in cc.write_queue for node in op.node_ids]
                cc.ack_write_queue.append(
                    SimpleNamespace(node_ids=ids, finish_event=MagicMock())
                )
                cc.write_queue.clear()

        cc.start_writing = flush
        occupied = pool.alloc(2) if rank == failed_stage else None
        offset = rank * 100
        for token in (10, 20, 30):
            if token == 30 and occupied is not None:
                pool.free(occupied)
            cache._apply_cache_actions(
                [BackupKV(node_ids=[token + offset])], RadixKey(array("q", [token]))
            )
            for _ in range(cache.pp_sync_window + 2):
                cache.check_hicache_events()
        # Each rank has one slot left. Equally sized, differently named
        # prefixes must not consume it or displace the next shared prefix.
        for token in (40 + rank, 50):
            cache._apply_cache_actions(
                [BackupKV(node_ids=[token + offset])], RadixKey(array("q", [token]))
            )
            for _ in range(cache.pp_sync_window + 2):
                cache.check_hicache_events()
        assert pool.available_size() == 0
        assert [node - offset for node in completed] == [10, 30, 50]

        def reclaim(needed):
            assert needed == 1
            return pool.free(published.pop(10 + offset))

        cache.evict_host = reclaim
        cache._apply_cache_actions(
            [BackupKV(node_ids=[60 + offset])], RadixKey(array("q", [60]))
        )
        for _ in range(cache.pp_sync_window + 2):
            cache.check_hicache_events()
        # A miss on one stage must leave the following backup usable everywhere.
        assert [node - offset for node in completed] == [10, 30, 50, 60]
        assert not cache.cache_controller.ack_write_queue
        assert [node - offset for node in published] == [30, 50, 60]
        for slots in published.values():
            pool.free(slots)
        assert pool.available_size() == pool.size

        # A rank can acquire all its side pools while a peer's last pool is
        # full. Return both the anchor and newly allocated side slots, keeping
        # any storage supplied by the caller.
        side = object.__new__(MLATokenToKVPoolHost)
        side.size, side.page_size, side.device = 3, 1, "cpu"
        side.layout, side.can_use_write_back_jit = "layer_first", False
        side.lock = threading.Lock()
        side.clear()
        cc.mem_pool_host = HostPoolGroup(
            [
                PoolEntry("kv", pool, device_pool=None, layer_mapper=lambda i: i),
                PoolEntry("swa", side, device_pool=None, layer_mapper=lambda i: i),
            ]
        )
        borrowed = side.alloc(1)
        occupied = side.alloc(2) if rank == failed_stage else None
        cache._build_backup_sidecar = lambda device, _: [
            PoolTransfer(name="swa", device_indices=device, host_indices=borrowed),
            PoolTransfer(name="swa", device_indices=device),
        ]
        cache._apply_cache_actions(
            [BackupKV(node_ids=[70 + offset])], RadixKey(array("q", [70]))
        )
        for _ in range(cache.pp_sync_window + 2):
            cache.check_hicache_events()
        assert len(completed) == 4 and pool.available_size() == pool.size
        if occupied is not None:
            side.free(occupied)
        assert side.available_size() == side.size - 1
        side.free(borrowed)
        assert side.available_size() == side.size
        # Independent leaves in one flush share a bounded number of round
        # trips even though each has a separate host allocation.
        pool.size = 128
        pool.clear()
        cache._build_backup_sidecar = lambda *_: []
        votes = []
        reduce = cache._pp_write_min

        def count_vote(values):
            votes.append(len(values))
            return reduce(values)

        cache._pp_write_min = count_vote
        for token in range(1000, 1128):
            cache._apply_cache_actions(
                [BackupKV(node_ids=[token + offset])], RadixKey(array("q", [token]))
            )
        for _ in range(cache.pp_sync_window + 2):
            cache.check_hicache_events()
        assert [node - offset for node in completed[-128:]] == list(range(1000, 1128))
        assert len(votes) <= 4 and pool.available_size() == 0
        for node in completed[-128:]:
            pool.free(published.pop(node))
        assert pool.available_size() == pool.size
        votes.clear()
        before = len(completed)
        occupied = pool.alloc(1) if rank == failed_stage else None
        for token in range(2000, 2128):
            cache._apply_cache_actions(
                [BackupKV(node_ids=[token + offset])], RadixKey(array("q", [token]))
            )
        for _ in range(cache.pp_sync_window + 2):
            cache.check_hicache_events()
        assert [node - offset for node in completed[before:]] == list(range(2000, 2127))
        assert len(votes) <= 7
        for node in completed[before:]:
            pool.free(published.pop(node))
        if occupied is not None:
            pool.free(occupied)
        assert pool.available_size() == pool.size
        for work_round in cache.work_rounds:
            for work in work_round:
                work.wait()
    finally:
        dist.destroy_process_group()


class TestOrdinaryPPWriteThrough(CustomTestCase):
    def test_one_stage_cannot_reserve_host_slots(self):
        """A full host pool on either stage skips the same backup everywhere."""
        for failed_stage in (0, 1):
            with (
                self.subTest(failed_stage=failed_stage),
                tempfile.TemporaryDirectory() as tmp,
            ):
                torch.multiprocessing.spawn(
                    _write_allocation_miss_rank,
                    args=((Path(tmp) / "store").as_uri(), failed_stage),
                    nprocs=2,
                )


class TestPPSyncWindowGloo(CustomTestCase):
    """With real gloo sends, rank 0 must issue a window's worth of rounds while rank 1
    has posted no recv; a single-round window blocks rank 0 on rank 1 instead."""

    def test_leader_runs_ahead_by_window(self):
        ctx = torch.multiprocessing.get_context("spawn")
        with ctx.Manager() as manager:
            result = manager.dict()
            port = _free_port()
            procs = [
                ctx.Process(target=_gloo_pp_rank, args=(rank, port, result))
                for rank in range(2)
            ]
            for proc in procs:
                proc.start()
            for proc in procs:
                proc.join(timeout=60)
            self.assertTrue(all(proc.exitcode == 0 for proc in procs), procs)
            elapsed = dict(result)
        self.assertGreater(elapsed["drain"], 0.5, elapsed)
        self.assertLess(elapsed["window"], 0.5, elapsed)


class TestUnifiedPPSyncBatching(CustomTestCase):
    def _make_cache(self, pp_rank, write_ready, load_ready):
        cache = object.__new__(UnifiedRadixCache)
        cache.tree_core = SimpleNamespace(
            enable_storage=False,
            write_back_duplicate_reclaim_digest=0,
        )
        cache.pp_rank = pp_rank
        cache.pp_size = 2
        cache.pp_sync_window = 1
        cache.work_rounds = deque([[]])
        cache.host_memory_mode = "cache"
        # The per-tick deadline vote runs under storage; no live prefetch
        # means it issues no collective, which the asserts below pin.
        cache.prefetch_stop_policy = "timeout"
        cache.ongoing_prefetch = {}
        cache.enable_storage_metrics = False
        cache.storage_metrics_collector = None
        cache.buffer_pipeline = None
        cache.linker = None
        cache._all_reduce = MagicMock()
        cache.writing_check = MagicMock()
        cache.loading_check = MagicMock()
        cache.cache_controller = SimpleNamespace(
            start_writing=MagicMock(),
            # The tick's rank-local dead-worker check; no workers started here.
            dead_storage_workers=lambda: [],
            ack_write_queue=[
                SimpleNamespace(
                    finish_event=SimpleNamespace(query=MagicMock(return_value=ready))
                )
                for ready in write_ready
            ],
            ack_load_queue=[
                SimpleNamespace(
                    finish_event=SimpleNamespace(query=MagicMock(return_value=ready))
                )
                for ready in load_ready
            ],
        )
        return cache

    def _arm_live_prefetch(self, cache):
        """Cache-mode storage with one live prefetch on this PP stage."""
        cache.tree_core.enable_storage = True
        cache._drain_storage_control_queues_impl = MagicMock()
        cache.storage_prefetch_retries = StoragePrefetchRetries()
        cache._storage_prefetch_wait = {}
        cc = cache.cache_controller
        for name in (
            "prefetch_hit_queue",
            "ack_prefetch_queue",
            "ack_backup_queue",
            "host_mem_release_queue",
        ):
            setattr(cc, name, Queue())
        cc.extra_host_mem_release_queues = {}
        cc.terminate_prefetch = MagicMock()
        handle = CacheRequestHandle("r", 0)
        operation = SimpleNamespace(
            handle=handle,
            deadline_expired=False,
            host_indices=None,
            hash_value=[],
            completed_tokens=0,
        )
        cache.ongoing_prefetch[handle] = _OngoingPrefetch(
            0, None, None, operation, None, {}
        )
        return handle, operation

    def test_pp_relays_the_deadline_vote_after_the_counts(self):
        """PP0 evaluates the clocks and votes; a follower contributes zeros,
        receives the verdict through the same tag after the counts, and marks
        and terminates the operation without reading its own clock."""
        leader = self._make_cache(0, [True], [True])
        handle, operation = self._arm_live_prefetch(leader)
        wait = _StoragePrefetchWait(timeout_s=1.0)
        with patch("time.perf_counter", return_value=100.0):
            wait.timer.observe(ready=False)
        leader._storage_prefetch_wait[handle] = wait
        with patch("time.perf_counter", return_value=103.0):
            leader.check_hicache_events()

        counts, vote = leader._all_reduce.call_args_list
        self.assertIs(counts.args[1], torch.distributed.ReduceOp.MIN)
        self.assertEqual(vote.args[0].tolist(), [1])
        self.assertIs(vote.args[1], torch.distributed.ReduceOp.MAX)
        self.assertTrue(wait.expired)
        self.assertTrue(operation.deadline_expired)
        leader.cache_controller.terminate_prefetch.assert_called_once_with(operation)

        follower = self._make_cache(1, [True], [True])
        handle, operation = self._arm_live_prefetch(follower)
        follower._prefetch_deadline_expired = MagicMock()
        relayed = []

        def relay_from_pp0(data, _):
            relayed.append(data.tolist())
            if len(relayed) == 1:
                data[0], data[1] = 1, 1  # counts; digest slots stay local
            else:
                data.fill_(1)  # PP0's verdict

        follower._all_reduce.side_effect = relay_from_pp0
        follower.check_hicache_events()

        self.assertEqual(relayed[1], [0])
        self.assertEqual(follower._all_reduce.call_count, 2)
        follower._prefetch_deadline_expired.assert_not_called()
        self.assertTrue(follower._storage_prefetch_wait[handle].expired)
        self.assertTrue(operation.deadline_expired)
        follower.cache_controller.terminate_prefetch.assert_called_once_with(operation)
        follower.writing_check.assert_called_once_with(finish_count=1)

    def test_pp_batches_write_and_load_counts_once(self):
        leader = self._make_cache(0, [True, False], [True, True])
        leader.check_hicache_events()

        leader._all_reduce.assert_called_once()
        self.assertEqual(leader._all_reduce.call_args.args[0].tolist(), [1, 2, 0, 0])
        leader.writing_check.assert_called_once_with(finish_count=1)
        leader.loading_check.assert_called_once_with(finish_count=2)

        follower = self._make_cache(1, [True], [True])

        def reduce_to_min(counts, _):
            counts.copy_(torch.tensor([1, 1, 0, 0], dtype=torch.int64))

        follower._all_reduce.side_effect = reduce_to_min
        follower.check_hicache_events()

        for queue in (
            follower.cache_controller.ack_write_queue,
            follower.cache_controller.ack_load_queue,
        ):
            queue[0].finish_event.query.assert_not_called()
        follower._all_reduce.assert_called_once()
        follower.writing_check.assert_called_once_with(finish_count=1)
        follower.loading_check.assert_called_once_with(finish_count=1)

    def test_buffer_mode_follower_drains_rank_local_completions(self):
        cache = self._make_cache(1, [True, False], [True, True])
        cache.host_memory_mode = "buffer_only"
        cache.tree_core.enable_storage = True
        cache._all_reduce_attn_groups = MagicMock()
        cache._drain_storage_control_queues_impl = MagicMock()
        cc = cache.cache_controller
        for size, name in enumerate(
            (
                "prefetch_hit_queue",
                "ack_prefetch_queue",
                "ack_backup_queue",
                "host_mem_release_queue",
            ),
            start=1,
        ):
            queue = Queue()
            for _ in range(size):
                queue.put(object())
            setattr(cc, name, queue)

        cache.check_hicache_events()

        cache._all_reduce.assert_not_called()
        cache._all_reduce_attn_groups.assert_called_once()
        cache.writing_check.assert_called_once_with(finish_count=1)
        cache.loading_check.assert_called_once_with(finish_count=2)
        cache._drain_storage_control_queues_impl.assert_called_once_with(
            n_storage_hit=1,
            n_ack_prefetch=2,
            n_backup=3,
            n_release=4,
            extra_release_counts={},
            log_metrics=True,
        )


if __name__ == "__main__":
    unittest.main()
