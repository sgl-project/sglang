"""Unit tests for HiCache PP synchronization."""

import socket
import time
import unittest
from collections import deque
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.distributed
import torch.multiprocessing

from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.storage_events import StorageEventReducer
from sglang.srt.mem_cache.storage_prefetch import StoragePrefetchRetries
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


class _PublishingBackend:
    """A storage backend stand-in that reports its own residency."""

    def __init__(self, *, publishes=True, pending=0, shard=0, shard_count=1):
        self.publishes_kv_events = publishes
        self.kv_event_shard = shard
        self.kv_event_shard_count = shard_count
        self.pending = pending
        self.take_kv_events = MagicMock(return_value=[])

    def pending_kv_events(self):
        return self.pending


class TestStorageKvEventSync(CustomTestCase):
    """The residency column exists only for a backend that publishes KV events."""

    def _make_cache(self, backend, *, launched=True):
        cache = object.__new__(UnifiedRadixCache)
        # init_hicache builds the reducer only for a launch-time publisher.
        cache._storage_event_reducer = (
            StorageEventReducer(backend.kv_event_shard_count)
            if launched and getattr(backend, "publishes_kv_events", False)
            else None
        )
        cache.tree_core = SimpleNamespace(
            enable_storage=True,
            write_back_duplicate_reclaim_digest=7,
            kv_events=SimpleNamespace(enabled=False),
        )
        cache.pp_rank = 0
        cache.pp_size = 1
        cache.host_memory_mode = "cache"
        cache.prefetch_stop_policy = "best_effort"
        cache.ongoing_prefetch = {}
        cache._shm_count_exchange = MagicMock(return_value=None)
        cache._all_reduce = MagicMock()
        cache._drain_storage_kv_events = MagicMock()
        cache.cache_controller = SimpleNamespace(
            ack_write_queue=[],
            ack_load_queue=[],
            prefetch_hit_queue=Queue(),
            ack_prefetch_queue=Queue(),
            ack_backup_queue=Queue(),
            host_mem_release_queue=Queue(),
            extra_host_mem_release_queues={},
            storage_backend=backend,
        )
        return cache

    def test_non_publishing_backend_keeps_the_row_unchanged(self):
        # A publisher left attached at runtime (a failed rollback of a rejected
        # attach) has no launch-time reducer and must not widen the row.
        for backend, launched in (
            (None, True),
            (_PublishingBackend(publishes=False, pending=3), True),
            (_PublishingBackend(pending=3), False),
        ):
            with self.subTest(backend=backend, launched=launched):
                cache = self._make_cache(backend, launched=launched)
                self.assertEqual(
                    cache._sync_hicache_ready_counts(), (0, 0, (0, 0, 0, 0), ())
                )
                self.assertEqual(
                    cache._all_reduce.call_args.args[0].tolist(),
                    [0, 0, 0, 0, 0, 0, 7, -7],
                )
                cache._drain_storage_kv_events.assert_not_called()

    def test_publishing_backend_adds_a_column_and_drains_on_any_rank(self):
        for pending, peer_pending, drains in ((0, 0, False), (0, 2, True)):
            with self.subTest(pending=pending, peer_pending=peer_pending):
                cache = self._make_cache(_PublishingBackend(pending=pending))

                def reduce_with_peer(counts, _, peer=peer_pending):
                    counts[6] = min(int(counts[6]), 0 if peer else 1)

                cache._all_reduce.side_effect = reduce_with_peer
                self.assertEqual(
                    cache._sync_hicache_ready_counts(), (0, 0, (0, 0, 0, 0), ())
                )
                self.assertEqual(
                    cache._all_reduce.call_args.args[0].tolist(),
                    [0, 0, 0, 0, 0, 0, 0 if peer_pending else 1, 7, -7],
                )
                self.assertEqual(cache._drain_storage_kv_events.called, drains)

    def test_shared_memory_exchange_carries_the_column(self):
        for pending, peer_pending, drains in ((0, 0, False), (0, 2, True)):
            with self.subTest(pending=pending, peer_pending=peer_pending):
                cache = self._make_cache(_PublishingBackend(pending=pending))
                exchange = MagicMock(width=16)

                def allgather(values, peer=peer_pending):
                    # The shm exchange stores only values in [0, 2**40).
                    self.assertTrue(all(0 <= value < 1 << 40 for value in values))
                    return [values, [*values[:6], 0 if peer else 1, *values[7:]]]

                exchange.allgather.side_effect = allgather
                cache._shm_count_exchange.return_value = exchange
                self.assertEqual(
                    cache._sync_hicache_ready_counts(), (0, 0, (0, 0, 0, 0), ())
                )
                self.assertEqual(cache._drain_storage_kv_events.called, drains)
                cache._all_reduce.assert_not_called()

    def test_follower_joins_the_gather_without_retaining_events(self):
        # A follower that skipped the collective would hang the publisher.
        cache = object.__new__(UnifiedRadixCache)
        backend = _PublishingBackend(shard=1, shard_count=2)
        backend.take_kv_events.return_value = ["event"]
        cache.cache_controller = SimpleNamespace(storage_backend=backend)
        cache.tree_core = SimpleNamespace(kv_events=SimpleNamespace(enabled=False))
        group = object()
        cache.attn_cp_group = None
        cache.attn_tp_group = group
        cache.tp_world_size = 2
        cache._storage_event_reducer = MagicMock()
        sent = []

        def gather(out, batch, group):
            sent.append((batch, group))
            out[:] = [{0: []}, batch]

        with (
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(torch.distributed, "all_gather_object", side_effect=gather),
        ):
            UnifiedRadixCache._drain_storage_kv_events(cache)

        self.assertEqual(sent, [({1: ["event"]}, group)])
        cache._storage_event_reducer.update.assert_not_called()

    def test_publisher_reduces_shards_and_enqueues(self):
        from sglang.srt.disaggregation.kv_events import BlockStored, StorageMedium
        from sglang.srt.mem_cache.events import KVCacheEventRecorder

        stored = BlockStored(
            block_hashes=[5],
            parent_block_hash=None,
            token_ids=[1, 2],
            block_size=2,
            lora_id=None,
            medium=StorageMedium.CPU,
            ownership="owner",
        )
        cache = object.__new__(UnifiedRadixCache)
        backend = _PublishingBackend(shard=0, shard_count=2)
        backend.take_kv_events.return_value = [stored]
        cache.cache_controller = SimpleNamespace(storage_backend=backend)
        recorder = KVCacheEventRecorder(enabled=True, page_size=2)
        cache.tree_core = SimpleNamespace(kv_events=recorder)
        group = object()
        cache.attn_cp_group = None
        cache.attn_tp_group = group
        cache.tp_world_size = 3
        cache._storage_event_reducer = StorageEventReducer(2)
        # Shard 1 also holds the page; rank 2 replicates a shard out of range.
        peers = [{1: [stored]}, {2: [stored]}]

        def gather(out, batch, group):
            out[:] = [batch, *peers]

        with (
            patch.object(torch.distributed, "get_world_size", return_value=3),
            patch.object(torch.distributed, "all_gather_object", side_effect=gather),
        ):
            UnifiedRadixCache._drain_storage_kv_events(cache)

        self.assertEqual(recorder.take(), [stored])


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
