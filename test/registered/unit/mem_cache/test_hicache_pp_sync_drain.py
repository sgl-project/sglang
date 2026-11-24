"""Unit tests for HiCache PP synchronization."""

import socket
import threading
import time
import unittest
from collections import deque
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.distributed
import torch.multiprocessing

from sglang.srt.disaggregation.pp_consensus_store import PPConsensusStore
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.storage_prefetch import StoragePrefetchRetries
from sglang.srt.mem_cache.unified_cache.storage_attachment import StorageAttachment
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


class _StoreGroup:
    """broadcast_object stand-in: PP0's store publishes its address, PP1 reads it."""

    def __init__(self):
        self._obj = None

    def broadcast_object(self, obj, src=0):
        if obj is not None:
            self._obj = obj
        return self._obj


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

    def _storage_stage(self, pp_rank, group, backups):
        cache = self._make_cache(pp_rank, [], [])
        cache.tree_core.enable_storage = True
        cache.tree_core.page_size = 1
        cache._backup_requested_tokens = {}
        cache._storage_prefetch_hit_remaining_by_reqid = {}
        cache.ongoing_backup = {}
        cache._all_reduce_storage_drain = MagicMock()
        cc = cache.cache_controller
        for name in (
            "prefetch_hit_queue",
            "ack_prefetch_queue",
            "ack_backup_queue",
            "host_mem_release_queue",
        ):
            setattr(cc, name, Queue())
        cc.extra_host_mem_release_queues = {}
        cc.backup_skip = False
        for _ in range(backups):
            self._ack_backup(cache)
        cache._pp_store_expected = True
        if group is not None:
            cache._pp_ready_store = PPConsensusStore(2, pp_rank, group)
            self.addCleanup(cache._pp_ready_store.close)
        return cache

    def _ack_backup(self, cache):
        cc = cache.cache_controller
        cc.ack_backup_queue.put(
            SimpleNamespace(id=cc.ack_backup_queue.qsize(), completed_tokens=0)
        )

    def _leader_round(self, leader):
        leader.check_hicache_events()
        counts = leader._all_reduce.call_args.args[0].clone()
        return counts, counts.tolist()[4]  # [write, load, hit, prefetch, backup, ...]

    def _follower_round(self, follower, leader, counts):
        follower._all_reduce.side_effect = lambda data, _: data.copy_(counts)
        done = threading.Event()

        def run():
            follower.check_hicache_events()
            done.set()

        threading.Thread(target=run, daemon=True).start()
        self.assertTrue(
            done.wait(timeout=5), "PP1 blocked draining acks it has not received"
        )
        if follower._pp_ready_store is None:
            return
        # Let PP1's latest totals (one per storage queue) reach PP0.
        keys = [follower._storage_ready_key(index) for index in range(4)]
        self._wait_until(
            lambda: all(
                leader._pp_ready_store.collect(key)[1]
                == follower._pp_ready_store.get(key)
                for key in keys
            )
        )

    def _wait_until(self, condition):
        deadline = time.monotonic() + 5
        while not condition():
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.01)

    def test_cache_mode_pp_drains_only_what_every_stage_received(self):
        """In cache mode PP1 drains the storage-ack count PP0 relays, blocking
        until it has that many. PP0 must relay only what every stage has
        received, even when a stage's last report predates its last drain."""
        group = _StoreGroup()
        leader = self._storage_stage(0, group, backups=2)
        follower = self._storage_stage(1, group, backups=1)
        relayed = []
        for ack_after_round in (False, False, True, False, False):
            counts, n_backup = self._leader_round(leader)
            relayed.append(n_backup)
            self._follower_round(follower, leader, counts)
            if ack_after_round:
                self._ack_backup(follower)

        # Round 3: PP1 last reported 1 ack and has drained it, so PP0 relays 0;
        # its second ack is reported in round 4 and drained in round 5.
        self.assertEqual(relayed, [0, 1, 0, 0, 1])
        for cache in (leader, follower):
            self.assertEqual(cache.cache_controller.ack_backup_queue.qsize(), 0)

    def test_detach_drain_starts_a_new_epoch(self):
        """The detach drain consumes each stage's queues rank-locally, so the
        totals PP1 reported before it are no longer a lower bound. PP0, running
        ahead, must ignore them, relay 0 until PP1 reports again, then resume."""
        group = _StoreGroup()
        leader = self._storage_stage(0, group, backups=1)
        follower = self._storage_stage(1, group, backups=2)
        for _ in range(2):
            counts, _ = self._leader_round(leader)
            self._follower_round(follower, leader, counts)
        old_key = follower._storage_ready_key(2)
        self.assertEqual(leader._pp_ready_store.collect(old_key)[1], 2)

        # PP0 drains and decides its next round before PP1 reaches its drain,
        # while PP1's pre-detach total of 2 is still in PP0's view.
        leader.drain_storage_control_queues_local()
        self._ack_backup(leader)
        counts, n_backup = self._leader_round(leader)
        relayed = [n_backup]
        follower.drain_storage_control_queues_local()
        self._follower_round(follower, leader, counts)
        self._wait_until(lambda: leader._pp_ready_store.collect(old_key)[1] is None)
        self.assertEqual(follower._storage_epoch, leader._storage_epoch)

        self._ack_backup(follower)
        for _ in range(2):
            counts, n_backup = self._leader_round(leader)
            relayed.append(n_backup)
            self._follower_round(follower, leader, counts)
        # PP1's new ack is reported in round 2 and drained in round 3.
        self.assertEqual(relayed, [0, 0, 1])
        for cache in (leader, follower):
            self.assertEqual(cache.cache_controller.ack_backup_queue.qsize(), 0)

    def test_store_opens_in_the_tick_and_outlives_detach(self):
        """The store opens in the first tick with storage on, with PP0's address
        relayed down the PP-sync channel (no collective, which would wait for a
        stage that has not seen the attach). A detach keeps it and starts a new
        epoch; after the next attach draining resumes without a second relay."""
        leader = self._storage_stage(0, None, backups=1)
        follower = self._storage_stage(1, None, backups=1)
        for cache in (leader, follower):  # cleanups run last-in first, PP1 first
            self.addCleanup(
                lambda c=cache: c._pp_ready_store and c._pp_ready_store.close()
            )
        relayed = []
        leader._pp_sync = MagicMock(
            side_effect=lambda data: relayed.append(data.clone())
        )
        follower._pp_sync = MagicMock(
            side_effect=lambda data: data.copy_(relayed.pop(0))
        )
        for cache in (leader, follower):
            cache.tree_core.enable_storage = False
            cache.check_hicache_events()
            self.assertIsNone(cache._pp_ready_store)
            cache.tree_core.enable_storage = True

        def rounds(n):
            out = []
            for _ in range(n):
                counts, n_backup = self._leader_round(leader)
                out.append(n_backup)
                self._follower_round(follower, leader, counts)
            return out

        self.assertEqual(rounds(2), [0, 1])
        self.assertEqual(relayed, [])  # PP1 consumed PP0's one address
        stores = [leader._pp_ready_store, follower._pp_ready_store]
        old_key = follower._storage_ready_key(2)
        for cache in (leader, follower):
            cache.cache_controller.detach_storage_backend = MagicMock()
            self.assertTrue(StorageAttachment(cache).detach()[0])
            cache.tree_core.enable_storage = True  # what the next attach turns on
            self._ack_backup(cache)
        self._wait_until(lambda: leader._pp_ready_store.collect(old_key)[1] is None)

        # The new epoch hears PP1 from its first round, so round 2 drains.
        self.assertEqual(rounds(3), [0, 1, 0])
        self.assertEqual([leader._pp_ready_store, follower._pp_ready_store], stores)
        self.assertEqual(leader._pp_sync.call_count, 1)
        self.assertEqual(follower._storage_epoch, leader._storage_epoch)
        for cache in (leader, follower):
            self.assertEqual(cache.cache_controller.ack_backup_queue.qsize(), 0)

    def test_store_open_failure_retries_next_tick(self):
        """If PP0 cannot open its store, PP1 hears "no store" on the same channel,
        so the PP-sync stream stays aligned; nothing is drained, PP0 warns once,
        escalates once to an error, and a later tick opens the store and drains."""
        leader = self._storage_stage(0, None, backups=1)
        follower = self._storage_stage(1, None, backups=1)
        for cache in (leader, follower):  # cleanups run last-in first, PP1 first
            self.addCleanup(
                lambda c=cache: c._pp_ready_store and c._pp_ready_store.close()
            )
        relayed = []
        leader._pp_sync = MagicMock(
            side_effect=lambda data: relayed.append(data.clone())
        )
        follower._pp_sync = MagicMock(
            side_effect=lambda data: data.copy_(relayed.pop(0))
        )
        failures_left = [2]

        def flaky_store(pp_size, pp_rank, group):
            if pp_rank == 0 and failures_left[0]:
                failures_left[0] -= 1
                raise OSError("no routable address")
            return PPConsensusStore(pp_size, pp_rank, group)

        module = "sglang.srt.mem_cache.unified_radix_cache"
        relayed_counts = []
        with (
            patch(f"{module}.PPConsensusStore", flaky_store),
            patch(f"{module}._PP_STORE_OPEN_ERROR_TICKS", 2),
            self.assertLogs(module, level="WARNING") as logs,
        ):
            for _ in range(4):
                counts, n_backup = self._leader_round(leader)
                relayed_counts.append(n_backup)
                self._follower_round(follower, leader, counts)
                self.assertEqual(relayed, [])  # PP1 heard every address message

        # Ticks 1-2 fail (no store anywhere); tick 3 opens both, tick 4 drains.
        self.assertEqual(relayed_counts, [0, 0, 0, 1])
        self.assertEqual([r.levelname for r in logs.records], ["WARNING", "ERROR"])
        for cache in (leader, follower):
            self.assertIsNotNone(cache._pp_ready_store)
            self.assertEqual(cache.cache_controller.ack_backup_queue.qsize(), 0)

    def test_pp0_warns_once_when_a_stage_never_reports(self):
        """A stage whose epoch is out of step never reports; PP0 relays 0 and
        must say so once instead of stalling silently."""
        leader = self._storage_stage(0, _StoreGroup(), backups=1)
        with (
            patch(
                "sglang.srt.mem_cache.unified_radix_cache._STORAGE_UNREPORTED_WARN_ROUNDS",
                2,
            ),
            self.assertLogs(
                "sglang.srt.mem_cache.unified_radix_cache", level="WARNING"
            ) as logs,
        ):
            relayed = [self._leader_round(leader)[1] for _ in range(3)]
        self.assertEqual(relayed, [0, 0, 0])
        self.assertEqual(len(logs.records), 1)
        self.assertIn("PP ranks [1]", logs.output[0])


if __name__ == "__main__":
    unittest.main()
