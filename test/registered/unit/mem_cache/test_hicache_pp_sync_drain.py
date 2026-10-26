"""Unit tests for HiCache PP synchronization."""

import socket
import time
import unittest
from collections import deque
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
import torch.distributed
import torch.multiprocessing

from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
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
        cache.enable_storage_metrics = False
        cache.storage_metrics_collector = None
        cache.buffer_pipeline = None
        cache.linker = None
        cache._all_reduce = MagicMock()
        cache.writing_check = MagicMock()
        cache.loading_check = MagicMock()
        cache.cache_controller = SimpleNamespace(
            start_writing=MagicMock(),
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
