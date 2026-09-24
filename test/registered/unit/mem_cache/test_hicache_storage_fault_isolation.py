"""CPU tests: HiCache storage threads survive a failing L3 backend.

A fake storage backend raises, reports short writes, or succeeds on demand.
The tests check that the prefetch/backup threads stay alive and always ack,
that hit queries still join the cross-rank all-reduce, and that the per
direction circuit breaker opens, rejects, and recovers.
"""

import threading
import unittest
from datetime import timedelta
from pathlib import Path
from queue import Empty, Queue
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import torch
import torch.distributed
import torch.multiprocessing

from sglang.srt.managers.cache_controller import (
    HiCacheController,
    PrefetchOperation,
    StorageOperation,
)
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache import hybrid_cache_controller as hybrid
from sglang.srt.mem_cache.storage_circuit_breaker import StorageCircuitBreaker
from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

PAGE_SIZE = 4
TIMEOUT_S = 10


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class _FakeStorage:
    """Zero-copy (v1) backend whose reads/writes can raise or fail."""

    def __init__(self):
        self.read_error = None  # exception raised by exists/get
        self.write_error = None  # exception raised by set
        self.short_writes = False  # set returns False (e.g. EACCES/ENOSPC)
        self.stored = set()
        self.num_exists = self.num_gets = self.num_sets = 0

    def batch_exists(self, keys, extra_info=None):
        self.num_exists += 1
        if self.read_error is not None:
            raise self.read_error
        hits = 0
        for key in keys:
            if key not in self.stored:
                break
            hits += 1
        return hits

    def batch_get_v1(self, keys, host_indices, extra_info=None):
        self.num_gets += 1
        if self.read_error is not None:
            raise self.read_error
        return [key in self.stored for key in keys]

    def batch_set_v1(self, keys, host_indices, extra_info=None):
        self.num_sets += 1
        if self.write_error is not None:
            raise self.write_error
        if self.short_writes:
            return [False] * len(keys)
        self.stored.update(keys)
        return [True] * len(keys)


def _make_controller(cls=HiCacheController, threshold=3, cooldown_s=10.0):
    c = cls.__new__(cls)
    c.page_size = PAGE_SIZE
    c.backup_skip = False
    c.storage_backend = storage = _FakeStorage()
    c.page_get_func = c._page_get_zero_copy
    c.page_set_func = c._page_set_zero_copy
    c.storage_stop_event = threading.Event()
    for name in (
        "prefetch_queue",
        "backup_queue",
        "prefetch_buffer",
        "prefetch_sync_queue",
        "prefetch_hit_queue",
        "ack_prefetch_queue",
        "ack_backup_queue",
        "host_mem_release_queue",
    ):
        setattr(c, name, Queue())
    c.prefetch_hits_sync_groups = []
    c.prefetch_completion_sync_groups = []
    c._init_storage_fault_isolation()
    c.clock = clock = _Clock()
    c.storage_breakers = {
        d: StorageCircuitBreaker(d, threshold, cooldown_s, clock=clock)
        for d in ("read", "write")
    }
    return c, storage


def _backup_op(num_pages=2, tag="w", cls=StorageOperation):
    hashes = [f"{tag}{i}" for i in range(num_pages)]
    return cls(
        torch.arange(num_pages * PAGE_SIZE),
        list(range(num_pages * PAGE_SIZE)),
        hash_value=hashes,
    )


def _prefetch_op(num_pages=2):
    return PrefetchOperation("req", list(range(num_pages * PAGE_SIZE)))


class _Thread:
    """Run a storage loop in a thread and stop it through its stop event."""

    def __init__(self, controller, target, wake_queue):
        self.controller = controller
        self.wake_queue = wake_queue
        self.thread = threading.Thread(target=target, daemon=True)

    def __enter__(self):
        self.controller.storage_stop_event.clear()
        self.thread.start()
        return self.thread

    def __exit__(self, *exc):
        self.controller.storage_stop_event.set()
        self.wake_queue.put(None)
        self.thread.join(TIMEOUT_S)


def _get(queue):
    try:
        return queue.get(timeout=TIMEOUT_S)
    except Empty:
        raise AssertionError("storage thread did not produce the expected ack")


class TestStorageCircuitBreaker(CustomTestCase):
    def test_opens_after_threshold_and_recovers_after_cooldown(self):
        clock = _Clock()
        breaker = StorageCircuitBreaker("write", 3, 10.0, clock=clock)
        for _ in range(2):
            breaker.record_failure("backup", OSError("EIO"))
        self.assertTrue(breaker.allow())
        breaker.record_failure("backup", OSError("EIO"))
        self.assertFalse(breaker.allow())
        self.assertEqual(breaker.num_opens, 1)

        clock.now = 9.9
        self.assertFalse(breaker.allow())
        clock.now = 10.0
        self.assertTrue(breaker.allow())  # half-open probe
        breaker.record_success()
        self.assertFalse(breaker.is_open())
        # Closed again: a single failure no longer opens it.
        breaker.record_failure("backup", OSError("EIO"))
        self.assertTrue(breaker.allow())

    def test_half_open_failure_reopens_immediately(self):
        clock = _Clock()
        breaker = StorageCircuitBreaker("read", 2, 5.0, clock=clock)
        breaker.record_failure("hit query", OSError())
        breaker.record_failure("hit query", OSError())
        clock.now = 5.0
        self.assertTrue(breaker.allow())
        breaker.record_failure("hit query", OSError())
        self.assertFalse(breaker.allow())
        self.assertEqual(breaker.num_opens, 2)

    def test_success_resets_consecutive_count(self):
        breaker = StorageCircuitBreaker("read", 2, 5.0, clock=_Clock())
        for _ in range(5):
            breaker.record_failure("hit query", OSError())
            breaker.record_success()
        self.assertTrue(breaker.allow())
        self.assertEqual(breaker.num_opens, 0)

    def test_zero_threshold_disables(self):
        breaker = StorageCircuitBreaker("write", 0, 5.0, clock=_Clock())
        for _ in range(100):
            breaker.record_failure("backup", OSError())
        self.assertTrue(breaker.allow())


class TestBackupThreadFaultIsolation(CustomTestCase):
    def test_backup_exception_keeps_thread_alive_and_acks(self):
        c, storage = _make_controller(threshold=0)
        storage.write_error = OSError("Stale file handle")
        ops = [_backup_op(tag=f"w{i}") for i in range(3)]
        with _Thread(c, c.backup_thread_func, c.backup_queue) as thread:
            for op in ops:
                c.backup_queue.put(op)
            acked = [_get(c.ack_backup_queue) for _ in ops]
            self.assertTrue(thread.is_alive())
            # Storage heals: the same thread writes again.
            storage.write_error = None
            c.backup_queue.put(ok := _backup_op(tag="ok"))
            self.assertIs(_get(c.ack_backup_queue), ok)
        self.assertEqual([op.id for op in acked], [op.id for op in ops])
        self.assertEqual([op.completed_tokens for op in acked], [0, 0, 0])
        self.assertEqual(ok.completed_tokens, 2 * PAGE_SIZE)
        self.assertEqual(storage.num_sets, 4)

    def test_write_breaker_counts_exceptions_and_short_writes(self):
        c, storage = _make_controller(threshold=3, cooldown_s=10.0)
        storage.short_writes = True
        with _Thread(c, c.backup_thread_func, c.backup_queue):
            storage.write_error = OSError("EIO")
            c.backup_queue.put(_backup_op(tag="a"))
            _get(c.ack_backup_queue)
            storage.write_error = None  # short writes from here on
            for i in range(4):
                c.backup_queue.put(_backup_op(tag=f"b{i}"))
            for _ in range(4):
                _get(c.ack_backup_queue)
        # 1 exception + 2 short writes open the breaker; the rest are skipped.
        self.assertEqual(storage.num_sets, 3)
        self.assertEqual(c.storage_breakers["write"].num_opens, 1)
        self.assertFalse(c.storage_breakers["read"].is_open())

    def test_write_breaker_closes_after_cooldown(self):
        c, storage = _make_controller(threshold=1, cooldown_s=10.0)
        storage.write_error = OSError("ENOSPC")
        with _Thread(c, c.backup_thread_func, c.backup_queue):
            c.backup_queue.put(_backup_op(tag="a"))
            _get(c.ack_backup_queue)
            storage.write_error = None
            c.backup_queue.put(skipped := _backup_op(tag="b"))
            _get(c.ack_backup_queue)
            c.clock.now = 10.0
            c.backup_queue.put(written := _backup_op(tag="c"))
            _get(c.ack_backup_queue)
        self.assertEqual(skipped.completed_tokens, 0)
        self.assertEqual(written.completed_tokens, 2 * PAGE_SIZE)
        self.assertFalse(c.storage_breakers["write"].is_open())

    def test_backlog_limit_skips_but_acks(self):
        c, storage = _make_controller(threshold=0)
        c.storage_backup_backlog_limit = 2
        ops = [_backup_op(tag=f"w{i}") for i in range(5)]
        for op in ops:
            c.backup_queue.put(op)
        with _Thread(c, c.backup_thread_func, c.backup_queue):
            acked = [_get(c.ack_backup_queue) for _ in ops]
        # Popping ops 0-2 leaves >= 2 queued behind them, so they are skipped.
        self.assertEqual(
            [op.completed_tokens for op in acked], [0, 0, 0] + [2 * PAGE_SIZE] * 2
        )
        self.assertEqual(c.num_storage_backups_shed, 3)
        self.assertEqual(storage.num_sets, 2)

    def test_default_backlog_is_unbounded(self):
        c, storage = _make_controller(threshold=0)
        self.assertEqual(c.storage_backup_backlog_limit, 0)
        ops = [_backup_op(tag=f"w{i}") for i in range(8)]
        for op in ops:
            c.backup_queue.put(op)
        with _Thread(c, c.backup_thread_func, c.backup_queue):
            for _ in ops:
                _get(c.ack_backup_queue)
        self.assertEqual(storage.num_sets, 8)

    def test_hybrid_backup_exception_keeps_thread_alive(self):
        c, storage = _make_controller(hybrid.HybridCacheController, threshold=2)
        c.storage_backend_type = "nixl"
        storage.write_error = OSError("Stale file handle")
        ops = [_backup_op(tag=f"w{i}", cls=hybrid.StorageOperation) for i in range(3)]
        with _Thread(c, c.backup_thread_func, c.backup_queue) as thread:
            for op in ops:
                c.backup_queue.put(op)
            for _ in ops:
                _get(c.ack_backup_queue)
            self.assertTrue(thread.is_alive())
        self.assertEqual(storage.num_sets, 2)  # breaker opened after two
        self.assertTrue(c.storage_breakers["write"].is_open())


class TestPrefetchFaultIsolation(CustomTestCase):
    def test_hit_query_exception_reports_zero_hits(self):
        c, storage = _make_controller(threshold=0)
        storage.read_error = OSError("Stale file handle")
        c._all_reduce = Mock(wraps=c._all_reduce)
        ops = [_prefetch_op() for _ in range(3)]
        with _Thread(c, c.prefetch_thread_func, c.prefetch_queue) as thread:
            for op in ops:
                c.prefetch_queue.put(op)
            results = [_get(c.prefetch_hit_queue) for _ in ops]
            self.assertTrue(thread.is_alive())
        self.assertEqual([op.storage_hit_count for op in results], [0, 0, 0])
        self.assertEqual([op.hash_value for op in results], [[], [], []])
        # Every op still joined the cross-rank hit all-reduce.
        self.assertEqual(c._all_reduce.call_count, 3)

    def test_reads_keep_working_when_only_writes_fail(self):
        c, storage = _make_controller(threshold=2)
        with _Thread(c, c.backup_thread_func, c.backup_queue):
            c.backup_queue.put(_backup_op(tag="x"))
            _get(c.ack_backup_queue)
            storage.write_error = OSError("EACCES")
            for i in range(3):
                c.backup_queue.put(_backup_op(tag=f"y{i}"))
                _get(c.ack_backup_queue)
        self.assertTrue(c.storage_breakers["write"].is_open())

        op = _prefetch_op()
        storage.stored.update(
            get_storage_hash_str(op.token_ids, None, page_size=PAGE_SIZE)
        )
        with _Thread(c, c.prefetch_thread_func, c.prefetch_queue):
            c.prefetch_queue.put(op)
            result = _get(c.prefetch_hit_queue)
        self.assertEqual(result.storage_hit_count, 2 * PAGE_SIZE)
        self.assertFalse(c.storage_breakers["read"].is_open())

    def test_read_breaker_skips_hit_query_until_cooldown(self):
        c, storage = _make_controller(threshold=2, cooldown_s=10.0)
        storage.read_error = OSError("EIO")
        with _Thread(c, c.prefetch_thread_func, c.prefetch_queue):
            for _ in range(4):
                c.prefetch_queue.put(_prefetch_op())
                _get(c.prefetch_hit_queue)
            self.assertEqual(storage.num_exists, 2)
            storage.read_error = None
            c.clock.now = 10.0
            c.prefetch_queue.put(_prefetch_op())
            _get(c.prefetch_hit_queue)
        self.assertEqual(storage.num_exists, 3)
        self.assertFalse(c.storage_breakers["read"].is_open())

    def test_transfer_exception_still_emits_every_ack(self):
        c, storage = _make_controller(threshold=0)
        storage.read_error = OSError("Stale file handle")
        op = _prefetch_op(num_pages=3)
        op.hash_value = ["h0", "h1", "h2"]
        op.host_indices = torch.arange(3 * PAGE_SIZE)
        with (
            patch("sglang.srt.managers.cache_controller.STORAGE_BATCH_SIZE", 1),
            _Thread(c, c.prefetch_io_aux_func, c.prefetch_buffer) as thread,
        ):
            c.prefetch_buffer.put(op)
            acks = [_get(c.prefetch_sync_queue) for _ in range(4)]
            self.assertTrue(thread.is_alive())
        # One progress ack per batch (the peers all-reduce each), then final.
        self.assertEqual([a.completed_tokens for a in acks[:3]], [0, 0, 0])
        self.assertTrue(acks[3].completed_req)
        self.assertEqual(storage.num_gets, 1)  # later batches skipped

    def test_unexpected_transfer_error_still_emits_final_ack(self):
        c, _ = _make_controller(threshold=0)
        c._page_transfer = Mock(side_effect=RuntimeError("bug"))
        with _Thread(c, c.prefetch_io_aux_func, c.prefetch_buffer) as thread:
            c.prefetch_buffer.put(_prefetch_op())
            ack = _get(c.prefetch_sync_queue)
            self.assertTrue(thread.is_alive())
        self.assertTrue(ack.completed_req)

    def test_hybrid_sidecar_read_exception_still_acks(self):
        c, storage = _make_controller(hybrid.HybridCacheController, threshold=0)
        storage.batch_get_v2 = Mock(side_effect=OSError("Stale file handle"))
        op = hybrid.PrefetchOperation(
            CacheRequestHandle(rid="req", attempt_id=0),
            list(range(2 * PAGE_SIZE)),
            pool_transfers=[PoolTransfer(PoolName.SWA, keys=["s0"])],
        )
        op.hash_value = ["h0", "h1"]
        c._page_transfer_sidecar(op, kv_completed_pages=2)
        ack = c.prefetch_sync_queue.get_nowait()
        self.assertEqual(ack.pool_hits, {})
        storage.batch_get_v2.assert_called_once()


def _run_hit_query_rank(rank: int, world_size: int, init_file: str) -> None:
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=Path(init_file).as_uri(),
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=20),
    )
    try:
        c, storage = _make_controller(threshold=0)
        c.prefetch_hits_sync_groups = [torch.distributed.group.WORLD]
        op = _prefetch_op()
        # Rank 1 has every page; rank 0's store raises on the hit query.
        if rank == 0:
            storage.read_error = OSError("Stale file handle")
        else:
            storage.batch_exists = lambda keys, extra_info=None: len(keys)
        with _Thread(c, c.prefetch_thread_func, c.prefetch_queue):
            c.prefetch_queue.put(op)
            result = _get(c.prefetch_hit_queue)
        if result.storage_hit_count != 0:
            raise AssertionError(f"rank {rank}: {result.storage_hit_count} != 0")
    finally:
        torch.distributed.destroy_process_group()


class TestHitQueryConsensus(CustomTestCase):
    def test_one_rank_storage_error_does_not_strand_peers(self):
        with TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                _run_hit_query_rank,
                args=(2, str(Path(directory) / "gloo-init")),
                nprocs=2,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
