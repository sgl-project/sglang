"""HiCache storage workers: ordinary backend failures keep the per-operation
ack sequence, a worker that died fails its rank loudly in the next tick, and a
stop drains in lockstep without backend calls."""

import tempfile
import threading
import time
import unittest
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers import cache_controller
from sglang.srt.managers.cache_controller import HiCacheController, PrefetchOperation
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache import hybrid_cache_controller
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    PrefetchOperation as HybridPrefetchOperation,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=70, suite="base-a-test-cpu")

PAGE = 2


def _controller(cls=HiCacheController):
    """A controller with the state its storage workers and stop path read."""
    cc = cls.__new__(cls)
    cc.page_size = PAGE
    cc.storage_stop_event = threading.Event()
    cc.prefetch_thread = cc.prefetch_io_aux_thread = None
    cc.prefetch_sync_thread = cc.backup_thread = None
    cc._storage_worker_failures = {}
    for name in (
        "prefetch_queue",
        "prefetch_buffer",
        "prefetch_sync_queue",
        "prefetch_hit_queue",
        "ack_prefetch_queue",
        "backup_queue",
        "ack_backup_queue",
        "host_mem_release_queue",
    ):
        setattr(cc, name, Queue())
    cc.ack_write_queue = []
    cc.ack_load_queue = []
    cc.extra_host_mem_release_queues = {}
    cc.prefetch_hits_sync_groups = []
    cc.prefetch_completion_sync_groups = []
    # The sidecar ack reduce agrees the host-pool names once per group set.
    cc._host_pool_schemas = []
    if cls is HybridCacheController:
        cc.pp_prefetch_command_thread = None
        cc.pp_prefetch_state_lock = threading.Lock()
        cc.pp_prefetch_states = {}
    return cc


def _tick_check(cc):
    """The tick's rank-local dead-worker check over a bare cache."""
    cache = object.__new__(UnifiedRadixCache)
    cache.tree_core = SimpleNamespace(enable_storage=True)
    cache.cache_controller = cc
    return cache._raise_if_storage_workers_died


def _init_two_ranks(rank, rendezvous):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )


def _wait_for(queue, count, timeout=30.0):
    items = []
    while len(items) < count:
        items.append(queue.get(timeout=timeout))
    return items


def _same_on_both_ranks(value):
    gathered = [None, None]
    torch.distributed.all_gather_object(gathered, value)
    assert gathered[0] == gathered[1], gathered
    return gathered[0]


def _hybrid_operation(rid, pool_transfers):
    op = HybridPrefetchOperation(
        CacheRequestHandle(rid, 0),
        list(range(2 * PAGE)),
        pool_transfers=pool_transfers,
    )
    op.hash_value = ["h0", "h1"]
    op.host_indices = torch.arange(2 * PAGE)
    return op


def _two_rank_backend_failures_keep_the_ack_sequence(rank, rendezvous):
    """Ordinary backend failures on one rank do not change what that rank
    publishes: a sidecar read that raises still publishes its ack, with zero
    hits on every pool after the MIN; a hit query that raises still joins the
    hits reduce with zero hits. The workers stay up and both ranks see the
    same sequences."""
    _init_two_ranks(rank, rendezvous)
    try:
        cc = _controller(HybridCacheController)
        world = [torch.distributed.group.WORLD]
        cc.prefetch_hits_sync_groups = cc.prefetch_completion_sync_groups = world
        cc.mem_pool_host = SimpleNamespace(
            layout_lease=nullcontext,
            entry_map={PoolName.KV: None, PoolName.SWA: None},
        )
        cc._page_transfer_kv_batch = lambda op, hashes, *_: len(hashes)
        fail = {"sidecar_read": False, "query": False}

        def batch_get_v2(transfers, extra_info=None):
            if fail["sidecar_read"] and rank == 1:
                raise RuntimeError("sidecar read failed")
            return {str(t.name): [True] * len(t.keys) for t in transfers}

        def storage_hit_query(operation):
            if fail["query"] and rank == 1:
                raise RuntimeError("hit query failed")
            return list(operation.hash_value), len(operation.hash_value) * PAGE

        cc._sync_trailing_keys = lambda *_: None
        cc.storage_backend = SimpleNamespace(batch_get_v2=batch_get_v2)
        cc._storage_hit_query = storage_hit_query
        cc._start_storage_worker_threads()

        swa = [PoolTransfer(name=PoolName.SWA, keys=["h0", "h1"])]
        transfer_summaries = []
        with patch.object(cache_controller, "STORAGE_BATCH_SIZE", 2):
            for rid, failure in (("ok", None), ("read", "sidecar_read")):
                fail[failure] = True if failure else False
                cc.prefetch_buffer.put(_hybrid_operation(rid, swa))
                acks = _wait_for(cc.ack_prefetch_queue, 3)
                fail["sidecar_read"] = False
                transfer_summaries.append(
                    [
                        (a.rid, a.completed_tokens, a.pool_hits, a.completed_req)
                        for a in acks
                    ]
                )
                assert cc.prefetch_io_aux_thread.is_alive(), rid
            fail["query"] = True
            cc.prefetch_queue.put(_hybrid_operation("query", None))
            queried = cc.prefetch_hit_queue.get(timeout=30)
            fail["query"] = False
            assert cc.prefetch_thread.is_alive()
        agreed = _same_on_both_ranks(
            (transfer_summaries, (queried.storage_hit_count, queried.hash_value))
        )
        no_pool_hits = {str(PoolName.KV): 0, str(PoolName.SWA): 0}
        # A healthy read agrees the two window pages on both ranks; the read
        # that raised on rank 1 becomes zero hits on every pool for both.
        assert agreed == (
            [
                [
                    ("ok", 2 * PAGE, None, None),
                    ("ok", None, {str(PoolName.KV): 0, str(PoolName.SWA): 2}, None),
                    ("ok", None, None, True),
                ],
                [
                    ("read", 2 * PAGE, None, None),
                    ("read", None, no_pool_hits, None),
                    ("read", None, None, True),
                ],
            ],
            (0, []),
        ), agreed
        assert cc.dead_storage_workers() == []
        cc._stop_storage_threads()
        assert cc._storage_worker_threads() == []
    finally:
        torch.distributed.destroy_process_group()


def _two_rank_stop_answers_backlog_without_backend(rank, rendezvous):
    """After the stop event the prefetch worker answers every queued query as
    a miss without a backend call, still joining the hits reduce. The workers
    see the stop at different points (rank 1 before its first query, rank 0
    during it) and still publish identical outcomes."""
    _init_two_ranks(rank, rendezvous)
    try:
        cc = _controller()
        cc.prefetch_hits_sync_groups = [torch.distributed.group.WORLD]

        def batch_exists(hashes, extra_info):
            # Rank 0 is stopped while inside its first query; rank 1 before it.
            cc.storage_stop_event.set()
            return len(hashes)

        cc.storage_backend = SimpleNamespace(
            batch_exists=Mock(side_effect=batch_exists)
        )
        if rank == 1:
            cc.storage_stop_event.set()
        for rid in "abc":
            cc.prefetch_queue.put(PrefetchOperation(rid, list(range(2 * PAGE))))
        cc.prefetch_queue.put(None)
        thread = threading.Thread(target=cc.prefetch_thread_func, daemon=True)
        thread.start()
        thread.join(timeout=30)
        assert not thread.is_alive()
        published = [
            (op.request_id, op.storage_hit_count, op.hash_value)
            for op in _wait_for(cc.prefetch_hit_queue, 3)
        ]
        _same_on_both_ranks(published)
        # Rank 0's real hit on "a" is outvoted by rank 1's stop-time miss.
        assert published == [(rid, 0, []) for rid in "abc"], published
        assert cc.storage_backend.batch_exists.call_count == (1 if rank == 0 else 0)
    finally:
        torch.distributed.destroy_process_group()


def _two_rank_stop_drains_in_flight_transfers(rank, rendezvous):
    """A stop that meets transfers in flight: at the stop event rank 0's IO
    worker is inside operation X's first batch and rank 1's is still inside
    operation W's first batch with X queued behind it. Both ranks publish the
    same ack sequence for W and X (no backend read after the event, the
    queued X taken and acked on rank 1), both sync threads reduce them all,
    and both stops succeed."""
    _init_two_ranks(rank, rendezvous)
    sync_group = torch.distributed.new_group(ranks=[0, 1], backend="gloo")
    try:
        cc = _controller()
        cc.prefetch_hits_sync_groups = cc.prefetch_completion_sync_groups = [sync_group]
        gated = "x" if rank == 0 else "w"
        entered, release = threading.Event(), threading.Event()
        reads = []

        def kv_batch(operation, hashes, *_):
            reads.append((operation.request_id, cc.storage_stop_event.is_set()))
            if operation.request_id == gated and len(reads) == (4 if rank == 0 else 1):
                entered.set()
                release.wait(30)
            return len(hashes)

        cc._page_transfer_kv_batch = kv_batch

        def operation(rid):
            op = PrefetchOperation(rid, list(range(3 * PAGE)))
            op.hash_value = ["h0", "h1", "h2"]
            op.host_indices = torch.arange(3 * PAGE)
            return op

        with patch.object(cache_controller, "STORAGE_BATCH_SIZE", 1):
            cc._start_storage_worker_threads()
            cc.prefetch_buffer.put(operation("w"))
            cc.prefetch_buffer.put(operation("x"))
            assert entered.wait(30)
            # Both IO workers are inside their gated batch; stop in lockstep.
            _same_on_both_ranks("blocked")
            threading.Timer(0.3, release.set).start()
            started = time.monotonic()
            with patch.object(cache_controller, "STORAGE_STOP_JOIN_S", 20.0):
                cc._stop_storage_threads()
            stop_seconds = time.monotonic() - started
            acks = _wait_for(cc.ack_prefetch_queue, 8)
        assert cc._storage_worker_threads() == []
        assert cc.dead_storage_workers() == []
        assert stop_seconds < 10, stop_seconds
        # No backend read was issued after the stop event on either rank.
        assert all(not stopped for _, stopped in reads), reads
        assert [rid for rid, _ in reads] == (
            ["w", "w", "w", "x"] if rank == 0 else ["w"]
        ), reads
        agreed = _same_on_both_ranks(
            [(a.rid, a.completed_tokens, a.completed_req) for a in acks]
        )
        # W: rank 1 read one page before the stop, rank 0 all three; the MIN
        # keeps one. X: rank 0 read one page, rank 1 never read it: zero.
        assert agreed == [
            ("w", PAGE, None),
            ("w", PAGE, None),
            ("w", PAGE, None),
            ("w", None, True),
            ("x", 0, None),
            ("x", 0, None),
            ("x", 0, None),
            ("x", None, True),
        ], agreed
    finally:
        torch.distributed.destroy_process_group()


def _two_rank_stop_waits_for_the_ticket_producer(rank, rendezvous):
    """buffer_only PP>1: the PP ticket thread is a producer into
    prefetch_buffer that may deliver an operation after the stop event. Rank
    0's IO worker is idle at the stop, rank 1's is inside W; the producer puts
    T after the event on both. Both IO workers wait for the producer, both
    ranks publish the same acks for W and T, and both stops return."""
    _init_two_ranks(rank, rendezvous)
    sync_group = torch.distributed.new_group(ranks=[0, 1], backend="gloo")
    try:
        cc = _controller(HybridCacheController)
        cc.pp_rank = 1  # PP1+: the stop only joins the ticket thread
        cc.prefetch_hits_sync_groups = cc.prefetch_completion_sync_groups = [sync_group]
        cc.mem_pool_host = SimpleNamespace(layout_lease=nullcontext)
        entered, release = threading.Event(), threading.Event()
        reads = []

        def kv_batch(operation, hashes, *_):
            reads.append((operation.request_id, cc.storage_stop_event.is_set()))
            if rank == 1 and operation.request_id == "w":
                entered.set()
                release.wait(30)
            return len(hashes)

        cc._page_transfer_kv_batch = kv_batch
        cc._start_storage_worker_threads()

        def ticket_thread():
            # The ticket whose PP-wide reduce was in flight at the stop.
            cc.storage_stop_event.wait()
            time.sleep(0.6)
            cc.prefetch_buffer.put(_hybrid_operation("t", None))

        cc.pp_prefetch_command_thread = threading.Thread(
            target=ticket_thread, name="hicache-pp-ticket", daemon=True
        )
        cc.pp_prefetch_command_thread.start()
        with patch.object(cache_controller, "STORAGE_BATCH_SIZE", 1):
            cc.prefetch_buffer.put(_hybrid_operation("w", None))
            if rank == 1:
                assert entered.wait(30)
            else:
                _wait_for(cc.ack_prefetch_queue, 0)
            _same_on_both_ranks("ready")
            if rank == 1:
                threading.Timer(1.0, release.set).start()
            started = time.monotonic()
            with patch.object(cache_controller, "STORAGE_STOP_JOIN_S", 20.0):
                cc._stop_storage_threads()
            stop_seconds = time.monotonic() - started
            acks = _wait_for(cc.ack_prefetch_queue, 6)
        assert stop_seconds < 10, stop_seconds
        assert cc._storage_worker_threads() == []
        assert cc.pp_prefetch_command_thread is None
        assert all(not stopped for _, stopped in reads), reads
        agreed = _same_on_both_ranks(
            [(a.rid, a.completed_tokens, a.completed_req) for a in acks]
        )
        # W: rank 0 read both pages before the stop, rank 1 one page; the MIN
        # keeps one. T arrived after the event on both: never read, acked.
        assert agreed == [
            ("w", PAGE, None),
            ("w", PAGE, None),
            ("w", None, True),
            ("t", 0, None),
            ("t", 0, None),
            ("t", None, True),
        ], agreed
    finally:
        torch.distributed.destroy_process_group()


def _two_rank_worker_death_under_the_stop_event(rank, rendezvous):
    """Rank 1's IO worker dies on a programming error while the stop event is
    set. Its stop raises naming the failure (the death is recorded, the event
    does not hide it); rank 0's sync thread is left alone in the completed
    tokens reduce rank 1 never joins, so rank 0's stop raises at the bound
    too, instead of hanging: both ranks are out within the bound."""
    _init_two_ranks(rank, rendezvous)
    sync_group = torch.distributed.new_group(ranks=[0, 1], backend="gloo")
    try:
        cc = _controller()
        cc.prefetch_hits_sync_groups = cc.prefetch_completion_sync_groups = [sync_group]

        def kv_batch(operation, hashes, *_):
            if rank == 1:
                raise RuntimeError("host indices out of range")
            return len(hashes)

        cc._page_transfer_kv_batch = kv_batch
        cc._start_storage_worker_threads()
        cc.storage_stop_event.set()
        op = PrefetchOperation("w", list(range(PAGE)))
        op.hash_value = ["h0"]
        op.host_indices = torch.arange(PAGE)
        with patch.object(cache_controller, "STORAGE_BATCH_SIZE", 1):
            # Rank 1's worker is already stopping: the batch read is skipped and
            # it must not raise; make it the transfer scaffolding that fails.
            if rank == 1:
                cc._page_transfer = Mock(side_effect=RuntimeError("bad operation"))
            cc.prefetch_buffer.put(op)
            started = time.monotonic()
            outcome = None
            with patch.object(cache_controller, "STORAGE_STOP_JOIN_S", 3.0):
                try:
                    cc._stop_storage_threads()
                except RuntimeError as e:
                    outcome = str(e)
            elapsed = time.monotonic() - started
        assert elapsed < 6, elapsed
        gathered = [None, None]
        torch.distributed.all_gather_object(gathered, outcome)
        assert gathered[0] == "Failed to stop HiCache storage threads cleanly.", (
            gathered
        )
        assert gathered[1].startswith(
            "HiCache storage worker thread(s) exited on an exception: "
            "hicache-prefetch-io (RuntimeError('bad operation'))"
        ), gathered
        if rank == 1:
            assert cc.dead_storage_workers() == ["hicache-prefetch-io"]
            # Release rank 0's orphaned sync thread so both processes exit.
            torch.distributed.all_reduce(
                torch.tensor(0, dtype=torch.int),
                op=torch.distributed.ReduceOp.MIN,
                group=sync_group,
            )
        else:
            cc.prefetch_sync_thread.join(timeout=30)
            cc._stop_storage_threads()
            assert cc.dead_storage_workers() == []
    finally:
        torch.distributed.destroy_process_group()


class TestStorageWorkersAcrossRanks(unittest.TestCase):
    def _spawn(self, fn):
        with tempfile.TemporaryDirectory(prefix="hicache-workers-") as directory:
            torch.multiprocessing.spawn(
                fn, args=(f"{directory}/store",), nprocs=2, join=True
            )

    def test_backend_failures_keep_the_ack_sequence(self):
        self._spawn(_two_rank_backend_failures_keep_the_ack_sequence)

    def test_stop_answers_the_backlog_without_backend_calls(self):
        self._spawn(_two_rank_stop_answers_backlog_without_backend)

    def test_stop_drains_in_flight_transfers_on_every_rank(self):
        self._spawn(_two_rank_stop_drains_in_flight_transfers)

    def test_stop_waits_for_the_ticket_producer(self):
        self._spawn(_two_rank_stop_waits_for_the_ticket_producer)

    def test_worker_death_under_the_stop_event_fails_both_stops_in_bound(self):
        self._spawn(_two_rank_worker_death_under_the_stop_event)


class TestStorageWorkerLiveness(unittest.TestCase):
    def test_dead_worker_fails_the_rank_in_its_tick(self):
        """A started worker that exited makes this rank's next tick raise,
        naming it; live workers, a controller that started none, and a worker
        finishing under a pending stop do not."""
        cc = _controller()
        _tick_check(cc)()
        release = threading.Event()
        cc.backup_thread = threading.Thread(
            target=release.wait, name="hicache-backup", daemon=True
        )
        cc.backup_thread.start()
        _tick_check(cc)()
        dead = threading.Thread(target=lambda: None, name="hicache-prefetch-io")
        dead.start()
        dead.join()
        cc.prefetch_io_aux_thread = dead
        with self.assertRaises(RuntimeError) as raised:
            _tick_check(cc)()
        self.assertEqual(
            str(raised.exception),
            "HiCache storage worker thread(s) exited on this rank: hicache-prefetch-io",
        )
        # Under a pending stop a finished worker did what it was asked, but a
        # recorded crash is a death regardless.
        cc.storage_stop_event.set()
        _tick_check(cc)()
        cc._storage_worker_failures["hicache-backup"] = "RuntimeError('boom')"
        with self.assertRaises(RuntimeError) as raised:
            _tick_check(cc)()
        self.assertIn("hicache-backup", str(raised.exception))
        release.set()

    def test_escaping_transfer_failure_kills_the_worker_and_the_rank_raises(self):
        """An exception the transfer lets escape (the host layout lease
        poisoned, before any batch ack) ends the IO worker, as on main, and
        the owning rank's next tick raises naming it; no ack is invented."""
        cc = _controller(HybridCacheController)

        @contextmanager
        def layout_lease():
            raise RuntimeError("Unified host layout lease poisoned.")
            yield

        cc.mem_pool_host = SimpleNamespace(layout_lease=layout_lease)
        cc._start_storage_worker_threads()
        with self.assertLogs(cache_controller.logger, level="ERROR") as logs:
            cc.prefetch_buffer.put(_hybrid_operation("r", None))
            cc.prefetch_io_aux_thread.join(timeout=30)
        self.assertFalse(cc.prefetch_io_aux_thread.is_alive())
        self.assertIn("prefetch IO worker exits", logs.output[0])
        # Only main's completed_req from the finally, no batch ack.
        acks = _wait_for(cc.ack_prefetch_queue, 1)
        self.assertEqual((acks[0].rid, acks[0].completed_req), ("r", True))
        self.assertEqual(cc.dead_storage_workers(), ["hicache-prefetch-io"])
        with self.assertRaises(RuntimeError) as raised:
            _tick_check(cc)()
        self.assertIn("hicache-prefetch-io", str(raised.exception))
        # The stop joins the survivors and still fails the rank on the death.
        with self.assertRaises(RuntimeError) as raised:
            cc._stop_storage_threads()
        self.assertIn(
            "hicache-prefetch-io (RuntimeError('Unified host layout lease poisoned.'))",
            str(raised.exception),
        )

    def test_invariant_violation_before_the_sidecar_ack_kills_the_worker(self):
        """A sidecar key re-alignment that fails is not a backend failure: it
        propagates, the IO worker exits, the death is recorded, the tick
        raises and so does the stop, naming the worker and the error."""
        cc = _controller(HybridCacheController)
        cc.mem_pool_host = SimpleNamespace(layout_lease=nullcontext)
        cc._page_transfer_kv_batch = lambda op, hashes, *_: len(hashes)
        cc._sync_trailing_keys = Mock(
            side_effect=AssertionError("trailing-page sidecar keys misaligned")
        )
        cc._start_storage_worker_threads()
        with self.assertLogs(cache_controller.logger, level="ERROR"):
            cc.prefetch_buffer.put(
                _hybrid_operation(
                    "r", [PoolTransfer(name=PoolName.SWA, keys=["h0", "h1"])]
                )
            )
            cc.prefetch_io_aux_thread.join(timeout=30)
        self.assertFalse(cc.prefetch_io_aux_thread.is_alive())
        self.assertEqual(cc.dead_storage_workers(), ["hicache-prefetch-io"])
        with self.assertRaises(RuntimeError):
            _tick_check(cc)()
        with self.assertRaises(RuntimeError) as raised:
            cc._stop_storage_threads()
        self.assertIn("hicache-prefetch-io (AssertionError(", str(raised.exception))

    def test_tick_calls_the_dead_worker_check_before_the_drain(self):
        class _Proceeded(Exception):
            pass

        cc = _controller()
        cache = object.__new__(UnifiedRadixCache)
        cache.tree_core = SimpleNamespace(enable_storage=True)
        cache.cache_controller = cc
        cache.linker = None
        cache._drain_async_work = Mock(side_effect=_Proceeded)
        dead = threading.Thread(target=lambda: None, name="hicache-backup")
        dead.start()
        dead.join()
        cc.backup_thread = dead
        with self.assertRaises(RuntimeError):
            cache.check_hicache_events()
        cache._drain_async_work.assert_not_called()
        # Under a pending stop the finished worker is not dead: the tick goes on.
        cc.storage_stop_event.set()
        with self.assertRaises(_Proceeded):
            cache.check_hicache_events()


class TestStorageWorkerStop(unittest.TestCase):
    def test_backup_stops_issuing_batches_after_stop_and_acks_progress(self):
        cc = _controller()
        cc.storage_config = SimpleNamespace(is_storage_writer=True)
        operation = SimpleNamespace(
            hash_value=["h0", "h1", "h2"],
            host_indices=torch.arange(3 * PAGE),
            prefix_keys=None,
            completed_tokens=0,
        )

        def page_set(hashes, host_indices, extra_info):
            cc.storage_stop_event.set()
            return True

        cc.page_set_func = Mock(side_effect=page_set)
        cc.backup_queue.put(operation)
        thread = threading.Thread(target=cc.backup_thread_func, daemon=True)
        with patch.object(cache_controller, "STORAGE_BATCH_SIZE", 1):
            thread.start()
            acked = cc.ack_backup_queue.get(timeout=30)
            thread.join(timeout=30)
        self.assertFalse(thread.is_alive())
        self.assertIs(acked, operation)
        self.assertEqual(cc.page_set_func.call_count, 1)
        self.assertEqual(operation.completed_tokens, PAGE)

    def test_transfer_stops_reading_after_stop_but_acks_every_batch(self):
        cc = _controller()
        operation = PrefetchOperation("r", list(range(3 * PAGE)))
        operation.hash_value = ["h0", "h1", "h2"]
        operation.host_indices = torch.arange(3 * PAGE)

        def kv_batch(op, hashes, *_):
            cc.storage_stop_event.set()
            return len(hashes)

        cc._page_transfer_kv_batch = Mock(side_effect=kv_batch)
        with patch.object(cache_controller, "STORAGE_BATCH_SIZE", 1):
            self.assertEqual(cc._page_transfer(operation), 1)
        self.assertEqual(cc._page_transfer_kv_batch.call_count, 1)
        acks = [cc.prefetch_sync_queue.get_nowait() for _ in range(3)]
        self.assertEqual([a.completed_tokens for a in acks], [PAGE, PAGE, PAGE])
        self.assertTrue(cc.prefetch_sync_queue.empty())

        # The hybrid sidecar read is skipped after the stop; its ack still goes out.
        hybrid = _controller(HybridCacheController)
        hybrid.storage_stop_event.set()
        hybrid.storage_backend = SimpleNamespace(batch_get_v2=Mock())
        op = _hybrid_operation(
            "s", [PoolTransfer(name=PoolName.SWA, keys=["h0", "h1"])]
        )
        hybrid._page_transfer_sidecar(op, kv_completed_pages=2)
        hybrid.storage_backend.batch_get_v2.assert_not_called()
        ack = hybrid.prefetch_sync_queue.get_nowait()
        self.assertEqual(
            (ack.rid, ack.pool_hits, ack.completed_tokens), ("s", {}, None)
        )

    def test_hybrid_backup_stops_issuing_batches_after_stop(self):
        """Both hybrid backup loops, the KV batches and the sidecar groups,
        check the stop event per batch; the ack carries what completed."""
        cc = _controller(HybridCacheController)
        cc.storage_config = SimpleNamespace(is_storage_writer=True)

        def page_set(hashes, host_indices, extra_info):
            cc.storage_stop_event.set()
            return True

        cc.page_set_func = Mock(side_effect=page_set)
        operation = SimpleNamespace(
            hash_value=["h0", "h1", "h2"],
            host_indices=torch.arange(3 * PAGE),
            prefix_keys=None,
            completed_tokens=0,
        )
        with patch.object(hybrid_cache_controller, "STORAGE_BATCH_SIZE", 1):
            cc._page_backup_kv_with_trace(operation)
        self.assertEqual(cc.page_set_func.call_count, 1)
        self.assertEqual(operation.completed_tokens, PAGE)

        cc = _controller(HybridCacheController)
        cc.storage_config = SimpleNamespace(is_storage_writer=False)
        cc.should_backup = lambda transfer: True
        cc._resolve_sidecar_kv_derived_pool_transfers = Mock()
        cc._resolve_sidecar_nonkv_derived_pool_transfers = Mock()

        def batch_set_v2(transfers, extra_info=None):
            cc.storage_stop_event.set()
            return {str(t.name): [True] * len(t.keys) for t in transfers}

        cc.storage_backend = SimpleNamespace(
            batch_set_v2=Mock(side_effect=batch_set_v2)
        )
        operation = SimpleNamespace(
            id=7,
            hash_value=["h0", "h1"],
            host_indices=torch.arange(2 * PAGE),
            prefix_keys=["p"],
            completed_tokens=0,
            pool_transfers=[
                PoolTransfer(name=PoolName.SWA, keys=["h1"]),
                PoolTransfer(name=PoolName.MAMBA, keys=["h0", "h1"]),
            ],
            pool_storage_result=Mock(),
            trace_ctx=None,
        )
        cc._page_backup_with_stable_layout(operation)
        # Two trailing-chain groups; the second is not written after the stop.
        self.assertEqual(cc.storage_backend.batch_set_v2.call_count, 1)

    def test_hybrid_sync_thread_drains_the_acks_the_io_worker_still_owes(self):
        cc = _controller(HybridCacheController)
        operation = PrefetchOperation("r", list(range(PAGE)))
        operation.is_pp_broadcast = False

        def io_worker():
            time.sleep(0.3)
            for tokens in (PAGE, None):
                cc.prefetch_sync_queue.put(
                    cache_controller.PrefetchAck(
                        rid="r",
                        operation=operation,
                        completed_tokens=tokens,
                        completed_req=tokens is None,
                    )
                )

        cc.prefetch_io_aux_thread = threading.Thread(target=io_worker, daemon=True)
        cc.prefetch_sync_thread = threading.Thread(
            target=cc.prefetch_sync_thread_func, daemon=True
        )
        cc.prefetch_io_aux_thread.start()
        cc.storage_stop_event.set()
        cc.prefetch_sync_thread.start()
        cc.prefetch_sync_thread.join(timeout=10)
        self.assertFalse(cc.prefetch_sync_thread.is_alive())
        acks = _wait_for(cc.ack_prefetch_queue, 2, timeout=1.0)
        self.assertEqual([a.completed_req for a in acks], [False, True])

    def test_stop_join_bound_is_one_budget_for_all_workers(self):
        cc = _controller()
        release = threading.Event()
        for name in ("prefetch_thread", "backup_thread"):
            thread = threading.Thread(target=release.wait, name=name, daemon=True)
            thread.start()
            setattr(cc, name, thread)
        started = time.monotonic()
        with (
            patch.object(cache_controller, "STORAGE_STOP_JOIN_S", 0.4),
            self.assertRaises(RuntimeError),
        ):
            cc._stop_storage_threads()
        self.assertLess(time.monotonic() - started, 0.7)
        release.set()

    def test_unstarted_worker_handles_are_not_counted(self):
        cc = _controller()
        cc.backup_thread = threading.Thread(target=lambda: None, name="hicache-backup")
        self.assertEqual(cc._storage_worker_threads(), [])
        self.assertEqual(cc.dead_storage_workers(), [])
        cc._stop_storage_threads()

    def test_sync_thread_drains_the_acks_the_io_worker_still_owes(self):
        cc = _controller()
        operation = PrefetchOperation("r", list(range(PAGE)))

        def io_worker():
            time.sleep(0.3)
            for tokens in (PAGE, None):
                cc.prefetch_sync_queue.put(
                    cache_controller.PrefetchAck(
                        rid="r",
                        operation=operation,
                        completed_tokens=tokens,
                        completed_req=tokens is None,
                    )
                )

        cc.prefetch_io_aux_thread = threading.Thread(target=io_worker, daemon=True)
        cc.prefetch_sync_thread = threading.Thread(
            target=cc.prefetch_sync_thread_func, daemon=True
        )
        # Started in the controller's order: the IO worker before the sync
        # thread, so the sync thread never sees an unstarted worker as exited.
        cc.prefetch_io_aux_thread.start()
        cc.storage_stop_event.set()
        cc.prefetch_sync_thread.start()
        cc.prefetch_sync_thread.join(timeout=10)
        self.assertFalse(cc.prefetch_sync_thread.is_alive())
        self.assertFalse(cc.prefetch_io_aux_thread.is_alive())
        acks = _wait_for(cc.ack_prefetch_queue, 2, timeout=1.0)
        self.assertEqual([a.completed_req for a in acks], [False, True])

    def test_stop_raises_at_the_bound_naming_the_stuck_workers(self):
        cc = _controller()
        release = threading.Event()
        cc.backup_thread = threading.Thread(
            target=release.wait, name="hicache-backup", daemon=True
        )
        cc.backup_thread.start()
        with (
            patch.object(cache_controller, "STORAGE_STOP_JOIN_S", 0.3),
            self.assertLogs(cache_controller.logger, level="ERROR") as logs,
            self.assertRaises(RuntimeError),
        ):
            cc._stop_storage_threads()
        self.assertIn("['hicache-backup']", logs.output[0])
        # The thread stays in the started set until it is actually stopped,
        # and while the stop event is set it is not reported dead: the caller
        # of the failed stop retries, and the retry joins it.
        self.assertEqual(cc._storage_worker_threads(), [cc.backup_thread])
        self.assertEqual(cc.dead_storage_workers(), [])
        self.assertTrue(cc.storage_stop_event.is_set())
        threading.Timer(0.1, release.set).start()
        cc._stop_storage_threads()
        self.assertIsNone(cc.backup_thread)
        self.assertEqual(cc._storage_worker_threads(), [])
        self.assertEqual(cc.dead_storage_workers(), [])

    def test_hybrid_stop_sets_the_event_before_the_ticket_join(self):
        """A PP ticket thread that does not stop within the bound leaves the
        four workers under a set stop event, so they issue no backend calls."""
        cc = _controller(HybridCacheController)
        cc.pp_rank = 1
        release = threading.Event()
        cc.pp_prefetch_command_thread = threading.Thread(
            target=release.wait, name="hicache-pp-ticket", daemon=True
        )
        cc.pp_prefetch_command_thread.start()
        with (
            patch.object(cache_controller, "STORAGE_STOP_JOIN_S", 0.2),
            self.assertRaises(RuntimeError),
        ):
            cc._stop_storage_threads()
        self.assertTrue(cc.storage_stop_event.is_set())
        release.set()
        cc.pp_prefetch_command_thread.join(timeout=5)
        cc._stop_storage_threads()
        self.assertIsNone(cc.pp_prefetch_command_thread)

    def test_hybrid_liveness_set_excludes_the_pp_ticket_thread(self):
        # PP1+'s copy exits on PP0's None ticket one relay step before PP1+
        # processes the same flush; counting it would raise falsely there.
        cc = _controller(HybridCacheController)
        finished = threading.Thread(target=lambda: None, name="hicache-pp-ticket")
        finished.start()
        finished.join()
        cc.pp_prefetch_command_thread = finished
        self.assertEqual(cc._storage_worker_threads(), [])
        self.assertEqual(cc.dead_storage_workers(), [])


if __name__ == "__main__":
    unittest.main()
