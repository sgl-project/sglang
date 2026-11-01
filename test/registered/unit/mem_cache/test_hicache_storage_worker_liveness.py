"""HiCache storage workers: survive per-operation failures, fail loud on every
rank in the same tick when one died, and stop without querying the backlog."""

import tempfile
import threading
import unittest
from contextlib import contextmanager
from datetime import timedelta
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers import cache_controller
from sglang.srt.managers.cache_controller import HiCacheController, PrefetchOperation
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    PrefetchOperation as HybridPrefetchOperation,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

PAGE = 2


def _controller(cls=HiCacheController):
    """A controller with the state its storage workers and stop path read."""
    cc = cls.__new__(cls)
    cc.page_size = PAGE
    cc.storage_stop_event = threading.Event()
    cc.prefetch_thread = cc.prefetch_io_aux_thread = None
    cc.prefetch_sync_thread = cc.backup_thread = None
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
    if cls is HybridCacheController:
        cc.pp_prefetch_command_thread = None
        cc.pp_prefetch_state_lock = threading.Lock()
        cc.pp_prefetch_states = {}
    return cc


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


def _two_rank_transfer_failure_keeps_worker(rank, rendezvous):
    """An exception escaping the transfer body on one rank (the host layout
    lease raising on release; a sidecar key re-alignment failing; the lease
    raising on acquire, before any batch ack) leaves the IO worker alive, and
    both ranks drain the same ack sequence: the failing rank publishes the acks
    its transfer did not reach, carrying the prefix it really completed."""
    _init_two_ranks(rank, rendezvous)
    try:
        cc = _controller(HybridCacheController)
        world = [torch.distributed.group.WORLD]
        cc.prefetch_hits_sync_groups = cc.prefetch_completion_sync_groups = world
        fail = {"lease_exit": False, "trailing_keys": False, "lease_enter": False}

        @contextmanager
        def layout_lease():
            if fail["lease_enter"] and rank == 1:
                raise RuntimeError("Unified host layout lease poisoned.")
            yield
            if fail["lease_exit"] and rank == 1:
                raise RuntimeError(
                    "Unified host layout lease released without acquire."
                )

        cc.mem_pool_host = SimpleNamespace(layout_lease=layout_lease)
        cc._page_transfer_kv_batch = lambda op, hashes, *_: len(hashes)

        def sync_trailing_keys(*_):
            if fail["trailing_keys"] and rank == 1:
                raise AssertionError("trailing-page sidecar keys misaligned")

        cc._sync_trailing_keys = sync_trailing_keys
        cc.storage_backend = SimpleNamespace(
            batch_get_v2=lambda transfers, extra_info=None: {
                t.name.value: [True] * len(t.keys) for t in transfers
            }
        )
        cc._start_storage_worker_threads()

        def operation(rid, pool_transfers):
            op = HybridPrefetchOperation(
                CacheRequestHandle(rid, 0),
                list(range(2 * PAGE)),
                pool_transfers=pool_transfers,
            )
            op.hash_value = ["h0", "h1"]
            op.host_indices = torch.arange(2 * PAGE)
            return op

        swa = [PoolTransfer(name=PoolName.SWA, keys=["h0", "h1"])]
        summaries = []
        # (request id, failure to arm, pool transfers, batch size, acks expected)
        for rid, failure, transfers, batch_size, count in (
            ("lease", "lease_exit", None, 2, 2),
            ("sidecar", "trailing_keys", swa, 2, 3),
            ("acquire", "lease_enter", swa, 1, 4),
        ):
            fail[failure] = True
            with patch.object(cache_controller, "STORAGE_BATCH_SIZE", batch_size):
                cc.prefetch_buffer.put(operation(rid, transfers))
                acks = _wait_for(cc.ack_prefetch_queue, count)
            fail[failure] = False
            summaries.append(
                [
                    (a.rid, a.completed_tokens, a.pool_hits, a.completed_req)
                    for a in acks
                ]
            )
            assert cc.prefetch_io_aux_thread.is_alive(), rid
            assert cc.storage_workers_alive(), rid
        agreed = _same_on_both_ranks(summaries)
        no_pool_hits = {p.value: 0 for p in PoolName}
        # The failing rank's batch ack went out before the lease raised, so the
        # request keeps the prefix it has; the sidecar miss on one rank
        # becomes zero hits on every pool for both.
        assert agreed[0] == [
            ("lease", 2 * PAGE, None, None),
            ("lease", None, None, True),
        ]
        assert agreed[1] == [
            ("sidecar", 2 * PAGE, None, None),
            ("sidecar", None, no_pool_hits, None),
            ("sidecar", None, None, True),
        ]
        # Nothing was published before the acquire raised: rank 1 still owes
        # two batch acks (prefix 0) and the sidecar ack, rank 0's real
        # progress is outvoted by the MIN, and the sequences line up.
        assert agreed[2] == [
            ("acquire", 0, None, None),
            ("acquire", 0, None, None),
            ("acquire", None, no_pool_hits, None),
            ("acquire", None, None, True),
        ], agreed[2]

        cc._stop_storage_threads()
        assert cc._storage_worker_threads() == []
        assert cc.storage_workers_alive()
    finally:
        torch.distributed.destroy_process_group()


def _tree(cc):
    cache = object.__new__(UnifiedRadixCache)
    cache.tree_core = SimpleNamespace(
        enable_storage=True, write_back_duplicate_reclaim_digest=0
    )
    cache.pp_rank, cache.pp_size, cache.pp_group = 0, 1, None
    cache.host_memory_mode = "cache"
    # The tick also reads the stop policy and the live prefetch set.
    cache.prefetch_stop_policy = "wait_complete"
    cache.ongoing_prefetch = {}
    cache.attn_cp_group = cache.attn_tp_group = None
    cache.tp_group, cache.tp_world_size = torch.distributed.group.WORLD, 2
    cache.cache_controller = cc
    return cache


def _two_rank_dead_worker_raises_on_both(rank, rendezvous):
    """A worker that exited on one rank makes every rank raise in the same
    tick, the failing rank naming the thread; a controller with no started
    workers reports alive and the tick returns its counts."""
    _init_two_ranks(rank, rendezvous)
    try:
        cc = _controller()
        cache = _tree(cc)
        assert cache._sync_hicache_ready_counts() == (0, 0, (0, 0, 0, 0), ())

        release = threading.Event()
        if rank == 1:
            dead = threading.Thread(target=lambda: None, name="hicache-prefetch-io")
            dead.start()
            dead.join()
            cc.prefetch_io_aux_thread = dead
        else:
            cc.backup_thread = threading.Thread(
                target=release.wait, name="hicache-backup", daemon=True
            )
            cc.backup_thread.start()
        outcomes = []
        for _ in range(2):
            try:
                cache._sync_hicache_ready_counts()
                outcomes.append(None)
            except RuntimeError as e:
                outcomes.append(str(e))
        release.set()
        gathered = [None, None]
        torch.distributed.all_gather_object(gathered, outcomes)
        assert (
            gathered[0]
            == ["HiCache storage worker thread(s) exited on a peer rank"] * 2
        )
        assert (
            gathered[1]
            == [
                "HiCache storage worker thread(s) exited on this rank: hicache-prefetch-io"
            ]
            * 2
        ), gathered
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


class TestStorageWorkersAcrossRanks(unittest.TestCase):
    def _spawn(self, fn):
        with tempfile.TemporaryDirectory(prefix="hicache-workers-") as directory:
            torch.multiprocessing.spawn(
                fn, args=(f"{directory}/store",), nprocs=2, join=True
            )

    def test_transfer_failure_keeps_the_worker_and_the_ack_sequence(self):
        self._spawn(_two_rank_transfer_failure_keeps_worker)

    def test_dead_worker_raises_on_every_rank_in_the_same_tick(self):
        self._spawn(_two_rank_dead_worker_raises_on_both)

    def test_stop_answers_the_backlog_without_backend_calls(self):
        self._spawn(_two_rank_stop_answers_backlog_without_backend)


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

    def test_stop_logs_stuck_workers_and_raises_at_the_bound(self):
        cc = _controller()
        release = threading.Event()
        cc.backup_thread = threading.Thread(
            target=release.wait, name="hicache-backup", daemon=True
        )
        cc.backup_thread.start()
        with (
            patch.object(cache_controller, "STORAGE_STOP_JOIN_S", 0.3),
            patch.object(cache_controller, "STORAGE_STOP_LOG_EVERY_S", 0.1),
            self.assertLogs(cache_controller.logger, level="WARNING") as logs,
            self.assertRaises(RuntimeError),
        ):
            cc._stop_storage_threads()
        progress = [line for line in logs.output if "still running" in line]
        self.assertGreaterEqual(len(progress), 2)
        self.assertIn("hicache-backup in wait (threading.py", progress[0])
        # The thread stays in the started set until it is actually stopped,
        # and a worker that then finishes under the pending stop did what it
        # was asked: the next tick must not report it dead (the caller of the
        # failed stop retries it instead).
        self.assertEqual(cc._storage_worker_threads(), [cc.backup_thread])
        self.assertEqual(cc.dead_storage_workers(), [])
        release.set()
        cc.backup_thread.join(timeout=5)
        self.assertFalse(cc.backup_thread.is_alive())
        self.assertTrue(cc.storage_stop_event.is_set())
        self.assertEqual(cc.dead_storage_workers(), [])
        self.assertTrue(cc.storage_workers_alive())
        cc._stop_storage_threads()
        self.assertIsNone(cc.backup_thread)
        self.assertEqual(cc._storage_worker_threads(), [])

    def test_hybrid_liveness_set_excludes_the_pp_ticket_thread(self):
        # PP1+'s copy exits on PP0's None ticket one relay step before PP1+
        # processes the same flush; counting it would raise falsely there.
        cc = _controller(HybridCacheController)
        finished = threading.Thread(target=lambda: None, name="hicache-pp-ticket")
        finished.start()
        finished.join()
        cc.pp_prefetch_command_thread = finished
        self.assertEqual(cc._storage_worker_threads(), [])
        self.assertTrue(cc.storage_workers_alive())


if __name__ == "__main__":
    unittest.main()
