"""HiCache storage worker collectives must survive a failed gloo collective
and the scheduler must degrade (rank-agreed) instead of stranding requests."""

import os
import tempfile
import threading
import time
import unittest
from array import array
from collections import defaultdict, deque
from datetime import timedelta
from queue import Empty, Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist

from sglang.srt.managers.cache_controller import HiCacheController, PrefetchAck
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
    PrefetchOperation,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage_prefetch import StoragePrefetchRetries
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedRadixCache,
    _OngoingPrefetch,
)
from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

PAGE = 2
TOKENS = list(range(4))
BATCH_TOKENS = PAGE * 64
# A failing storage sync group times out after SYNC_TIMEOUT_S; the hung rank
# releases its backend call at HOLD_S, after every peer collective on that
# group has timed out. A healthy group outlives both by a wide margin, so a
# rank that blocks on it for its full timeout is a visible hang.
SYNC_TIMEOUT_S = 1.0
HOLD_S = 2.5
HEALTHY_TIMEOUT_S = 8.0
FAST_S = HEALTHY_TIMEOUT_S / 2


class _HoldingBackend:
    """batch_exists blocks on `hold` for the first call when hold_first; every
    call reports all pages present."""

    def __init__(self, hold_first: bool):
        self.hold_first = hold_first
        self.hold = threading.Event()
        self.calls = 0

    def batch_exists(self, hashes, extra_info=None):
        self.calls += 1
        if self.hold_first and self.calls == 1:
            self.hold.wait()
        return len(hashes)


def _idle_thread(stop_event: threading.Event) -> threading.Thread:
    thread = threading.Thread(target=stop_event.wait, daemon=True)
    thread.start()
    return thread


def _controller(hits_group, completion_group, backend) -> HybridCacheController:
    cc = HybridCacheController.__new__(HybridCacheController)
    cc.page_size = PAGE
    cc.storage_backend = backend
    cc.storage_stop_event = threading.Event()
    cc.prefetch_queue = Queue()
    cc.prefetch_hit_queue = Queue()
    cc.prefetch_sync_queue = Queue()
    cc.ack_prefetch_queue = Queue()
    cc.ack_backup_queue = Queue()
    cc.host_mem_release_queue = Queue()
    cc.extra_host_mem_release_queues = {}
    cc.ack_write_queue = []
    cc.ack_load_queue = []
    cc.start_writing = Mock()
    cc.append_host_mem_release = Mock()
    cc.prefetch_hits_sync_groups = [hits_group] if hits_group is not None else []
    cc.prefetch_completion_sync_groups = (
        [completion_group] if completion_group is not None else []
    )
    cc.storage_sync_failed = False
    cc.storage_sync_failure = None
    cc._storage_sync_lock = threading.Lock()
    cc.storage_failure_policy = "disable"
    cc.pp_prefetch_command_group = None
    cc.pp_prefetch_command_thread = None
    cc.pp_prefetch_decisions = {}
    # Workers not under test idle until the stop event.
    cc.prefetch_thread = _idle_thread(cc.storage_stop_event)
    cc.prefetch_io_aux_thread = _idle_thread(cc.storage_stop_event)
    cc.prefetch_sync_thread = _idle_thread(cc.storage_stop_event)
    cc.backup_thread = _idle_thread(cc.storage_stop_event)
    return cc


def _scheduler_cache(cc) -> UnifiedRadixCache:
    """The scheduler-thread side: the real check_hicache_events round, whose
    cross-rank MIN runs on the default (healthy) process group like the TP
    group does; only the L2 ack checks and the buffer flush are stubbed."""
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.cache_controller = cc
    # enable_storage and page_size are proxied to the tree core.
    cache.tree_core = SimpleNamespace(
        enable_storage=True, page_size=PAGE, write_back_duplicate_reclaim_digest=0
    )
    cache.storage_unavailable = False
    cache._storage_unavailable_warned_at = 0.0
    cache.host_memory_mode = "buffer_only"
    cache.prefetch_threshold = PAGE
    cache.enable_storage_metrics = False
    cache.storage_metrics_collector = None
    cache.linker = None
    cache.pp_rank = 0
    cache.pp_size = 1
    cache.pp_group = None
    cache.tp_group = None
    cache.tp_world_size = dist.get_world_size()
    cache.attn_cp_group = None
    cache.attn_tp_group = None
    cache.work_list = []
    cache.writing_check = Mock()
    cache.loading_check = Mock()
    cache.storage_prefetch_retries = StoragePrefetchRetries()
    cache._storage_prefetch_wait = {}
    cache._prefetch_outcome_stats = defaultdict(int)
    cache._storage_prefetch_hit_remaining_by_reqid = {}
    cache._record_storage_prefetch_hit = Mock()
    cache.revoke_pending_prefetch = Mock()
    cache._handle_prefetch_result = Mock()
    cache.buffer_pipeline = SimpleNamespace(
        pending_hit_allocs=deque(),
        has_staged=lambda request: False,
        flush_pending_writes=Mock(),
    )
    cache.ongoing_prefetch = {}
    return cache


def _operation(rid: str, host_indices=None) -> PrefetchOperation:
    operation = PrefetchOperation(CacheRequestHandle(rid, 0), TOKENS)
    operation.stats_requested_tokens = len(TOKENS)
    operation.host_indices = host_indices
    return operation


def _track(cache, operation) -> None:
    cache.ongoing_prefetch[operation.handle] = _OngoingPrefetch(
        0, RadixKey(array("q", TOKENS)), operation.host_indices, operation, None, {}
    )


def _ack(operation, completed=None, done=False) -> PrefetchAck:
    return PrefetchAck(
        rid=operation.request_id,
        operation=operation,
        completed_tokens=completed,
        completed_req=True if done else None,
    )


def _drain(queue: Queue, want: int, budget_s: float) -> list:
    items = []
    deadline = time.monotonic() + budget_s
    while len(items) < want and time.monotonic() < deadline:
        try:
            items.append(queue.get(timeout=0.05))
        except Empty:
            continue
    return items


def _same_on_every_rank(snapshot):
    snapshots = [None] * dist.get_world_size()
    dist.all_gather_object(snapshots, snapshot)
    assert all(other == snapshots[0] for other in snapshots), snapshots
    return snapshots[0]


def _wait_for(predicate, budget_s: float) -> None:
    t0 = time.monotonic()
    while not predicate():
        assert time.monotonic() - t0 < budget_s, "condition not met in time"
        time.sleep(0.02)


def _sync_groups():
    """Fresh (failing, healthy) gloo groups, created collectively."""
    failing = dist.new_group(backend="gloo", timeout=timedelta(seconds=SYNC_TIMEOUT_S))
    healthy = dist.new_group(
        backend="gloo", timeout=timedelta(seconds=HEALTHY_TIMEOUT_S)
    )
    return failing, healthy


def _start_workers(cc) -> tuple[threading.Thread, threading.Thread]:
    query_thread = threading.Thread(target=cc.prefetch_thread_func, daemon=True)
    sync_thread = threading.Thread(target=cc.prefetch_sync_thread_func, daemon=True)
    query_thread.start()
    sync_thread.start()
    cc.prefetch_thread, cc.prefetch_sync_thread = query_thread, sync_thread
    return query_thread, sync_thread


def _stop(cc, *workers) -> None:
    cc.storage_stop_event.set()
    cc.prefetch_queue.put(None)
    cc.prefetch_sync_queue.put(None)
    for worker in workers:
        worker.join(5)


def _hit_query_timeout_body(rank: int) -> None:
    """Rank 1 hangs in its backend; rank 0's hit MIN times out. Both query
    threads must survive, the untouched completion group must stay usable on
    both ranks, and check_hicache_events must degrade every rank at once."""
    hits_group, completion_group = _sync_groups()
    backend = _HoldingBackend(hold_first=(rank == 1))
    cc = _controller(hits_group, completion_group, backend)
    get_storage_hash_str(TOKENS, None, page_size=PAGE)  # JIT warm-up off the clock
    query_thread, sync_thread = _start_workers(cc)
    cache = _scheduler_cache(cc)
    dist.barrier()

    t0 = time.monotonic()
    ops = [_operation("op1"), _operation("op2")]
    for operation in ops:
        _track(cache, operation)
        cc.prefetch_queue.put(operation)
    if rank == 0:
        _wait_for(lambda: cc.storage_sync_failed, budget_s=FAST_S)
    # Cross-group: rank 1's query thread is still hung and rank 0 has already
    # failed on the hits group, yet an ack on the completion group must still
    # reduce promptly on both ranks (a peer that skipped it would strand this
    # rank for that group's whole timeout).
    dist.barrier()
    transfer = _operation("xfer", host_indices=torch.arange(BATCH_TOKENS))
    cc.prefetch_sync_queue.put(_ack(transfer, BATCH_TOKENS - rank * 10))
    acks = _drain(cc.ack_prefetch_queue, want=1, budget_s=FAST_S)
    assert [(a.rid, a.completed_tokens) for a in acks] == [
        ("xfer", BATCH_TOKENS - 10)
    ], acks
    if rank == 1:
        time.sleep(max(0.0, HOLD_S - (time.monotonic() - t0)))
        backend.hold.set()
    published = _drain(cc.prefetch_hit_queue, want=2, budget_s=FAST_S)
    published_at = time.monotonic() - t0
    # The hung rank's own collective fails as soon as it enters (the peer's
    # timeout closed the pair), so op 1 is a published miss on both ranks and
    # the query thread is alive to serve op 2: a retry on the broken group
    # fails at once instead of waiting out another timeout.
    assert [(op.request_id, op.storage_hit_count) for op in published] == [
        ("op1", 0),
        ("op2", 0),
    ], published
    assert published_at < (SYNC_TIMEOUT_S if rank == 0 else HOLD_S) + FAST_S / 2
    assert query_thread.is_alive() and sync_thread.is_alive()
    assert cc.storage_sync_failed and not cc.storage_workers_healthy()
    op3 = _operation("op3")
    _track(cache, op3)
    cc.prefetch_queue.put(op3)
    assert [op.request_id for op in _drain(cc.prefetch_hit_queue, 1, FAST_S)] == ["op3"]
    for operation in published + [op3]:
        cc.prefetch_hit_queue.put(operation)

    # Scheduler side, the production round: the MIN-reduced health bit
    # degrades every rank in the same tick; the drained queries resolve as
    # misses, an unpublished prefetch is cut, and no new prefetch is issued.
    op4 = _operation("op4")
    _track(cache, op4)
    cache.check_hicache_events()
    assert cache.storage_unavailable
    revoked = [
        call.args[0].rid for call in cache.revoke_pending_prefetch.call_args_list
    ]
    assert revoked == ["op1", "op2", "op3"], revoked
    assert cache.check_prefetch_progress(op4.handle)
    assert op4.is_terminated()
    assert cache.revoke_pending_prefetch.call_args.args[0] == op4.handle
    assert cache.prefetch_from_storage(CacheRequestHandle("op5", 0), 0, TOKENS) is None
    assert cc.prefetch_queue.empty()
    _same_on_every_rank(
        (
            revoked,
            cache._prefetch_outcome_stats["revoked_full_miss"],
            cache._prefetch_outcome_stats["l3_miss_tokens"],
        )
    )

    # The raise policy fails the scheduler loudly on the same evidence.
    cc.storage_failure_policy = "raise"
    loud = _scheduler_cache(cc)
    try:
        loud.check_hicache_events()
    except RuntimeError as e:
        assert "unhealthy" in str(e), e
    else:
        raise AssertionError("raise policy did not fail the drain")
    _stop(cc, query_thread, sync_thread)


def _ack_alignment_body(rank: int) -> None:
    """A completion-group timeout mid-transfer: one ack per batch still
    reaches the scheduler on every rank, in order; progress freezes at the
    last agreed value and each completion finishes its own operation."""
    completion_group, hits_group = _sync_groups()
    cc = _controller(hits_group, completion_group, _HoldingBackend(hold_first=False))
    query_thread, sync_thread = _start_workers(cc)
    cache = _scheduler_cache(cc)
    op_a = _operation("A", host_indices=torch.arange(3 * BATCH_TOKENS))
    op_b = _operation("B", host_indices=torch.arange(BATCH_TOKENS))
    for operation in (op_a, op_b):
        _track(cache, operation)

    dist.barrier()
    cc.prefetch_sync_queue.put(_ack(op_a, BATCH_TOKENS))
    _wait_for(lambda: cc.ack_prefetch_queue.qsize() == 1, budget_s=FAST_S)
    cache.check_hicache_events()  # healthy round: the agreed progress lands
    assert (op_a.completed_tokens, cache.storage_unavailable) == (BATCH_TOKENS, False)
    if rank == 1:
        time.sleep(HOLD_S)  # this rank's IO thread is stuck for batch 2
    cc.prefetch_sync_queue.put(_ack(op_a, 2 * BATCH_TOKENS - rank * 10))
    cc.prefetch_sync_queue.put(_ack(op_a, 3 * BATCH_TOKENS))
    cc.prefetch_sync_queue.put(_ack(op_a, done=True))
    cc.prefetch_sync_queue.put(_ack(op_b, BATCH_TOKENS))
    cc.prefetch_sync_queue.put(_ack(op_b, done=True))
    acks = _drain(cc.ack_prefetch_queue, want=5, budget_s=FAST_S)
    sequence = [(a.rid, a.completed_tokens, bool(a.completed_req)) for a in acks]
    assert sequence == [
        ("A", 0, False),
        ("A", 0, False),
        ("A", None, True),
        ("B", 0, False),
        ("B", None, True),
    ], sequence
    assert sync_thread.is_alive() and cc.storage_sync_failed
    _same_on_every_rank(sequence)
    # Cross-group: the hits group did not fail, so a query still reduces
    # promptly on both ranks and keeps its agreed value.
    dist.barrier()
    query = _operation("after")
    _track(cache, query)
    t0 = time.monotonic()
    cc.prefetch_queue.put(query)
    hits = _drain(cc.prefetch_hit_queue, want=1, budget_s=FAST_S)
    assert time.monotonic() - t0 < FAST_S / 2
    assert [(op.request_id, op.storage_hit_count) for op in hits] == [
        ("after", len(TOKENS))
    ], hits

    # Degraded round: no progress is applied from the failed acks, each
    # completion finishes its own operation and releases the right tail, and
    # the agreed hit reads as a miss.
    for ack in acks:
        cc.ack_prefetch_queue.put(ack)
    cc.prefetch_hit_queue.put(hits[0])
    cache.check_hicache_events()
    assert cache.storage_unavailable
    assert (op_a.completed_tokens, op_b.completed_tokens) == (BATCH_TOKENS, 0)
    finished = [
        call.args[0].request_id for call in cache._handle_prefetch_result.call_args_list
    ]
    assert finished == ["A", "B"], finished
    released = [len(call.args[0]) for call in cc.append_host_mem_release.call_args_list]
    assert released == [2 * BATCH_TOKENS, BATCH_TOKENS], released
    assert cache.revoke_pending_prefetch.call_args.args[0] == query.handle
    _stop(cc, query_thread, sync_thread)


def _dead_worker_body(rank: int) -> None:
    """One rank's dead worker degrades every rank in the same tick."""
    cc = _controller(None, None, SimpleNamespace(get_stats=lambda: None))
    if rank == 1:
        cc.backup_thread = threading.Thread(target=lambda: None)
        cc.backup_thread.start()
        cc.backup_thread.join()
    cache = _scheduler_cache(cc)
    cache.enable_storage_metrics = True
    cache.storage_metrics_collector = Mock()
    with patch("sglang.srt.mem_cache.unified_radix_cache.logger.error") as log_error:
        cache.check_hicache_events()
        cache.check_hicache_events()
    assert cache.storage_unavailable
    assert log_error.call_count == 1
    reason = log_error.call_args.args[1]
    assert ("exited" in reason) if rank == 1 else ("peer" in reason), reason
    cache.storage_metrics_collector.log_storage_healthy.assert_called_once_with(False)
    assert cache.prefetch_from_storage(CacheRequestHandle("r", 0), 0, TOKENS) is None
    cc.storage_stop_event.set()


def _rank_main(rank: int, rendezvous: str) -> None:
    # A regression that turns a recovery into a hang must fail, not block CI.
    watchdog = threading.Timer(120, lambda: os._exit(86))
    watchdog.daemon = True
    watchdog.start()
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        for body in (_hit_query_timeout_body, _ack_alignment_body, _dead_worker_body):
            body(rank)
            dist.barrier()
    finally:
        dist.destroy_process_group()


class TestStorageSyncRecovery(unittest.TestCase):
    def test_two_rank_recovery_and_degrade(self):
        """A gloo timeout inside a worker thread's MIN (a peer stuck in its
        backend) used to kill that thread on every rank, stranding every later
        L3-hit request under wait_complete; one spawn runs the hit-query,
        ack-alignment and dead-worker cases back to back."""
        with tempfile.TemporaryDirectory(prefix="hicache-sync-test-") as directory:
            torch.multiprocessing.spawn(
                _rank_main, args=(f"{directory}/store",), nprocs=2, join=True
            )

    def test_sync_knobs_are_popped_and_validated(self):
        cc = HiCacheController.__new__(HiCacheController)
        cc.storage_sync_failed, cc.storage_sync_failure = True, "stale"
        extra = {"storage_failure_policy": "raise", "storage_sync_timeout": 30, "x": 1}
        cc._apply_storage_sync_config(extra)
        self.assertEqual(extra, {"x": 1})
        self.assertEqual(
            (cc.storage_failure_policy, cc.storage_sync_timeout), ("raise", 30.0)
        )
        self.assertEqual(
            (cc.storage_sync_failed, cc.storage_sync_failure), (False, None)
        )
        cc._apply_storage_sync_config({})
        self.assertEqual(
            (cc.storage_failure_policy, cc.storage_sync_timeout), ("disable", None)
        )
        for bad in (
            {"storage_failure_policy": "ignore"},
            {"storage_sync_timeout": 0},
            {"storage_sync_timeout": "30"},
            {"storage_sync_timeout": True},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                cc._apply_storage_sync_config(bad)

        # The timeout reaches the gloo groups the controller creates.
        cc.attn_cp_group = cc.attn_tp_group = cc.pp_group = None
        cc.tp_group = object()
        cc.storage_sync_timeout = 12.5
        with (
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(
                torch.distributed, "get_process_group_ranks", return_value=[0, 1]
            ),
            patch(
                "sglang.srt.distributed.parallel_state.create_custom_parallel_group"
            ) as create_group,
        ):
            cc._create_sync_groups()
        create_group.assert_called_once_with(
            group_ranks=[0, 1], backend="gloo", timeout=timedelta(seconds=12.5)
        )


if __name__ == "__main__":
    unittest.main()
