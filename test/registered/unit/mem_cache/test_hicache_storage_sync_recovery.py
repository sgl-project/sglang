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

register_cpu_ci(est_time=50, suite="base-a-test-cpu")

PAGE = 2
TOKENS = list(range(4))
# The storage sync groups get a short timeout; the hung rank releases its
# backend call only after every peer collective on them has timed out.
SYNC_TIMEOUT_S = 3.0
HOLD_S = 6.0


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
    cc.prefetch_hits_sync_groups = [hits_group] if hits_group is not None else []
    cc.prefetch_completion_sync_groups = (
        [completion_group] if completion_group is not None else []
    )
    cc.storage_sync_failed = False
    cc.storage_sync_failure = None
    cc._storage_sync_lock = threading.Lock()
    cc.storage_failure_policy = "disable"
    cc.pp_prefetch_command_thread = None
    cc.pp_prefetch_decisions = {}
    # Workers not under test idle until the stop event.
    cc.prefetch_thread = _idle_thread(cc.storage_stop_event)
    cc.prefetch_io_aux_thread = _idle_thread(cc.storage_stop_event)
    cc.prefetch_sync_thread = _idle_thread(cc.storage_stop_event)
    cc.backup_thread = _idle_thread(cc.storage_stop_event)
    return cc


def _scheduler_cache(cc) -> UnifiedRadixCache:
    """The scheduler-thread side: a real queue drain whose cross-rank MIN runs
    on the default (healthy) process group, as the TP group does."""
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.cache_controller = cc
    # enable_storage and page_size are proxied to the tree core.
    cache.tree_core = SimpleNamespace(enable_storage=True, page_size=PAGE)
    cache.storage_unavailable = False
    cache.host_memory_mode = "buffer_only"
    cache.prefetch_threshold = PAGE
    cache.enable_storage_metrics = False
    cache.storage_metrics_collector = None
    cache.pp_rank = 0
    cache.pp_size = 1
    cache.pp_group = None
    cache.tp_group = None
    cache.tp_world_size = dist.get_world_size()
    cache.attn_cp_group = None
    cache.attn_tp_group = None
    cache.work_list = []
    cache.storage_prefetch_retries = StoragePrefetchRetries()
    cache._prefetch_outcome_stats = defaultdict(int)
    cache._storage_prefetch_hit_remaining_by_reqid = {}
    cache._record_storage_prefetch_hit = Mock()
    cache.revoke_pending_prefetch = Mock()
    cache._handle_prefetch_result = Mock()
    cache.buffer_pipeline = SimpleNamespace(
        pending_hit_allocs=deque(), has_staged=lambda request: False
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


def _drain(queue: Queue, want: int, budget_s: float) -> list:
    items = []
    deadline = time.monotonic() + budget_s
    while len(items) < want and time.monotonic() < deadline:
        try:
            items.append(queue.get(timeout=0.1))
        except Empty:
            continue
    return items


def _same_on_every_rank(snapshot):
    snapshots = [None] * dist.get_world_size()
    dist.all_gather_object(snapshots, snapshot)
    assert all(other == snapshots[0] for other in snapshots), snapshots
    return snapshots[0]


def _rank_main(rank: int, rendezvous: str, body) -> None:
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
        body(rank)
    finally:
        dist.destroy_process_group()


def _hit_query_timeout_body(rank: int) -> None:
    hits_group = dist.new_group(
        backend="gloo", timeout=timedelta(seconds=SYNC_TIMEOUT_S)
    )
    backend = _HoldingBackend(hold_first=(rank == 1))
    cc = _controller(hits_group, None, backend)
    get_storage_hash_str(TOKENS, None, page_size=PAGE)  # JIT warm-up off the clock
    worker = threading.Thread(target=cc.prefetch_thread_func, daemon=True)
    worker.start()
    cc.prefetch_thread = worker
    cache = _scheduler_cache(cc)
    dist.barrier()

    ops = [_operation("op1"), _operation("op2")]
    for operation in ops:
        _track(cache, operation)
        cc.prefetch_queue.put(operation)
    if rank == 1:
        time.sleep(HOLD_S)
        backend.hold.set()
    published = _drain(cc.prefetch_hit_queue, want=2, budget_s=HOLD_S + 20)
    # The hung rank's own collective fails once the peer's has timed out, so
    # op 1 is a published miss on both ranks and the query thread is alive to
    # serve op 2 (also a miss: the groups stay broken) without any collective.
    assert [(op.request_id, op.storage_hit_count) for op in published] == [
        ("op1", 0),
        ("op2", 0),
    ], published
    assert worker.is_alive()
    assert cc.storage_sync_failed and not cc.storage_workers_healthy()
    op3 = _operation("op3")
    _track(cache, op3)
    cc.prefetch_queue.put(op3)
    assert [op.request_id for op in _drain(cc.prefetch_hit_queue, 1, 5)] == ["op3"]
    for operation in published + [op3]:
        cc.prefetch_hit_queue.put(operation)

    # Scheduler side: the MIN-reduced health bit degrades every rank in the
    # same drain round; the drained queries resolve as misses, an unpublished
    # prefetch is cut, and no new prefetch is issued.
    op4 = _operation("op4")
    _track(cache, op4)
    cache.drain_storage_control_queues()
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
        loud.drain_storage_control_queues()
    except RuntimeError as e:
        assert "unhealthy" in str(e), e
    else:
        raise AssertionError("raise policy did not fail the drain")
    cc.storage_stop_event.set()
    cc.prefetch_queue.put(None)
    worker.join(5)


def _ack_alignment_body(rank: int) -> None:
    completion_group = dist.new_group(
        backend="gloo", timeout=timedelta(seconds=SYNC_TIMEOUT_S)
    )
    cc = _controller(None, completion_group, None)
    worker = threading.Thread(target=cc.prefetch_sync_thread_func, daemon=True)
    worker.start()
    cc.prefetch_sync_thread = worker
    op_a = _operation("A", host_indices=torch.arange(3 * PAGE * 64))
    op_b = _operation("B", host_indices=torch.arange(PAGE * 64))
    tokens_per_batch = PAGE * 64

    def ack(operation, completed=None, done=False):
        return PrefetchAck(
            rid=operation.request_id,
            operation=operation,
            completed_tokens=completed,
            completed_req=True if done else None,
        )

    dist.barrier()
    cc.prefetch_sync_queue.put(ack(op_a, tokens_per_batch))
    if rank == 1:
        # The IO thread of this rank is stuck in its backend for batch 2.
        time.sleep(HOLD_S)
    cc.prefetch_sync_queue.put(ack(op_a, 2 * tokens_per_batch - rank * 10))
    cc.prefetch_sync_queue.put(ack(op_a, 3 * tokens_per_batch))
    cc.prefetch_sync_queue.put(ack(op_a, done=True))
    cc.prefetch_sync_queue.put(ack(op_b, tokens_per_batch))
    cc.prefetch_sync_queue.put(ack(op_b, done=True))
    acks = _drain(cc.ack_prefetch_queue, want=6, budget_s=HOLD_S + 20)
    sequence = [(a.rid, a.completed_tokens, bool(a.completed_req)) for a in acks]
    # One ack per transfer batch reaches the scheduler on every rank, in
    # order; the batches after the failed collective carry no progress.
    assert sequence == [
        ("A", tokens_per_batch, False),
        ("A", 0, False),
        ("A", 0, False),
        ("A", None, True),
        ("B", 0, False),
        ("B", None, True),
    ], sequence
    assert worker.is_alive() and cc.storage_sync_failed
    _same_on_every_rank(sequence)

    # Scheduler side: the first ack was drained while healthy; from the
    # degrade round on, progress freezes at the last agreed value and each
    # completion still finishes its own operation and releases the tail.
    cache = _scheduler_cache(cc)
    cc.append_host_mem_release = Mock()
    for operation in (op_a, op_b):
        _track(cache, operation)
    for ack_ in acks:
        cc.ack_prefetch_queue.put(ack_)
    drain = lambda n: cache._drain_storage_control_queues_impl(
        n_storage_hit=0,
        n_ack_prefetch=n,
        n_backup=0,
        n_release=0,
        extra_release_counts={},
        log_metrics=False,
    )
    drain(1)
    assert op_a.completed_tokens == tokens_per_batch
    cache.storage_unavailable = True
    drain(5)
    assert (op_a.completed_tokens, op_b.completed_tokens) == (tokens_per_batch, 0)
    finished = [
        call.args[0].request_id for call in cache._handle_prefetch_result.call_args_list
    ]
    assert finished == ["A", "B"], finished
    released = [len(call.args[0]) for call in cc.append_host_mem_release.call_args_list]
    assert released == [2 * tokens_per_batch, tokens_per_batch], released
    cc.storage_stop_event.set()
    cc.prefetch_sync_queue.put(None)
    worker.join(5)


def _dead_worker_body(rank: int) -> None:
    cc = _controller(None, None, None)
    if rank == 1:
        cc.backup_thread = threading.Thread(target=lambda: None)
        cc.backup_thread.start()
        cc.backup_thread.join()
    cache = _scheduler_cache(cc)
    cache.enable_storage_metrics = True
    cache.storage_metrics_collector = Mock()
    with patch("sglang.srt.mem_cache.unified_radix_cache.logger.error") as log_error:
        cache.drain_storage_control_queues()
        cache.drain_storage_control_queues()
    # Both ranks degrade in the same round, from one rank's dead worker; the
    # healthy rank attributes it to a peer, and the gauge drops once.
    assert cache.storage_unavailable
    assert log_error.call_count == 1
    reason = log_error.call_args.args[1]
    assert ("exited" in reason) if rank == 1 else ("peer" in reason), reason
    cache.storage_metrics_collector.log_storage_healthy.assert_called_once_with(False)
    assert cache.prefetch_from_storage(CacheRequestHandle("r", 0), 0, TOKENS) is None
    cc.storage_stop_event.set()


class TestStorageSyncRecovery(unittest.TestCase):
    def _spawn(self, body):
        with tempfile.TemporaryDirectory(prefix="hicache-sync-test-") as directory:
            torch.multiprocessing.spawn(
                _rank_main, args=(f"{directory}/store", body), nprocs=2, join=True
            )

    def test_hit_query_survives_sync_timeout_and_degrades(self):
        """A gloo timeout inside the query thread's MIN (a peer stuck in its
        backend) used to kill the query thread on every rank, stranding every
        later L3-hit request under wait_complete."""
        self._spawn(_hit_query_timeout_body)

    def test_ack_sequence_stays_aligned_across_sync_failure(self):
        self._spawn(_ack_alignment_body)

    def test_one_dead_worker_degrades_every_rank(self):
        self._spawn(_dead_worker_body)

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
