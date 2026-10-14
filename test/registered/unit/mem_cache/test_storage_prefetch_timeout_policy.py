"""The `timeout` storage-prefetch stop policy on the Python path: one rank-agreed
vote per scheduler tick, a budget that runs from the first blocked admission
poll, no re-issue after a cut, and a cut that leaves storage beliefs alone.
Buffer-only mode with a mocked tree core; no GPU kernels."""

import tempfile
import threading
import time
import unittest
from collections import defaultdict, deque
from contextlib import nullcontext
from datetime import timedelta
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist

from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.buffer_mode.storage_existence_cache import (
    StorageExistenceCache,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.storage_prefetch import StoragePrefetchRetries
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

_PAGE = 2
_TOKENS = list(range(4))  # two pages
_CHAIN = get_storage_hash_str(_TOKENS, None, page_size=_PAGE)
_BASE_S = 1.0
_PER_PAGE_S = 0.5
_BUDGET_S = _BASE_S + len(_CHAIN) * _PER_PAGE_S


class _Backend:
    """Every page is present; the first query can be parked until released."""

    def __init__(self, hold_first=False):
        self.hold_first = hold_first
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def batch_exists(self, hashes, extra_info=None):
        self.calls += 1
        if self.hold_first and self.calls == 1:
            self.entered.set()
            self.release.wait()
        return len(hashes)


class _FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _controller(backend, sync_group=None):
    cc = HybridCacheController.__new__(HybridCacheController)
    cc.page_size = _PAGE
    cc.storage_backend = backend
    cc.storage_stop_event = threading.Event()
    cc.prefetch_queue = Queue()
    cc.prefetch_hit_queue = Queue()
    cc.ack_prefetch_queue = Queue()
    cc.ack_backup_queue = Queue()
    cc.host_mem_release_queue = Queue()
    cc.extra_host_mem_release_queues = {}
    cc.prefetch_hits_sync_groups = [] if sync_group is None else [sync_group]
    cc.pp_prefetch_decisions = {}
    cc.pp_prefetch_command_group = None
    cc.prefetch_tokens_occupied = 0
    cc.prefetch_threshold = _PAGE
    cc.append_host_mem_release = Mock()
    return cc


def _cache(cc, *, world=1, policy="timeout"):
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.tree_core = SimpleNamespace(
        page_size=_PAGE,
        enable_storage=True,
        is_eagle=False,
        prefetch_anchor_info=lambda node: (None, None),
    )
    cache.cache_controller = cc
    cache.host_memory_mode = "buffer_only"
    cache.prefetch_stop_policy = policy
    cache.prefetch_timeout_base = _BASE_S
    cache.prefetch_timeout_per_page = _PER_PAGE_S
    cache.pp_rank = 0
    cache.pp_size = 1
    cache.pp_group = None
    cache.attn_cp_group = None
    cache.attn_tp_group = None
    cache.tp_group = dist.group.WORLD if world > 1 else None
    cache.tp_world_size = world
    cache.work_list = []
    cache.tree_components = []
    cache.components = {}
    cache.prefetch_threshold = _PAGE
    cache._build_sidecar_transfers = Mock(return_value=[])
    cache.buffer_pipeline = SimpleNamespace(
        has_staged=lambda request: False,
        set_prefix_ctx=Mock(),
        pop_prefix_ctx=Mock(),
        release_anchor_lock=Mock(),
        pending_hit_allocs=deque(),
    )
    cache.storage_prefetch_retries = StoragePrefetchRetries()
    cache._storage_prefetch_wait = {}
    cache.storage_existence_cache = StorageExistenceCache()
    cache.ongoing_prefetch = {}
    cache._prefetch_outcome_stats = defaultdict(int)
    cache._storage_prefetch_hit_remaining_by_reqid = {}
    cache.enable_storage_metrics = False
    cache.storage_metrics_collector = None
    cache.prefetch_loaded_tokens_by_reqid = {}
    cache.prefetch_loaded_storage_start_by_reqid = {}
    cache.linker = None
    cache._record_storage_prefetch_hit = Mock()
    # Every scheduler-thread collective the policy issues, by tensor shape.
    collectives = []
    real_all_reduce = cache._all_reduce

    def counted_all_reduce(data, op):
        collectives.append(tuple(data.shape))
        real_all_reduce(data, op)

    cache._all_reduce = counted_all_reduce
    return cache, collectives


def _issue(cache, handle):
    assert (
        cache.prefetch_from_storage(handle, 0, _TOKENS, matched_prefix_tokens=[])
        is None
    )
    return cache.ongoing_prefetch[handle].operation


def _drain_hits(cache, n):
    cache._drain_storage_control_queues_impl(
        n_storage_hit=n,
        n_ack_prefetch=0,
        n_backup=0,
        n_release=0,
        extra_release_counts={},
        log_metrics=False,
    )


def _wait_for(predicate, timeout_s=15.0):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.01)


def _beliefs(cache):
    pool = cache.storage_existence_cache.pool(PoolName.KV)
    return [pool.contains(h) for h in _CHAIN]


def _cut_keeps_beliefs(rank, sync_group, *, held_rank, late_rank):
    """A deadline reached on rank 0 alone cuts the query on both ranks while
    the workers disagree on how far the query got; neither rank heals its
    beliefs from the unsynced per-rank query state."""
    backend = _Backend(hold_first=rank == held_rank)
    cc = _controller(backend, sync_group)
    cache, collectives = _cache(cc, world=2)
    handle = CacheRequestHandle("r", 0)
    operation = _issue(cache, handle)
    cache.storage_existence_cache.pool(PoolName.KV).add(_CHAIN)
    worker = threading.Thread(target=cc.prefetch_thread_func, daemon=True)
    if rank != late_rank:
        worker.start()
        if backend.hold_first:
            assert backend.entered.wait(10)
        else:
            _wait_for(lambda: backend.calls == 1)
            time.sleep(0.1)
    dist.barrier()
    assert not cache.check_prefetch_progress(handle)
    clock = (
        patch("time.monotonic", return_value=time.monotonic() + 1000.0)
        if rank == 0
        else nullcontext()
    )
    with clock:
        cache._retire_lapsed_prefetches()
    assert operation.deadline_expired and handle not in cache.ongoing_prefetch
    assert cache._storage_prefetch_wait[handle].expired
    assert cache.check_prefetch_progress(handle)
    post_cut = _beliefs(cache)
    backend.release.set()
    if rank == late_rank:
        worker.start()
    _wait_for(lambda: cc.prefetch_hit_queue.qsize() == 1)
    _drain_hits(cache, 1)
    post_drain = _beliefs(cache)
    # A later re-issue for the cut attempt (device-hit-loss retry) queries nothing.
    assert (
        cache.prefetch_from_storage(handle, 0, _TOKENS, matched_prefix_tokens=[])
        is None
    )
    assert handle not in cache.ongoing_prefetch
    req = SimpleNamespace(rid="r", storage_prefetch_retry_attempts=0)
    assert cache.storage_prefetch_retries.pop_ready([req], 0, 8) == []
    cc.storage_stop_event.set()
    cc.prefetch_queue.put(None)
    worker.join(10)
    assert not worker.is_alive()
    snapshots = [None, None]
    dist.all_gather_object(
        snapshots,
        (post_cut, post_drain, cache._prefetch_outcome_stats["issued"], collectives),
    )
    assert snapshots[0] == snapshots[1] == ([True, True], [True, True], 1, [(1,)]), (
        rank,
        snapshots,
    )


def _divergent_polls_agree(rank):
    """Rank 1 leaves the admission loop (batch_is_full) before polling the
    second request; the per-tick vote still retires both attempts on both
    ranks with one collective each, and both ranks suppress its re-issue."""
    cc = _controller(_Backend())
    cache, collectives = _cache(cc, world=2)
    first = CacheRequestHandle("a", 0)
    second = CacheRequestHandle("b", 0)
    _issue(cache, first)
    _issue(cache, second)
    for handle in [first] if rank == 1 else [first, second]:
        assert not cache.check_prefetch_progress(handle)
    with patch("time.monotonic", return_value=time.monotonic() + 1000.0):
        cache._retire_lapsed_prefetches()
    assert (
        cache.prefetch_from_storage(second, 0, _TOKENS, matched_prefix_tokens=[])
        is None
    )
    snapshot = (
        sorted(cache.ongoing_prefetch),
        sorted(h.rid for h, w in cache._storage_prefetch_wait.items() if w.expired),
        cache._prefetch_outcome_stats["issued"],
        collectives,
    )
    snapshots = [None, None]
    dist.all_gather_object(snapshots, snapshot)
    assert snapshots[0] == snapshots[1] == ([], ["a", "b"], 2, [(2,)]), (
        rank,
        snapshots,
    )


def _two_rank_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        # The worker-thread hit sync uses its own group, as in production.
        sync_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=60))
        # Rank 1's backend is still answering while rank 0 waits in the MIN sync.
        _cut_keeps_beliefs(rank, sync_group, held_rank=1, late_rank=None)
        # Rank 0's backend is answering; rank 1's worker has not dequeued yet.
        _cut_keeps_beliefs(rank, sync_group, held_rank=0, late_rank=1)
        _divergent_polls_agree(rank)
    finally:
        dist.destroy_process_group()


class TestStoragePrefetchTimeoutPolicy(CustomTestCase):
    def test_two_rank_cuts_agree_and_keep_beliefs(self):
        with tempfile.TemporaryDirectory(prefix="prefetch-timeout-") as directory:
            torch.multiprocessing.spawn(
                _two_rank_worker, args=(f"{directory}/store",), nprocs=2, join=True
            )

    def test_budget_runs_from_first_blocked_poll_and_survives_reissue(self):
        """A queued attempt spends no budget until admission first blocks on
        it; a store-miss re-issue keeps that budget; the cut arms no paced
        re-query, is counted as `deadline`, and admission releases it."""
        clock = _FakeClock()
        with patch("time.monotonic", clock):
            cc = _controller(_Backend())
            cache, collectives = _cache(cc)
            handle = CacheRequestHandle("r", 0)
            req = SimpleNamespace(rid="r", storage_prefetch_retry_attempts=0)
            waiting = [
                SimpleNamespace(rid="head", storage_prefetch_retry_attempts=0),
                req,
            ]
            first = _issue(cache, handle)
            clock.now += 5.0
            cache._retire_lapsed_prefetches()
            self.assertIn(handle, cache.ongoing_prefetch)
            self.assertFalse(cache.check_prefetch_progress(handle))
            deadline = clock.now + _BUDGET_S
            self.assertEqual(cache._storage_prefetch_wait[handle].deadline, deadline)
            # The store misses before the deadline: paced re-issue, same budget.
            cc.prefetch_queue.get_nowait()
            cc.prefetch_hit_queue.put(first)
            _drain_hits(cache, 1)
            self.assertTrue(cache.check_prefetch_progress(handle))
            self.assertEqual(
                cache.storage_prefetch_retries.pop_ready(waiting, 1, 8), []
            )
            self.assertEqual(
                cache.storage_prefetch_retries.pop_ready(waiting, 1, 8), [(req, None)]
            )
            second = _issue(cache, handle)
            self.assertEqual(cache._storage_prefetch_wait[handle].deadline, deadline)
            clock.now = deadline - 0.1
            cache._retire_lapsed_prefetches()
            self.assertFalse(cache.check_prefetch_progress(handle))
            self.assertIn(handle, cache.ongoing_prefetch)
            clock.now = deadline + 0.1
            cache.enable_storage_metrics = True
            collector = cache.storage_metrics_collector = Mock()
            cache._storage_prefetch_hit_remaining_by_reqid[handle] = len(_TOKENS)
            cache._retire_lapsed_prefetches()
            self.assertTrue(second.deadline_expired)
            self.assertNotIn(handle, cache.ongoing_prefetch)
            collector.log_storage_prefetch_unfulfilled_tokens.assert_called_once_with(
                len(_TOKENS), "deadline"
            )
            self.assertTrue(cache.check_prefetch_progress(handle))
            for _ in range(10):
                self.assertEqual(
                    cache.storage_prefetch_retries.pop_ready(waiting, 1, 8), []
                )
            self.assertIsNone(
                cache.prefetch_from_storage(
                    handle, 0, _TOKENS, matched_prefix_tokens=[]
                )
            )
            self.assertEqual(cache._prefetch_outcome_stats["issued"], 2)
            self.assertEqual(collectives, [(1,)] * 3)
            cache.finish_storage_prefetch_admission(handle, 0, None)
            self.assertEqual(cache._storage_prefetch_wait, {})
            # The request's next attempt starts with a budget of its own.
            _issue(cache, CacheRequestHandle("r", 1))
            self.assertEqual(cache._prefetch_outcome_stats["issued"], 3)

    def test_transfer_cut_keeps_pool_beliefs_and_counts_deadline(self):
        """A hybrid transfer cut mid-way is discarded all-or-nothing like a
        shortfall, but the store still holds the span: no belief is dropped
        and the loss is labelled `deadline`."""
        cc = _controller(_Backend())
        cache, _ = _cache(cc)
        cache.enable_storage_metrics = True
        collector = cache.storage_metrics_collector = Mock()
        handle = CacheRequestHandle("r", 0)
        operation = _issue(cache, handle)
        window = PoolTransfer(
            name=PoolName.SWA,
            keys=["s0", "s1"],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
            host_indices=torch.arange(4),
        )
        operation.pool_transfers = [window]
        operation.pool_transfers_done = False
        operation.hash_value = list(_CHAIN)
        operation.storage_hit_count = len(_TOKENS)
        operation.host_indices = torch.arange(len(_TOKENS))
        operation.completed_tokens = _PAGE
        info = cache.ongoing_prefetch[handle]
        cache.ongoing_prefetch[handle] = info._replace(
            host_indices=operation.host_indices
        )
        beliefs = cache.storage_existence_cache
        beliefs.pool(PoolName.KV).add(_CHAIN)
        beliefs.pool(PoolName.SWA).add(window.keys)
        cache._storage_prefetch_hit_remaining_by_reqid[handle] = len(_TOKENS)
        self.assertFalse(cache.check_prefetch_progress(handle))
        with patch("time.monotonic", return_value=time.monotonic() + 1000.0):
            cache._retire_lapsed_prefetches()
        self.assertTrue(operation.deadline_expired)
        self.assertNotIn(handle, cache.ongoing_prefetch)
        collector.log_storage_prefetch_unfulfilled_tokens.assert_called_once_with(
            len(_TOKENS), "deadline"
        )
        self.assertTrue(beliefs.pool(PoolName.KV).contains_all(_CHAIN))
        self.assertTrue(beliefs.pool(PoolName.SWA).contains_all(window.keys))
        self.assertEqual(cache.pop_prefetch_loaded_tokens(handle), 0)

    def test_polls_and_other_policies_issue_no_collective(self):
        """Only the per-tick `timeout` vote may reach a collective: the
        admission polls never do, and the other policies never do."""
        for policy, blocked in (("best_effort", False), ("wait_complete", True)):
            with self.subTest(policy=policy):
                cache, collectives = _cache(_controller(_Backend()), policy=policy)
                handle = CacheRequestHandle("r", 0)
                _issue(cache, handle)
                self.assertEqual(cache.check_prefetch_progress(handle), not blocked)
                cache._retire_lapsed_prefetches()
                self.assertEqual(collectives, [])
        cache, collectives = _cache(_controller(_Backend()))
        handles = [CacheRequestHandle(rid, 0) for rid in ("a", "b", "c")]
        for handle in handles:
            _issue(cache, handle)
            self.assertFalse(cache.check_prefetch_progress(handle))
        self.assertEqual(collectives, [])
        cache._retire_lapsed_prefetches()
        self.assertEqual(collectives, [(3,)])

    def test_check_hicache_events_votes_once_after_the_storage_drain(self):
        cache = Mock()
        cache.linker = None
        cache.enable_storage = True
        cache.buffer_pipeline = None
        cache.enable_storage_metrics = False
        cache._sync_hicache_ready_counts.return_value = (0, 0, (0, 0, 0, 0), ())
        UnifiedRadixCache.check_hicache_events(cache)
        names = [name for name, _, _ in cache.mock_calls]
        self.assertEqual(names.count("_retire_lapsed_prefetches"), 1)
        self.assertEqual(
            names.index("_retire_lapsed_prefetches"),
            names.index("_drain_storage_control_queues_impl") + 1,
        )


if __name__ == "__main__":
    unittest.main()
