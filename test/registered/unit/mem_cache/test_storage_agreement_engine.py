"""Scheduler-thread agreement of HiCache storage hit queries.

The query worker records a local verdict on the operation; one MIN per
scheduler tick agrees the verdicts across ranks and publishes them in
submission order. A slow or dead backend on one rank must never strand the
peers in a worker-thread collective, and every rank must publish the same
operation with the same agreed hit in the same tick.
"""

import tempfile
import threading
import time
import unittest
from array import array
from collections import defaultdict
from datetime import timedelta
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

import torch
import torch.distributed as dist

from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
    PrefetchOperation,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage_prefetch import StoragePrefetchRetries
from sglang.srt.mem_cache.unified_cache.storage_agreement import (
    QUERY_DONE,
    QUERY_PENDING,
    LocalQueryOutcome,
    OperationRowSource,
    StorageAgreementEngine,
    attention_group_min,
)
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedRadixCache,
    _OngoingPrefetch,
)
from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

PAGE = 2
TOKENS = list(range(8))  # four pages
CHAIN = ["h0", "h1", "h2", "h3"]
# Rank 1's backend holds its first query this long; the peer's scheduler
# must keep ticking meanwhile, each tick far quicker than the hold.
HOLD_S = 1.0
TICK_BUDGET_S = 0.3


def _operation(rid: str, pool_transfers=None) -> PrefetchOperation:
    operation = PrefetchOperation(
        CacheRequestHandle(rid, 0), list(TOKENS), pool_transfers=pool_transfers
    )
    operation.all_hash_values = list(CHAIN)
    return operation


def _trailing_operation(rid: str) -> PrefetchOperation:
    return _operation(
        rid,
        pool_transfers=[
            PoolTransfer(
                name=PoolName.SWA,
                keys=list(CHAIN),
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            )
        ],
    )


class _Rank:
    """One rank's engine plus the operations it would put in its rows; peers
    never tick, they only contribute rows to the rank under test."""

    def __init__(self, alive=True, stopped=False):
        self.alive = alive
        self.stopped = stopped
        self.peers: list[_Rank] = []
        self.engine = StorageAgreementEngine(
            PAGE,
            self._agree,
            lambda: self.alive,
            backend_stopped=lambda: self.stopped,
        )
        self.published: list = []

    def _agree(self, rows: list[list[int]]) -> list[list[int]]:
        agreed = [list(row) for row in rows]
        for peer in self.peers:
            peer_rows = [
                peer.engine._query_row(
                    operation, peer.alive, not peer.alive and peer.stopped
                )
                for operation in peer.engine.queries
            ]
            assert [len(r) for r in peer_rows] == [len(r) for r in rows]
            agreed = [
                [min(a, b) for a, b in zip(mine, theirs)]
                for mine, theirs in zip(agreed, peer_rows)
            ]
        return agreed

    def tick(self) -> int:
        return self.engine.tick(self.published.append)


def _ranks(n=2, alive=(True, True), stopped=(False, False)) -> list[_Rank]:
    ranks = [_Rank(alive=alive[i], stopped=stopped[i]) for i in range(n)]
    for rank in ranks:
        rank.peers = [other for other in ranks if other is not rank]
    return ranks


class TestStorageAgreementEngine(unittest.TestCase):
    def test_publication_is_fifo_behind_a_pending_head(self):
        """A rank still querying the head blocks later queries on every rank,
        even when those are done everywhere; the order is submission order."""
        local, peer = _ranks()
        ops = {
            rank: [_operation("a"), _operation("b"), _operation("c")]
            for rank in (local, peer)
        }
        for rank, (a, b, c) in ops.items():
            for operation in (a, b, c):
                rank.engine.register_query(operation)
            a.publish_local_query(4, {PoolName.KV: 4}, None)
            c.publish_local_query(4, {PoolName.KV: 4}, None)
        ops[local][1].publish_local_query(
            3, {PoolName.KV: 3}, None
        )  # the peer's b is pending

        self.assertEqual(local.tick(), 1)
        self.assertEqual([op.request_id for op in local.published], ["a"])
        self.assertEqual(local.engine.pending_queries(), 2)

        ops[peer][1].publish_local_query(2, {PoolName.KV: 2}, None)
        peer.engine.queries.popleft()  # the peer published a in the same tick
        self.assertEqual(local.tick(), 2)
        self.assertEqual([op.request_id for op in local.published], ["a", "b", "c"])
        b = ops[local][1]
        self.assertEqual(b.storage_hit_count, 2 * PAGE)
        self.assertEqual(b.hash_value, CHAIN[:2])
        self.assertEqual(local.engine.pending_queries(), 0)

    def test_shorter_peer_hit_and_missing_verdict_win(self):
        local, peer = _ranks()
        mine, theirs = _operation("a"), _operation("a")
        local.engine.register_query(mine)
        peer.engine.register_query(theirs)
        mine.publish_local_query(4, {PoolName.KV: 4, PoolName.SWA: 4}, None)
        theirs.publish_local_query(2, {PoolName.KV: 2}, None)  # no SWA verdict there

        self.assertEqual(local.tick(), 1)
        self.assertEqual(mine.storage_hit_count, 2 * PAGE)
        self.assertEqual(mine.hash_value, CHAIN[:2])
        # The pool one rank gave no verdict for heals from the folded cut only.
        self.assertEqual(mine.query_pool_hit_pages, {PoolName.KV: 2})

    def test_failed_query_is_a_miss_without_verdicts(self):
        """A query that raised on one rank votes 0 with no pool verdict, so no
        belief learns absence from a backend failure."""
        local, peer = _ranks()
        mine, theirs = _operation("a"), _operation("a")
        local.engine.register_query(mine)
        peer.engine.register_query(theirs)
        mine.publish_local_query(4, {PoolName.KV: 4}, None)
        theirs.publish_local_query(0, {}, None)

        self.assertEqual(local.tick(), 1)
        self.assertEqual(mine.storage_hit_count, 0)
        self.assertEqual(mine.hash_value, [])
        self.assertEqual(mine.query_pool_hit_pages, {})

    def test_trailing_pool_agrees_the_longest_common_restorable_prefix(self):
        """Trailing state can exist at 2 and 4 pages but not at 3: the agreed
        hit is the longest prefix every rank can restore, not MIN of the hits."""
        local, peer = _ranks()
        mine, theirs = _trailing_operation("a"), _trailing_operation("a")
        local.engine.register_query(mine)
        peer.engine.register_query(theirs)
        mine.publish_local_query(4, {PoolName.KV: 4, PoolName.SWA: 4}, (2, 4))
        theirs.publish_local_query(3, {PoolName.KV: 3, PoolName.SWA: 3}, (1, 2, 3))

        self.assertEqual(local.tick(), 1)
        self.assertEqual(mine.storage_hit_count, 2 * PAGE)
        self.assertEqual(mine.hash_value, CHAIN[:2])
        self.assertEqual(mine.query_pool_hit_pages, {PoolName.KV: 3, PoolName.SWA: 3})

    def test_cancelled_query_waits_for_its_worker_then_agrees_a_miss(self):
        """A cancelled operation is not published until its worker recorded a
        local outcome (the worker may still be inside the backend call and
        writing the operation), so the agreed fields have one writer. The rank
        that skipped the query votes a miss, which wins the MIN."""
        local, peer = _ranks()
        mine, theirs = _operation("a"), _operation("a")
        local.engine.register_query(mine)
        peer.engine.register_query(theirs)
        mine.mark_terminate()
        self.assertEqual(mine.query_status(), (QUERY_PENDING, None))
        self.assertEqual(local.tick(), 0)

        mine.publish_local_query(0, {}, None)  # the worker skipped it
        theirs.publish_local_query(4, {PoolName.KV: 4}, None)  # answered late
        self.assertEqual(local.tick(), 1)
        self.assertEqual(mine.storage_hit_count, 0)
        self.assertEqual(mine.hash_value, [])
        self.assertEqual(mine.query_pool_hit_pages, {})

    def test_dead_worker_raises_on_every_rank(self):
        dead, healthy = _ranks(alive=(False, True))
        for rank in (dead, healthy):
            operation = _operation("a")
            operation.publish_local_query(4, {PoolName.KV: 4}, None)
            rank.engine.register_query(operation)

        with self.assertRaisesRegex(RuntimeError, "query worker is gone"):
            dead.tick()
        with self.assertRaisesRegex(RuntimeError, "query worker is gone"):
            healthy.tick()

    def test_stopped_storage_answers_pending_queries_with_misses(self):
        """Storage stopped on purpose (a detach, finished or not) on one rank:
        its pending queries vote misses instead of killing every scheduler."""
        # Peers never tick in this fixture, so each rank's view gets its own pair.
        for ticking in (0, 1):
            ranks = _ranks(alive=(False, True), stopped=(True, False))
            for rank in ranks:
                rank.engine.register_query(_operation("a"))
            healthy = ranks[1]
            list(healthy.engine.queries)[0].publish_local_query(
                4, {PoolName.KV: 4}, None
            )
            self.assertEqual(ranks[ticking].tick(), 1)
            (operation,) = ranks[ticking].published
            self.assertEqual(operation.storage_hit_count, 0)
            self.assertEqual(operation.query_pool_hit_pages, {})

    def test_idle_engine_runs_no_collective(self):
        reduce = Mock()
        engine = StorageAgreementEngine(PAGE, reduce, lambda: True)
        self.assertEqual(engine.tick(Mock()), 0)
        reduce.assert_not_called()

    def test_native_backend_supplies_its_own_row_source(self):
        """A native backend's operation has neither the mixin nor token ids: its
        adapter reads the row off the native state and reports pool boundaries
        only, so a trailing-state pool adds no restorable flags."""

        class _Native:
            request_id = "n"
            pool_transfers = _trailing_operation("n").pool_transfers
            all_hash_values = list(CHAIN)
            hit = None

        class _NativeSource(OperationRowSource):
            def query_status(self, operation):
                if operation.hit is None:
                    return QUERY_PENDING, None
                return QUERY_DONE, LocalQueryOutcome(
                    operation.hit, {PoolName.KV: operation.hit, PoolName.SWA: 2}, None
                )

            def trailing_flag_width(self, operation):
                return 0

        rows_seen = []

        def agree(rows):
            rows_seen.append([list(r) for r in rows])
            return rows

        native = _Native()
        engine = StorageAgreementEngine(
            PAGE, agree, lambda: True, source=_NativeSource(PAGE)
        )
        engine.register_query(native)
        self.assertEqual(engine.tick(Mock()), 0)
        native.hit = 4
        published = []
        self.assertEqual(engine.tick(published.append), 1)
        self.assertEqual(published, [native])
        self.assertEqual(len(rows_seen[-1][0]), 2 + len(list(PoolName)))
        self.assertEqual(native.storage_hit_count, 4 * PAGE)
        self.assertEqual(native.query_pool_hit_pages, {PoolName.KV: 4, PoolName.SWA: 2})


# ---------------------------------------------------------------- two ranks


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


def _controller(backend) -> HybridCacheController:
    """The real query worker over a scripted backend; agreement is the
    scheduler's, so the worker only records local verdicts."""
    cc = HybridCacheController.__new__(HybridCacheController)
    cc.page_size = PAGE
    cc.storage_backend = backend
    cc.scheduler_agrees_queries = True
    cc.storage_stop_event = threading.Event()
    cc.prefetch_queue = Queue()
    cc.prefetch_hit_queue = Queue()
    cc.ack_prefetch_queue = Queue()
    cc.ack_backup_queue = Queue()
    cc.host_mem_release_queue = Queue()
    cc.extra_host_mem_release_queues = {}
    cc.ack_write_queue = []
    cc.ack_load_queue = []
    cc.start_writing = Mock()
    cc.prefetch_thread = threading.Thread(target=cc.prefetch_thread_func, daemon=True)
    cc.prefetch_thread.start()
    return cc


def _scheduler_cache(cc) -> UnifiedRadixCache:
    """The scheduler-thread side: the real check_hicache_events round with the
    real count exchange and agreement tick; the drain itself is observed."""
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.cache_controller = cc
    cache.tree_core = SimpleNamespace(
        enable_storage=True, page_size=PAGE, write_back_duplicate_reclaim_digest=0
    )
    cache.host_memory_mode = "cache"
    cache.buffer_pipeline = None
    cache.enable_storage_metrics = False
    cache.storage_metrics_collector = None
    cache.linker = None
    cache.pp_rank, cache.pp_size, cache.pp_group = 0, 1, None
    cache.tp_group = dist.group.WORLD
    cache.tp_world_size = dist.get_world_size()
    cache.attn_cp_group = cache.attn_tp_group = None
    cache.work_list = []
    cache.writing_check = Mock()
    cache.loading_check = Mock()
    cache._drain_storage_control_queues_impl = Mock()
    cache.storage_prefetch_retries = StoragePrefetchRetries()
    cache._storage_prefetch_wait = {}
    cache._prefetch_outcome_stats = defaultdict(int)
    cache.ongoing_prefetch = {}
    cache.storage_engine = StorageAgreementEngine(
        PAGE,
        attention_group_min(cache._reduce_min_attn_groups),
        cache._storage_query_worker_alive,
        backend_stopped=cache._storage_query_worker_stopped,
    )
    return cache


def _submit(cache, rid: str, enqueue: bool = True) -> PrefetchOperation:
    """What prefetch_from_storage does for the query: track, register, enqueue."""
    operation = PrefetchOperation(CacheRequestHandle(rid, 0), list(TOKENS))
    cache.ongoing_prefetch[operation.handle] = _OngoingPrefetch(
        0, RadixKey(array("q", TOKENS)), None, operation, None, {}
    )
    cache.storage_engine.register_query(operation)
    if enqueue:
        cache.cache_controller.prefetch_queue.put(operation)
    return operation


def _same_on_every_rank(snapshot):
    snapshots = [None] * dist.get_world_size()
    dist.all_gather_object(snapshots, snapshot)
    assert all(other == snapshots[0] for other in snapshots), snapshots
    return snapshots[0]


def _published_hits(cache) -> int:
    return cache._drain_storage_control_queues_impl.call_args.kwargs["n_storage_hit"]


def _slow_backend_body(rank: int) -> None:
    """Rank 1's first query blocks in its backend. Rank 0's scheduler keeps
    ticking (nothing waits on a worker), nothing is published until the hold
    lifts, and then both ranks publish the same agreed hit in the same tick."""
    backend = _HoldingBackend(hold_first=(rank == 1))
    cc = _controller(backend)
    cache = _scheduler_cache(cc)
    get_storage_hash_str(TOKENS, None, page_size=PAGE)  # JIT warm-up off the clock
    operation = _submit(cache, "slow")
    if rank == 1:
        threading.Timer(HOLD_S, backend.hold.set).start()

    published_at = None
    tick_times = []
    start = time.monotonic()
    for tick in range(400):
        t0 = time.monotonic()
        cache.check_hicache_events()
        tick_times.append(time.monotonic() - t0)
        if _published_hits(cache):
            published_at = tick
            break
        time.sleep(0.02)
    assert published_at is not None, "the agreed hit was never published"
    # Publication waited for the held query (HOLD_S of wall clock on both
    # ranks), yet no tick blocked: the scheduler never waits on a worker.
    assert time.monotonic() - start >= HOLD_S * 0.8, time.monotonic() - start
    assert max(tick_times) < TICK_BUDGET_S, max(tick_times)
    assert published_at >= 3, published_at
    assert cc.prefetch_hit_queue.qsize() == 1
    assert cc.prefetch_hit_queue.get_nowait() is operation
    _same_on_every_rank(
        (published_at, operation.storage_hit_count, operation.hash_value)
    )
    assert operation.storage_hit_count == len(TOKENS)
    assert cc.prefetch_thread.is_alive()
    assert cache.storage_engine.pending_queries() == 0

    # A second query goes through without any special handling.
    second = _submit(cache, "second")
    for _ in range(200):
        cache.check_hicache_events()
        if _published_hits(cache):
            break
        time.sleep(0.02)
    assert cc.prefetch_hit_queue.get_nowait() is second
    _same_on_every_rank((second.storage_hit_count, second.hash_value))

    # Divergent submission: only rank 0 registers a query. The count exchange
    # catches it on every rank before the agreement collective could hang.
    if rank == 0:
        _submit(cache, "only-here")
    raised = False
    try:
        cache.check_hicache_events()
    except RuntimeError as e:
        raised = "diverged" in str(e)
    assert raised, "FIFO divergence was not detected"
    if rank == 0:
        cache.storage_engine.queries.clear()
        cache.ongoing_prefetch.clear()

    # A dead query worker is a fatal row: every rank raises in the same tick.
    dist.barrier()
    gone = threading.Thread(target=lambda: None)
    gone.start()
    gone.join()
    cc.prefetch_thread = gone
    third = _submit(cache, "third")
    third.publish_local_query(4, {PoolName.KV: 4}, None)
    raised = False
    try:
        cache.check_hicache_events()
    except RuntimeError as e:
        raised = "query worker is gone" in str(e)
    assert raised, "dead worker was not reported"
    _same_on_every_rank(raised)

    # Storage stopped on purpose: the same gone worker now means "no backend".
    # A query that was answered keeps its verdict; one that no worker will
    # answer (never enqueued here) is published as an agreed miss, no raise.
    cc.storage_stop_event.set()
    fourth = _submit(cache, "fourth", enqueue=False)
    cache.check_hicache_events()
    assert _published_hits(cache) == 2
    assert cc.prefetch_hit_queue.get_nowait() is third
    assert cc.prefetch_hit_queue.get_nowait() is fourth
    _same_on_every_rank((third.storage_hit_count, fourth.storage_hit_count))
    assert third.storage_hit_count == len(TOKENS) and fourth.storage_hit_count == 0
    cc.prefetch_queue.put(None)


def _rank_main(rank: int, rendezvous: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        _slow_backend_body(rank)
    finally:
        dist.destroy_process_group()


class TestStorageAgreementTwoRanks(unittest.TestCase):
    def test_slow_backend_on_one_rank_never_strands_the_peer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            torch.multiprocessing.spawn(
                _rank_main, args=(f"{tmpdir}/gloo",), nprocs=2, join=True
            )


if __name__ == "__main__":
    unittest.main()
