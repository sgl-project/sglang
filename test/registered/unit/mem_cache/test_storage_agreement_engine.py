"""Scheduler-thread agreement of HiCache storage operations.

The workers record local verdicts and progress on the operation; one MIN per
scheduler tick agrees them across ranks and publishes them in submission
order. A slow or dead backend on one rank must never strand the peers in a
worker-thread collective, every rank must publish the same operation with the
same agreed values in the same tick, and a deadline that expired on one rank
cuts the operation on every rank.
"""

import tempfile
import threading
import time
import unittest
from array import array
from collections import defaultdict, deque
from contextlib import nullcontext
from datetime import timedelta
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

import torch
import torch.distributed as dist

from sglang.srt.managers.cache_controller import (
    HiCacheController,
    StorageCallLanes,
    StorageCallTimeout,
)
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

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

PAGE = 2
TOKENS = list(range(8))  # four pages
CHAIN = ["h0", "h1", "h2", "h3"]
# Rank 1's backend holds its first call this long; the peer's scheduler must
# keep ticking meanwhile, each tick far quicker than the hold.
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


def _sidecar_operation(rid: str) -> PrefetchOperation:
    return _operation(
        rid,
        pool_transfers=[PoolTransfer(name=PoolName.SWA, keys=list(CHAIN))],
    )


class _Rank:
    """One rank's engine plus the operations it would put in its rows; peers
    never tick, they only contribute rows to the rank under test."""

    def __init__(self, alive=True, stopped=False, source=None):
        self.alive = alive
        self.stopped = stopped
        self.peers: list[_Rank] = []
        self.engine = StorageAgreementEngine(
            PAGE,
            self._agree,
            lambda: self.alive,
            source=source,
            backend_stopped=lambda: self.stopped,
        )
        self.published: list = []
        self.acks: list = []
        self.cuts: list = []

    def _rows(self):
        stopped = not self.alive and self.stopped
        return [
            self.engine._query_row(op, self.alive, stopped)
            for op in self.engine.queries
        ] + [
            self.engine._transfer_row(op, self.alive, stopped)
            for op in self.engine.transfers
        ]

    def _agree(self, rows: list[list[int]]) -> list[list[int]]:
        agreed = [list(row) for row in rows]
        for peer in self.peers:
            peer_rows = peer._rows()
            assert [len(r) for r in peer_rows] == [len(r) for r in rows]
            agreed = [
                [min(a, b) for a, b in zip(mine, theirs)]
                for mine, theirs in zip(agreed, peer_rows)
            ]
        return agreed

    def tick(self) -> tuple[int, int]:
        return self.engine.tick(
            self.published.append,
            lambda op, completed, hits: self.acks.append((op, completed, hits)),
            self.cuts.append,
            self.cuts.append,
        )


def _ranks(
    n=2, alive=(True, True), stopped=(False, False), sources=(None, None)
) -> list[_Rank]:
    ranks = [
        _Rank(alive=alive[i], stopped=stopped[i], source=sources[i]) for i in range(n)
    ]
    for rank in ranks:
        rank.peers = [other for other in ranks if other is not rank]
    return ranks


class TestStorageAgreementQueries(unittest.TestCase):
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

        self.assertEqual(local.tick(), (1, 0))
        self.assertEqual([op.request_id for op in local.published], ["a"])
        self.assertEqual(local.engine.pending_queries(), 2)

        ops[peer][1].publish_local_query(2, {PoolName.KV: 2}, None)
        peer.engine.queries.popleft()  # the peer published a in the same tick
        self.assertEqual(local.tick(), (2, 0))
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

        self.assertEqual(local.tick(), (1, 0))
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

        self.assertEqual(local.tick(), (1, 0))
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

        self.assertEqual(local.tick(), (1, 0))
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
        self.assertEqual(local.tick(), (0, 0))

        mine.publish_local_query(0, {}, None)  # the worker skipped it
        theirs.publish_local_query(4, {PoolName.KV: 4}, None)  # answered late
        self.assertEqual(local.tick(), (1, 0))
        self.assertEqual(mine.storage_hit_count, 0)
        self.assertEqual(mine.hash_value, [])
        self.assertEqual(mine.query_pool_hit_pages, {})

    def test_expired_query_is_cut_on_every_rank(self):
        """The deadline expired on the peer only; MAX of the bit makes this rank
        cut too, in the same tick, while the query itself stays pending."""
        local, peer = _ranks(
            sources=(None, OperationRowSource(PAGE, query_expired=lambda op: True))
        )
        mine, theirs = _operation("a"), _operation("a")
        local.engine.register_query(mine)
        peer.engine.register_query(theirs)

        self.assertEqual(local.tick(), (0, 0))
        self.assertEqual(local.cuts, [mine])
        self.assertEqual(local.engine.pending_queries(), 1)

    def test_dead_worker_raises_on_every_rank(self):
        dead, healthy = _ranks(alive=(False, True))
        for rank in (dead, healthy):
            operation = _operation("a")
            operation.publish_local_query(4, {PoolName.KV: 4}, None)
            rank.engine.register_query(operation)

        with self.assertRaisesRegex(RuntimeError, "worker is gone"):
            dead.tick()
        with self.assertRaisesRegex(RuntimeError, "worker is gone"):
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
            self.assertEqual(ranks[ticking].tick(), (1, 0))
            (operation,) = ranks[ticking].published
            self.assertEqual(operation.storage_hit_count, 0)
            self.assertEqual(operation.query_pool_hit_pages, {})

    def test_idle_engine_runs_no_collective(self):
        reduce = Mock()
        engine = StorageAgreementEngine(PAGE, reduce, lambda: True)
        self.assertEqual(engine.tick(Mock(), Mock()), (0, 0))
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
        self.assertEqual(engine.tick(Mock(), Mock()), (0, 0))
        native.hit = 4
        published = []
        self.assertEqual(engine.tick(published.append, Mock()), (1, 0))
        self.assertEqual(published, [native])
        # state, hit pages, one boundary per pool, the expired bit; no flags.
        self.assertEqual(len(rows_seen[-1][0]), 3 + len(list(PoolName)))
        self.assertEqual(native.storage_hit_count, 4 * PAGE)
        self.assertEqual(native.query_pool_hit_pages, {PoolName.KV: 4, PoolName.SWA: 2})


class TestStorageAgreementTransfers(unittest.TestCase):
    def test_ack_waits_until_every_rank_finished_with_the_rows(self):
        """Progress on one rank means nothing until the other rank's worker is
        done with its host rows; the agreed prefix is the MIN of final progress."""
        local, peer = _ranks()
        mine, theirs = _operation("a"), _operation("a")
        local.engine.register_transfer(mine)
        peer.engine.register_transfer(theirs)
        mine.publish_local_transfer_progress(8)
        mine.mark_transfer_finished()
        theirs.publish_local_transfer_progress(4)

        self.assertEqual(local.tick(), (0, 0))
        self.assertEqual(local.engine.pending_transfers(), 1)

        theirs.publish_local_transfer_progress(6)
        theirs.mark_transfer_finished()
        self.assertEqual(local.tick(), (0, 1))
        self.assertEqual(local.acks, [(mine, 6, None)])
        self.assertEqual(local.engine.pending_transfers(), 0)

    def test_sidecar_hits_are_min_reduced_per_pool(self):
        local, peer = _ranks()
        mine, theirs = _sidecar_operation("a"), _sidecar_operation("a")
        local.engine.register_transfer(mine)
        peer.engine.register_transfer(theirs)
        for operation, hits in ((mine, 4), (theirs, 2)):
            operation.publish_local_transfer_progress(8)
            operation.publish_local_sidecar_hits({PoolName.SWA.value: hits})
            operation.mark_transfer_finished()

        self.assertEqual(local.tick(), (0, 1))
        ((operation, completed, pool_hits),) = local.acks
        self.assertIs(operation, mine)
        self.assertEqual(completed, 8)
        self.assertEqual(pool_hits[PoolName.SWA.value], 2)
        self.assertEqual(pool_hits[PoolName.KV.value], 0)

    def test_expired_transfer_is_cut_on_every_rank_but_published_later(self):
        local, peer = _ranks(
            sources=(None, OperationRowSource(PAGE, transfer_expired=lambda op: True))
        )
        mine, theirs = _operation("a"), _operation("a")
        local.engine.register_transfer(mine)
        peer.engine.register_transfer(theirs)
        mine.publish_local_transfer_progress(4)

        self.assertEqual(local.tick(), (0, 0))
        self.assertEqual(
            local.cuts, [mine]
        )  # cut although this rank's clock had not expired
        self.assertEqual(local.acks, [])

        # The workers stop at their batch boundary; only then is the ack agreed.
        for operation in (mine, theirs):
            operation.mark_transfer_finished()
        theirs.publish_local_transfer_progress(2)
        self.assertEqual(local.tick(), (0, 1))
        self.assertEqual(local.acks, [(mine, 2, None)])

    def test_dead_worker_fails_pending_transfers_too(self):
        dead, healthy = _ranks(alive=(False, True))
        for rank in (dead, healthy):
            operation = _operation("a")
            operation.mark_transfer_finished()
            rank.engine.register_transfer(operation)
        with self.assertRaisesRegex(RuntimeError, "worker is gone"):
            healthy.tick()

    def test_finished_transfer_publishes_past_a_pending_one(self):
        """A transfer whose rows a slow (orphaned) call still holds does not
        hold back later transfers: the agreed state names the operation."""
        local, peer = _ranks()
        for rank in (local, peer):
            slow, fast = _operation("slow"), _operation("fast")
            rank.engine.register_transfer(slow)
            rank.engine.register_transfer(fast)
            fast.publish_local_transfer_progress(8)
            fast.mark_transfer_finished()
        self.assertEqual(local.tick(), (0, 1))
        self.assertEqual(local.acks[0][0].request_id, "fast")
        self.assertEqual([op.request_id for op in local.engine.transfers], ["slow"])

    def test_query_rows_come_before_transfer_rows_in_one_agreement(self):
        """Both FIFOs ride one collective; a pending query does not hold back a
        finished transfer and vice versa."""
        local, peer = _ranks()
        for rank in (local, peer):
            query, transfer = _operation("q"), _operation("t")
            rank.engine.register_query(query)
            rank.engine.register_transfer(transfer)
            transfer.publish_local_transfer_progress(8)
            transfer.mark_transfer_finished()
        self.assertEqual(local.tick(), (0, 1))


class TestAgreedHitReservations(unittest.TestCase):
    """The hit drain reserves locally, agrees the outcomes in one MIN, and only
    then dispatches: a peer's shortfall parks the operation here too, and a
    parked hit keeps its turn."""

    def _cache(self, peer_outcomes):
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.host_memory_mode = "buffer_only"
        cache.prefetch_threshold = PAGE
        cache.enable_storage_metrics = False
        cache.storage_metrics_collector = None
        cache.storage_prefetch_retries = StoragePrefetchRetries()
        cache._storage_prefetch_wait = {}
        cache._prefetch_outcome_stats = defaultdict(int)
        cache._storage_prefetch_hit_remaining_by_reqid = {}
        cache._record_storage_prefetch_hit = Mock()
        cache.revoke_pending_prefetch = Mock()
        cache._invalidate_absent_from_hit_query = Mock()
        cache._account_prefetch_outcome = Mock()
        cache._trim_buffer_prefetch_full_head = (
            lambda request, info, operation, full, hit: (info, hit, hit)
        )
        cache.storage_engine = object()  # agreement on
        dispatched, rolled_back = [], []
        pipeline = SimpleNamespace(
            pending_hit_allocs=deque(),
            try_lock_anchor=lambda request, tokens: ("locked", 0),
            staged_span_covered=lambda request, tokens: False,
            reserve_prefetch=lambda operation, info, hit, aux: "reserved",
            dispatch_reserved_prefetch=dispatched.append,
            rollback_reserved_prefetch=rolled_back.append,
            release_anchor_lock=lambda request: None,
        )
        cache.buffer_pipeline = pipeline
        cache.cache_controller = SimpleNamespace(
            prefetch_hit_queue=Queue(),
            ack_prefetch_queue=Queue(),
            ack_backup_queue=Queue(),
            host_mem_release_queue=Queue(),
            extra_host_mem_release_queues={},
            prefetch_rate_limited=lambda: False,
        )
        cache.ongoing_prefetch = {}

        def reduce_min(tensor, op):
            peer = torch.tensor(peer_outcomes, dtype=torch.int64)
            torch.minimum(tensor, peer, out=tensor)

        cache._all_reduce_attn_groups = reduce_min
        return cache, pipeline, dispatched, rolled_back

    def _hit(self, cache, rid):
        operation = _operation(rid)
        operation.storage_hit_count = len(TOKENS)
        operation.hash_value = list(CHAIN)
        cache.ongoing_prefetch[operation.handle] = _OngoingPrefetch(
            0, RadixKey(array("q", TOKENS)), None, operation, None, {}
        )
        cache.cache_controller.prefetch_hit_queue.put(operation)
        return operation

    def _drain(self, cache, hits):
        cache._drain_storage_control_queues_impl(
            n_storage_hit=hits,
            n_ack_prefetch=0,
            n_backup=0,
            n_release=0,
            extra_release_counts=None,
            log_metrics=False,
        )

    def test_peer_shortfall_parks_a_locally_reserved_hit(self):
        # The peer is short on a and reserved b; b behind a keeps its turn.
        cache, pipeline, dispatched, rolled_back = self._cache(
            [[0, 0], [1, len(TOKENS)]]
        )
        a, b = self._hit(cache, "a"), self._hit(cache, "b")
        self._drain(cache, 2)
        self.assertEqual(rolled_back, [a, b])
        self.assertEqual(dispatched, [])
        self.assertEqual(list(pipeline.pending_hit_allocs), [a, b])
        self.assertEqual(cache._prefetch_outcome_stats["declined_rate_limited"], 2)

    def test_agreed_reservations_dispatch_in_order(self):
        cache, pipeline, dispatched, rolled_back = self._cache(
            [[1, len(TOKENS)], [1, len(TOKENS)]]
        )
        a, b = self._hit(cache, "a"), self._hit(cache, "b")
        self._drain(cache, 2)
        self.assertEqual(dispatched, [a, b])
        self.assertEqual(rolled_back, [])
        self.assertFalse(pipeline.pending_hit_allocs)

    def test_local_shortfall_parks_without_rollback(self):
        cache, pipeline, dispatched, rolled_back = self._cache([[1, len(TOKENS)]])
        pipeline.reserve_prefetch = lambda operation, info, hit, aux: "short"
        a = self._hit(cache, "a")
        self._drain(cache, 1)
        self.assertEqual((rolled_back, dispatched), ([], []))
        self.assertEqual(list(pipeline.pending_hit_allocs), [a])

    def test_settled_outcome_must_match_the_peers(self):
        # This rank declines (span device-covered) while the peer reserved:
        # the tree state is no longer lockstep, which is fatal, not a park.
        cache, pipeline, *_ = self._cache([[1, len(TOKENS)]])
        pipeline.staged_span_covered = lambda request, tokens: True
        self._hit(cache, "a")
        with self.assertRaisesRegex(RuntimeError, "diverged"):
            self._drain(cache, 1)

    def test_parked_hit_is_retried_first_and_dispatches_once_agreed(self):
        cache, pipeline, dispatched, rolled_back = self._cache([[1, len(TOKENS)]])
        a = self._hit(cache, "a")
        pipeline.reserve_prefetch = lambda operation, info, hit, aux: "short"
        self._drain(cache, 1)
        self.assertEqual(list(pipeline.pending_hit_allocs), [a])
        pipeline.reserve_prefetch = lambda operation, info, hit, aux: "reserved"
        self._drain(cache, 0)
        self.assertEqual(dispatched, [a])
        self.assertFalse(pipeline.pending_hit_allocs)
        self.assertEqual(cache._prefetch_outcome_stats["declined_rate_limited"], 1)


class _HoldingBackend:
    """Every call reports all pages present; the first query and the first
    read each block on an event when asked to."""

    def __init__(self, hold_query: bool, hold_get: bool):
        self.query_hold = threading.Event()
        self.get_hold = threading.Event()
        self.hold_query, self.hold_get = hold_query, hold_get
        self.queries = 0
        self.gets = 0

    def batch_exists(self, hashes, extra_info=None):
        self.queries += 1
        if self.hold_query and self.queries == 1:
            self.query_hold.wait()
        return len(hashes)

    def batch_get_v1(self, hashes, host_indices, extra_info=None):
        self.gets += 1
        if self.hold_get and self.gets == 1:
            self.get_hold.wait()
        return [True] * len(hashes)


class TestStorageCallLanes(unittest.TestCase):
    def test_unbounded_lane_runs_inline(self):
        backend = _HoldingBackend(hold_query=False, hold_get=False)
        lanes = StorageCallLanes(lambda: backend)
        self.assertEqual(lanes.batch_exists(CHAIN), 4)
        self.assertTrue(lanes.idle())
        ran = []
        lanes.when_idle(object(), lambda: ran.append(True))
        self.assertEqual(ran, [True])

    def test_timed_out_call_is_orphaned_and_holds_its_operation(self):
        backend = _HoldingBackend(hold_query=False, hold_get=True)
        lanes = StorageCallLanes(lambda: backend, timeout_s=0.1)
        operation = _operation("held")
        lanes.serve(operation)
        t0 = time.monotonic()
        with self.assertRaises(StorageCallTimeout):
            lanes.batch_get_v1(CHAIN, torch.arange(8))
        self.assertLess(time.monotonic() - t0, 2.0)
        self.assertFalse(lanes.idle())
        finished = []
        lanes.when_idle(operation, lambda: finished.append(True))
        self.assertEqual(finished, [])  # the orphan still runs
        # The thread's next call does not wait behind the orphan.
        t0 = time.monotonic()
        self.assertEqual(lanes.batch_get_v1(CHAIN, torch.arange(8)), [True] * 4)
        self.assertLess(time.monotonic() - t0, 0.1)
        backend.get_hold.set()
        deadline = time.monotonic() + 2.0
        while not finished and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(finished, [True])
        self.assertTrue(lanes.idle())
        self.assertEqual(lanes.orphaned_calls, 1)
        lanes.close()

    def test_bound_must_be_positive(self):
        with self.assertRaises(ValueError):
            StorageCallLanes(lambda: object(), timeout_s=0)


class TestResetInPlace(unittest.TestCase):
    """With scheduler-thread agreement the workers only do I/O: reset waits
    until they hold no rows and clears the queues under them, joining nothing."""

    def _controller(self, backend):
        cc = HiCacheController.__new__(HiCacheController)
        cc.page_size = PAGE
        cc.enable_storage = True
        cc.scheduler_agrees_storage = True
        cc.storage_backend = backend
        cc.storage_calls = StorageCallLanes(lambda: backend, timeout_s=0.2)
        cc.storage_call_timeout_s = 0.2
        cc._inflight_lock = threading.Lock()
        cc._inflight_transfers, cc._inflight_backups = set(), set()
        cc.on_prefetch_dispatched = None
        cc.storage_stop_event = threading.Event()
        cc.write_queue, cc.load_queue = [], []
        cc.ack_write_queue, cc.ack_load_queue = [], []
        cc.prefetch_queue, cc.backup_queue = Queue(), Queue()
        cc.prefetch_buffer, cc.prefetch_sync_queue = Queue(), Queue()
        cc.prefetch_hit_queue, cc.ack_prefetch_queue = Queue(), Queue()
        cc.ack_backup_queue, cc.host_mem_release_queue = Queue(), Queue()
        cc.prefetch_tokens_occupied = 3
        cc.page_get_func = cc._page_get_zero_copy
        cc.prefetch_sync_thread = cc.backup_thread = cc.prefetch_thread = None
        cc.prefetch_io_aux_thread = threading.Thread(
            target=cc.prefetch_io_aux_func, daemon=True
        )
        cc.prefetch_io_aux_thread.start()
        return cc

    def test_reset_waits_for_held_rows_and_keeps_the_workers(self):
        backend = _HoldingBackend(hold_query=False, hold_get=True)
        cc = self._controller(backend)
        operation = _operation("held")
        operation.hash_value = list(CHAIN)
        operation.host_indices = torch.arange(8)
        cc.prefetch_hit_queue.put(object())
        cc.dispatch_prefetch_transfer(operation)
        deadline = time.monotonic() + 2.0
        while not cc._inflight_transfers and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertTrue(cc._inflight_transfers)
        # The read is held past the call bound: the worker orphans it, the
        # operation stays held, and reset waits for the orphan to return.
        threading.Timer(0.5, backend.get_hold.set).start()
        t0 = time.monotonic()
        cc.reset()
        self.assertGreaterEqual(time.monotonic() - t0, 0.3)
        self.assertFalse(cc._inflight_transfers)
        self.assertTrue(operation.local_transfer_progress().finished)
        self.assertTrue(cc.prefetch_io_aux_thread.is_alive())
        self.assertFalse(cc.storage_stop_event.is_set())
        self.assertTrue(cc.prefetch_hit_queue.empty())
        self.assertEqual(cc.prefetch_tokens_occupied, 0)
        cc.storage_stop_event.set()
        cc.prefetch_buffer.put(None)


# ---------------------------------------------------------------- two ranks


def _controller(backend) -> HybridCacheController:
    """The real query and transfer workers over a scripted backend; agreement
    is the scheduler's, so the workers only record local outcomes."""
    cc = HybridCacheController.__new__(HybridCacheController)
    cc.page_size = PAGE
    cc.storage_backend = backend
    cc.storage_calls = StorageCallLanes(lambda: backend)
    cc.storage_call_timeout_s = None
    cc._inflight_lock = threading.Lock()
    cc._inflight_transfers = set()
    cc._inflight_backups = set()
    cc.scheduler_agrees_storage = True
    cc.on_prefetch_dispatched = None
    cc.storage_stop_event = threading.Event()
    cc.prefetch_queue = Queue()
    cc.prefetch_buffer = Queue()
    cc.prefetch_hit_queue = Queue()
    cc.ack_prefetch_queue = Queue()
    cc.ack_backup_queue = Queue()
    cc.host_mem_release_queue = Queue()
    cc.extra_host_mem_release_queues = {}
    cc.ack_write_queue = []
    cc.ack_load_queue = []
    cc.start_writing = Mock()
    cc.page_get_func = cc._page_get_zero_copy
    cc.mem_pool_host = SimpleNamespace(layout_lease=nullcontext, page_size=PAGE)
    cc.prefetch_sync_thread = None
    cc.backup_thread = None
    cc.prefetch_thread = threading.Thread(target=cc.prefetch_thread_func, daemon=True)
    cc.prefetch_io_aux_thread = threading.Thread(
        target=cc.prefetch_io_aux_func, daemon=True
    )
    cc.prefetch_thread.start()
    cc.prefetch_io_aux_thread.start()
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
    cache.prefetch_stop_policy = "wait_complete"
    cache.storage_prefetch_retries = StoragePrefetchRetries()
    cache._storage_prefetch_wait = {}
    cache._prefetch_outcome_stats = defaultdict(int)
    cache.ongoing_prefetch = {}
    cache.storage_engine = StorageAgreementEngine(
        PAGE,
        attention_group_min(cache._reduce_min_attn_groups),
        cache._storage_workers_alive,
        OperationRowSource(
            PAGE,
            query_expired=cache._query_expired,
            transfer_expired=cache._transfer_expired,
        ),
        backend_stopped=cache._storage_workers_stopped,
    )
    cc.on_prefetch_dispatched = cache.storage_engine.register_transfer
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


def _dispatch(cache, operation) -> None:
    """What the hit drain does after the bounce alloc."""
    operation.host_indices = torch.arange(len(TOKENS))
    cache.ongoing_prefetch[operation.handle] = cache.ongoing_prefetch[
        operation.handle
    ]._replace(host_indices=operation.host_indices)
    cache.cache_controller.dispatch_prefetch_transfer(operation)


def _same_on_every_rank(snapshot):
    snapshots = [None] * dist.get_world_size()
    dist.all_gather_object(snapshots, snapshot)
    assert all(other == snapshots[0] for other in snapshots), snapshots
    return snapshots[0]


def _drain_counts(cache) -> tuple[int, int]:
    kwargs = cache._drain_storage_control_queues_impl.call_args.kwargs
    return kwargs["n_storage_hit"], kwargs["n_ack_prefetch"]


def _tick_until(cache, done, budget_s=20.0):
    """Tick until ``done(hits, acks)`` holds; returns (tick index, max tick time)."""
    start = time.monotonic()
    worst = 0.0
    for tick in range(int(budget_s / 0.02)):
        t0 = time.monotonic()
        cache.check_hicache_events()
        worst = max(worst, time.monotonic() - t0)
        if done(*_drain_counts(cache)):
            return tick, worst, time.monotonic() - start
        time.sleep(0.02)
    raise AssertionError("the agreed outcome was never published")


def _slow_backend_body(rank: int) -> None:
    """Rank 1's backend blocks its first query, then its first read. Rank 0's
    scheduler keeps ticking (nothing waits on a worker), nothing is published
    until each hold lifts, and then both ranks publish the same agreed outcome
    in the same tick."""
    backend = _HoldingBackend(hold_query=(rank == 1), hold_get=(rank == 1))
    cc = _controller(backend)
    cache = _scheduler_cache(cc)
    get_storage_hash_str(TOKENS, None, page_size=PAGE)  # JIT warm-up off the clock

    # Query phase.
    operation = _submit(cache, "slow")
    if rank == 1:
        threading.Timer(HOLD_S, backend.query_hold.set).start()
    tick, worst, elapsed = _tick_until(cache, lambda hits, acks: hits > 0)
    assert elapsed >= HOLD_S * 0.8, elapsed
    assert worst < TICK_BUDGET_S, worst
    assert tick >= 3, tick
    assert cc.prefetch_hit_queue.get_nowait() is operation
    _same_on_every_rank((tick, operation.storage_hit_count, operation.hash_value))
    assert operation.storage_hit_count == len(TOKENS)

    # Transfer phase: the read is held on rank 1 the same way.
    _dispatch(cache, operation)
    if rank == 1:
        threading.Timer(HOLD_S, backend.get_hold.set).start()
    tick, worst, elapsed = _tick_until(cache, lambda hits, acks: acks > 0)
    assert elapsed >= HOLD_S * 0.8, elapsed
    assert worst < TICK_BUDGET_S, worst
    ack = cc.ack_prefetch_queue.get_nowait()
    assert ack.operation is operation and ack.completed_req
    assert ack.pool_hits is None
    _same_on_every_rank((tick, ack.completed_tokens))
    assert ack.completed_tokens == len(TOKENS)
    assert operation.local_transfer_progress().finished
    assert cache.storage_engine.pending_transfers() == 0
    assert cc.storage_workers_alive()

    # Deadline phase: the clock expired on rank 0 only; both ranks cut the
    # transfer in the same tick and still agree the outcome afterwards.
    cache.prefetch_stop_policy = "timeout"
    second = _submit(cache, "deadline")
    _tick_until(cache, lambda hits, acks: hits > 0)
    cc.prefetch_hit_queue.get_nowait()
    backend.hold_get, backend.gets = True, 0
    backend.get_hold.clear()
    _dispatch(cache, second)
    if rank == 0:
        cache._storage_prefetch_wait[second.handle] = SimpleNamespace(expired=True)
    cut_tick, _, _ = _tick_until(
        cache, lambda hits, acks: second.is_terminated() or acks > 0
    )
    assert second.deadline_expired and second.is_terminated()
    _same_on_every_rank(cut_tick)
    backend.get_hold.set()
    _tick_until(cache, lambda hits, acks: acks > 0)
    ack = cc.ack_prefetch_queue.get_nowait()
    _same_on_every_rank(ack.completed_tokens)
    cache.prefetch_stop_policy = "wait_complete"

    # Orphan phase: calls are bounded on both ranks; rank 1 holds the read of
    # the first operation past the bound, so its worker orphans the call and
    # serves the next operation. That one publishes first on every rank; the
    # held one stays unfinished (its rows quarantined) until the orphan
    # returns, then publishes with the agreed prefix (0: it failed on rank 1).
    cc.storage_calls = StorageCallLanes(lambda: backend, timeout_s=0.3)
    cc.storage_call_timeout_s = 0.3
    held = _submit(cache, "held")
    behind = _submit(cache, "behind")
    _tick_until(cache, lambda hits, acks: cc.prefetch_hit_queue.qsize() == 2)
    cc.prefetch_hit_queue.get_nowait()
    cc.prefetch_hit_queue.get_nowait()
    backend.hold_get, backend.gets = rank == 1, 0
    backend.get_hold.clear()
    _dispatch(cache, held)
    _dispatch(cache, behind)
    if rank == 1:
        threading.Timer(HOLD_S, backend.get_hold.set).start()
    tick, _, elapsed = _tick_until(cache, lambda hits, acks: acks > 0)
    ack = cc.ack_prefetch_queue.get_nowait()
    assert ack.operation is behind, ack.operation.request_id
    assert elapsed < HOLD_S * 0.9, elapsed  # did not wait behind the held call
    _same_on_every_rank((tick, ack.completed_tokens))
    assert cache.storage_engine.pending_transfers() == 1
    _tick_until(cache, lambda hits, acks: acks > 0)
    ack = cc.ack_prefetch_queue.get_nowait()
    assert ack.operation is held
    _same_on_every_rank(ack.completed_tokens)
    assert ack.completed_tokens == 0
    assert cc.storage_calls.idle()
    cc.storage_calls = StorageCallLanes(lambda: backend)
    cc.storage_call_timeout_s = None

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

    # A dead storage worker is a fatal row: every rank raises in the same tick.
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
        raised = "worker is gone" in str(e)
    assert raised, "dead worker was not reported"
    _same_on_every_rank(raised)

    # Storage stopped on purpose: the same gone worker now means "no backend".
    # A query that was answered keeps its verdict; one that no worker will
    # answer (never enqueued here) is published as an agreed miss, no raise.
    cc.storage_stop_event.set()
    fourth = _submit(cache, "fourth", enqueue=False)
    cache.check_hicache_events()
    assert _drain_counts(cache)[0] == 2
    assert cc.prefetch_hit_queue.get_nowait() is third
    assert cc.prefetch_hit_queue.get_nowait() is fourth
    _same_on_every_rank((third.storage_hit_count, fourth.storage_hit_count))
    assert third.storage_hit_count == len(TOKENS) and fourth.storage_hit_count == 0
    cc.prefetch_queue.put(None)
    cc.prefetch_buffer.put(None)


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
