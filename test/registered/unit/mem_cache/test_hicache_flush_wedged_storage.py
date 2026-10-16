"""A HiCache storage worker that cannot be joined must not take the scheduler
down on a cache flush: the flush completes with fresh workers, the stuck
worker is left on a dead generation with its host memory quarantined, and
every rank degrades L3 together (CPU only, mock backends)."""

import os
import tempfile
import threading
import time
import unittest
from collections import defaultdict, deque
from datetime import timedelta
from queue import Empty, Queue
from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist

import sglang.srt.managers.cache_controller as cache_controller_module
import sglang.srt.mem_cache.storage_call_guard as storage_call_guard_module
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.managers.cache_controller import PrefetchOperation as BasePrefetchOp
from sglang.srt.managers.cache_controller import StorageOperation as BaseStorageOp
from sglang.srt.mem_cache.memory_pool_host import LogicalHostPool
from sglang.srt.mem_cache.pool_host.unified import UnifiedPageEnvelopeHostPool
from sglang.srt.mem_cache.storage_call_guard import (
    StorageCallGuard,
    StorageCallTimeouts,
)
from sglang.srt.mem_cache.storage_prefetch import StoragePrefetchRetries
from sglang.srt.mem_cache.unified_memory_pool import init_unified_swa_pools
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

PAGE_SIZE = 2
POOL_SIZE = 16


class _Backend:
    """Every batch call succeeds, except the (method, call index) pairs that
    block on ``gate``."""

    def __init__(self, hang=()):
        self.hang = set(hang)
        self.calls = defaultdict(int)
        self.gate = threading.Event()
        self.hung = threading.Event()

    def _enter(self, method):
        index = self.calls[method]
        self.calls[method] += 1
        if (method, index) in self.hang:
            self.hung.set()
            self.gate.wait()

    def batch_exists(self, keys, extra_info=None):
        self._enter("batch_exists")
        return len(keys)

    def batch_get_v1(self, keys, host_indices, extra_info=None):
        self._enter("batch_get_v1")
        return [True] * len(keys)

    def batch_set_v1(self, keys, host_indices, extra_info=None):
        self._enter("batch_set_v1")
        return [True] * len(keys)


def _controller(backend, timeouts=None, pool=None):
    cc = HiCacheController.__new__(HiCacheController)
    pool = (
        pool
        if pool is not None
        else LogicalHostPool(size=POOL_SIZE, page_size=PAGE_SIZE)
    )
    cc.page_size = PAGE_SIZE
    cc.storage_backend = backend
    cc.storage_backend_type = "mooncake"
    cc.storage_stop_event = threading.Event()
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
    cc.write_queue, cc.load_queue = [], []
    cc.ack_write_queue, cc.ack_load_queue = [], []
    cc.prefetch_hits_sync_groups = []
    cc.prefetch_completion_sync_groups = []
    cc.storage_sync_failed, cc.storage_sync_failure = False, None
    cc._storage_sync_lock = threading.Lock()
    cc.storage_failure_policy = "disable"
    cc._init_storage_worker_tracking()
    cc.backup_skip = False
    cc.page_get_func = cc._page_get_zero_copy
    cc.page_set_func = cc._page_set_zero_copy
    cc.storage_host_pool = cc.mem_pool_host = pool
    cc.prefetch_threshold = PAGE_SIZE
    cc.prefetch_tokens_occupied = 0
    cc.enable_storage = True
    cc.prefetch_thread = cc.prefetch_io_aux_thread = None
    cc.prefetch_sync_thread = cc.backup_thread = None
    if timeouts is None:
        timeouts = StorageCallTimeouts.disabled()
    cc.storage_call_guard = StorageCallGuard(timeouts, stop_event=cc.storage_stop_event)
    return cc, pool


def _backup_op(pool, num_pages=2):
    host_indices = pool.alloc(num_pages * PAGE_SIZE)
    return BaseStorageOp(
        host_indices,
        list(range(num_pages * PAGE_SIZE)),
        hash_value=[f"b{i}" for i in range(num_pages)],
    )


def _prefetch_op(pool, rid="op", num_pages=2):
    op = BasePrefetchOp(rid, list(range(num_pages * PAGE_SIZE)))
    op.hash_value = [f"{rid}-{i}" for i in range(num_pages)]
    op.host_indices = pool.alloc(num_pages * PAGE_SIZE)
    return op


def _start(target, name):
    thread = threading.Thread(target=target, daemon=True, name=name)
    thread.start()
    return thread


def _drain(queue, want, budget_s=5.0):
    got = []
    deadline = time.monotonic() + budget_s
    while len(got) < want and time.monotonic() < deadline:
        try:
            got.append(queue.get(timeout=0.1))
        except Empty:
            continue
    return got


def _wait_until(predicate, budget_s=5.0):
    deadline = time.monotonic() + budget_s
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


def _allocatable(pool):
    slots = []
    while True:
        got = pool.alloc(pool.page_size)
        if got is None:
            break
        slots.extend(got.tolist())
    return sorted(slots)


class TestResetWithStuckWorkers(CustomTestCase):
    def setUp(self):
        # A bounded call notices the stop request within its 0.5 s poll and
        # then gets the (shortened) grace period; the join must outlast both,
        # while keeping the wedged case fast.
        for module, name, value in (
            (cache_controller_module, "_STORAGE_THREAD_JOIN_TIMEOUT_S", 2.0),
            (storage_call_guard_module, "_STOP_GRACE_S", 0.2),
        ):
            patcher = mock.patch.object(module, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_reset_abandons_a_wedged_worker_on_a_dead_generation(self):
        """With deadlines off a worker stuck in the backend cannot be joined.
        reset must report it instead of raising, restart fresh workers, keep
        the stuck operation's host memory quarantined across the pool clear,
        flag the rank unhealthy, and the stuck worker, once it returns, must
        publish nothing into the live queues and release its quarantine. A
        reset with nothing in flight stays healthy, and a later clean reset
        does not lift the degrade: only a re-attach does."""
        backend = _Backend(hang={("batch_set_v1", 0)})
        cc, pool = _controller(backend)
        cc.backup_thread = _start(cc.backup_thread_func, "backup")
        self.assertIsNone(cc.reset())
        self.assertTrue(cc.storage_workers_healthy())
        cc.clear_host_pool()
        self.assertEqual(pool.available_size(), POOL_SIZE)
        wedged = cc.backup_thread
        stuck = _backup_op(pool)
        cc.backup_queue.put(stuck)
        self.assertTrue(backend.hung.wait(5))

        reason = cc.reset()
        self.assertIn("did not stop", reason)
        self.assertEqual(cc.storage_health_reason(), reason)
        self.assertFalse(cc.storage_workers_healthy())
        self.assertTrue(cc.enable_storage)
        self.assertFalse(cc.storage_stop_event.is_set())
        self.assertIsNot(cc.backup_thread, wedged)
        self.assertTrue(cc.backup_thread.is_alive() and wedged.is_alive())
        cc.clear_host_pool()
        self.assertEqual(pool.available_size(), POOL_SIZE - len(stuck.host_indices))
        self.assertFalse(set(stuck.host_indices.tolist()) & set(_allocatable(pool)))
        pool.clear()
        with self.assertRaisesRegex(RuntimeError, "abandoned by a flush"):
            cc.detach_storage_backend()
        # The new generation serves while the old one is still stuck.
        fresh = _backup_op(pool)
        cc.backup_queue.put(fresh)
        (acked,) = _drain(cc.ack_backup_queue, 1)
        self.assertIs(acked, fresh)

        backend.gate.set()
        wedged.join(5)
        self.assertFalse(wedged.is_alive())
        self.assertTrue(cc.ack_backup_queue.empty())
        self.assertEqual(pool.available_size(), POOL_SIZE - len(fresh.host_indices))
        # A later flush joins the new generation, but the degrade persists
        # until a re-attach clears the reason.
        self.assertIsNone(cc.reset())
        self.assertFalse(cc.storage_workers_healthy())
        cc.detach_storage_backend()
        self.assertFalse(cc.enable_storage)

    def test_abandoned_transfer_worker_cannot_touch_the_live_queues(self):
        """The io worker of a dead generation must drop its progress acks and
        its host releases: a late ack would misalign the peers' ack sequence
        and a late release would double-free pages the clear already freed."""
        backend = _Backend(hang={("batch_get_v1", 0)})
        cc, pool = _controller(backend)
        wedged = cc.prefetch_io_aux_thread = _start(cc.prefetch_io_aux_func, "io")
        stuck = _prefetch_op(pool)
        cc.prefetch_buffer.put(stuck)
        self.assertTrue(backend.hung.wait(5))
        self.assertIn("did not stop", cc.reset())
        cc.clear_host_pool()
        self.assertEqual(pool.available_size(), POOL_SIZE - len(stuck.host_indices))

        backend.gate.set()
        wedged.join(5)
        self.assertFalse(wedged.is_alive())
        self.assertTrue(cc.prefetch_sync_queue.empty())
        self.assertTrue(cc.host_mem_release_queue.empty())
        self.assertEqual(pool.available_size(), POOL_SIZE)
        # A stale worker's release is dropped even when issued directly.
        released = threading.Event()

        def stale_release():
            cc._enter_storage_worker()
            cc._storage_generation += 1
            cc.append_host_mem_release(torch.arange(PAGE_SIZE))
            released.set()

        _start(stale_release, "stale")
        self.assertTrue(released.wait(5))
        self.assertTrue(cc.host_mem_release_queue.empty())
        cc.storage_stop_event.set()
        cc.prefetch_buffer.put(None)
        cc.prefetch_io_aux_thread.join(5)


def _scheduler_cache(cc) -> UnifiedRadixCache:
    """The scheduler-thread side: the production check_hicache_events round,
    whose cross-rank MIN runs on the default process group; only the L2 ack
    checks and the buffer flush are stubbed."""
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.cache_controller = cc
    cache.tree_core = SimpleNamespace(
        enable_storage=True, page_size=PAGE_SIZE, write_back_duplicate_reclaim_digest=0
    )
    cache.storage_unavailable = False
    cache._storage_unavailable_warned_at = 0.0
    cache.host_memory_mode = "buffer_only"
    cache.prefetch_threshold = PAGE_SIZE
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
    cache.writing_check = mock.Mock()
    cache.loading_check = mock.Mock()
    cache.storage_prefetch_retries = StoragePrefetchRetries()
    cache._prefetch_outcome_stats = defaultdict(int)
    cache._storage_prefetch_hit_remaining_by_reqid = {}
    cache._record_storage_prefetch_hit = mock.Mock()
    cache.revoke_pending_prefetch = mock.Mock()
    cache._handle_prefetch_result = mock.Mock()
    cache.buffer_pipeline = SimpleNamespace(
        pending_hit_allocs=deque(),
        has_staged=lambda request: False,
        flush_pending_writes=mock.Mock(),
    )
    cache.ongoing_prefetch = {}
    return cache


def _wedged_rank_body(rank: int) -> None:
    backend = _Backend(hang={("batch_set_v1", 0)} if rank == 1 else ())
    cc, pool = _controller(backend)
    cc.extra_host_mem_release_queues = {}
    cc.backup_thread = threading.Thread(target=cc.backup_thread_func, daemon=True)
    cc.backup_thread.start()
    cc.prefetch_thread = threading.Thread(target=cc.prefetch_thread_func, daemon=True)
    cc.prefetch_thread.start()
    cc.prefetch_io_aux_thread = threading.Thread(
        target=cc.prefetch_io_aux_func, daemon=True
    )
    cc.prefetch_io_aux_thread.start()
    cc.prefetch_sync_thread = threading.Thread(
        target=cc.prefetch_sync_thread_func, daemon=True
    )
    cc.prefetch_sync_thread.start()
    if rank == 1:
        cc.backup_queue.put(_backup_op(pool))
        assert backend.hung.wait(5)
    dist.barrier()

    # The operator flushes: only rank 1 has a worker that will not stop.
    with mock.patch.object(
        cache_controller_module, "_STORAGE_THREAD_JOIN_TIMEOUT_S", 1.0
    ):
        reason = cc.reset()
    assert (reason is None) == (rank == 0), (rank, reason)
    assert cc.enable_storage
    cache = _scheduler_cache(cc)
    # The next scheduler round MIN-reduces the health bit: both ranks degrade
    # together and the rank-agreed all-reduce keeps one tensor length
    # everywhere; a round on the degraded ranks still agrees.
    cache.check_hicache_events()
    assert cache.storage_unavailable, rank
    cache.check_hicache_events()
    snapshots = [None, None]
    dist.all_gather_object(snapshots, (cache.storage_unavailable, cc.enable_storage))
    assert snapshots == [(True, True), (True, True)], snapshots
    backend.gate.set()
    cc.storage_stop_event.set()
    for queue in (
        cc.backup_queue,
        cc.prefetch_queue,
        cc.prefetch_buffer,
        cc.prefetch_sync_queue,
    ):
        queue.put(None)
    for thread in cc._storage_worker_threads():
        thread.join(5)


def _rank_main(rank: int, rendezvous: str) -> None:
    # A regression that turns the degrade into a hang must fail, not block CI.
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
        _wedged_rank_body(rank)
    finally:
        dist.destroy_process_group()


class TestWedgeIsRankAgreed(CustomTestCase):
    def test_a_wedged_rank_degrades_storage_on_every_rank(self):
        """A rank-local wedge must not leave the healthy ranks with workers
        waiting on a peer that has none: the health bit drops storage on both
        ranks in the same drain round, with enable_storage untouched."""
        with tempfile.TemporaryDirectory(prefix="hicache-wedge-test-") as directory:
            torch.multiprocessing.spawn(
                _rank_main, args=(f"{directory}/store",), nprocs=2, join=True
            )


class TestUnifiedPoolAbandonedLeases(CustomTestCase):
    def _pools(self):
        bundle = init_unified_swa_pools(
            device="cpu",
            kv_cache_dtype=torch.float16,
            head_num=2,
            head_dim=4,
            v_head_dim=4,
            swa_head_num=2,
            swa_head_dim=4,
            swa_v_head_dim=4,
            page_size=4,
            start_layer=0,
            end_layer=4,
            swa_attention_layer_ids=[3],
            full_attention_layer_ids=[0, 1, 2],
            total_bytes=4096,
            enable_memory_saver=False,
            need_sort=False,
            lazy_compaction=True,
        )
        device_pool = bundle.token_to_kv_pool
        full, swa = UnifiedPageEnvelopeHostPool.build_hybrid_swa_pool_pair(
            device_pools=(device_pool.full_kv_pool, device_pool.swa_kv_pool),
            host_to_device_ratio=1.0,
            host_size=0,
            page_size=device_pool.page_size,
            layout="layer_first",
            pin_memory=False,
            allocator_type="default",
        )
        self.addCleanup(full.destroy)
        self.addCleanup(swa.destroy)
        return full

    def test_clear_keeps_pinned_pages_and_skips_abandoned_leases(self):
        """A flush must not wait for the lease of an abandoned storage call,
        and the page that call may still write must survive the clear in
        place, unallocatable, until it is released."""
        full = self._pools()
        backing = full._shared_backing
        size = full.available_size()
        first = full.alloc(full.page_size)
        pinned = full.alloc(full.page_size)
        full.quarantine(pinned)
        physical_before = backing.translate_indices(full.pool_label, pinned).clone()
        leased = threading.Event()
        release = threading.Event()

        def abandoned_call():
            with backing.layout_lease():
                leased.set()
                release.wait()

        orphan = threading.Thread(target=abandoned_call, daemon=True)
        orphan.start()
        self.assertTrue(leased.wait(5))
        backing.abandon_leases_of({orphan.ident})
        started = time.monotonic()
        full.clear()
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(full.available_size(), size - full.page_size)
        self.assertFalse(set(pinned.tolist()) & set(_allocatable(full)))
        self.assertTrue(
            torch.equal(
                backing.translate_indices(full.pool_label, pinned), physical_before
            )
        )
        release.set()
        orphan.join(5)
        full.release_quarantine(pinned)
        self.assertEqual(full.available_size(), full.page_size)
        full.clear()
        self.assertEqual(full.available_size(), size)
        del first

    def test_compaction_leaves_pinned_pages_in_place(self):
        """Compaction packs live pages but must not move a quarantined one,
        whose abandoned writer addresses its old physical slot."""
        full = self._pools()
        backing = full._shared_backing
        pages = [full.alloc(full.page_size) for _ in range(3)]
        pinned = pages[2]
        full.quarantine(pinned)
        physical_before = backing.translate_indices(full.pool_label, pinned).clone()
        full.free(pages[0])
        full.free(pages[1])
        with backing.lock:
            backing._compact()
        self.assertTrue(
            torch.equal(
                backing.translate_indices(full.pool_label, pinned), physical_before
            )
        )
        later = full.alloc(2 * full.page_size)
        self.assertIsNotNone(later)
        self.assertFalse(set(later.tolist()) & set(pinned.tolist()))
        full.release_quarantine(pinned)

    def test_abandoned_leases_are_per_acquire_not_per_thread(self):
        """A thread ident is reused once its thread exits, so abandonment must
        mark the leases held at marking time, never the ident: a lease taken
        afterwards under the same ident is waited for again."""
        full = self._pools()
        backing = full._shared_backing

        def holder(leased, release):
            with backing.layout_lease():
                leased.set()
                release.wait()

        def start_holder():
            leased, release = threading.Event(), threading.Event()
            thread = threading.Thread(
                target=holder, args=(leased, release), daemon=True
            )
            thread.start()
            self.assertTrue(leased.wait(5))
            return thread, release

        first, first_release = start_holder()
        backing.abandon_leases_of({first.ident})
        with backing.lock:
            self.assertEqual(backing._awaited_layout_users(), 0)
        first_release.set()
        first.join(5)
        with backing.lock:
            self.assertEqual(backing._abandoned_lease_tokens, set())
        # Marking an ident that holds nothing abandons nothing; the next
        # lease under it (the recycled ident) is waited for.
        backing.abandon_leases_of({first.ident})
        second, second_release = start_holder()
        with backing.lock:
            self.assertEqual(backing._awaited_layout_users(), 1)
            with self.assertRaisesRegex(RuntimeError, "still leased"):
                backing._begin_compaction(timeout=0.2)
        second_release.set()
        second.join(5)
        with backing.lock:
            self.assertEqual(backing._awaited_layout_users(), 0)


if __name__ == "__main__":
    unittest.main()
