"""Bounded HiCache storage-backend calls: a hung or failing backend call costs
one operation, never the worker thread or the host memory it touches (CPU
only, mock backends)."""

import threading
import time
import unittest
from collections import defaultdict
from queue import Empty, Queue
from unittest import mock

import torch

import sglang.srt.mem_cache.storage_call_guard as storage_call_guard_module
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.managers.cache_controller import PrefetchOperation as BasePrefetchOp
from sglang.srt.managers.cache_controller import StorageOperation as BaseStorageOp
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageExtraInfo,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    PrefetchOperation as HybridPrefetchOp,
)
from sglang.srt.mem_cache.memory_pool_host import LogicalHostPool
from sglang.srt.mem_cache.pool_host import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.pool_host.base import HostKVCache
from sglang.srt.mem_cache.pool_host.unified import UnifiedPageEnvelopeHostPool
from sglang.srt.mem_cache.storage_call_guard import (
    StorageCallGuard,
    StorageCallKind,
    StorageCallTimeouts,
)
from sglang.srt.mem_cache.unified_memory_pool import init_unified_swa_pools
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

PAGE_SIZE = 2
POOL_SIZE = 16
BOUND_S = 0.2


class _Backend:
    """Every batch call succeeds, except the (method, call index) pairs that
    block on ``gate`` or raise."""

    def __init__(self, hang=(), fail=(), error=RuntimeError):
        self.hang, self.fail, self.error = set(hang), set(fail), error
        self.calls = defaultdict(int)
        self.threads = defaultdict(list)
        self.gate = threading.Event()
        self.hung = threading.Event()

    def _enter(self, method):
        index = self.calls[method]
        self.calls[method] += 1
        self.threads[method].append(threading.current_thread().name)
        if (method, index) in self.hang:
            self.hung.set()
            self.gate.wait()
        if (method, index) in self.fail:
            raise self.error(f"injected {method} #{index}")

    def batch_exists(self, keys, extra_info=None):
        self._enter("batch_exists")
        return len(keys)

    def batch_get_v1(self, keys, host_indices, extra_info=None):
        self._enter("batch_get_v1")
        return [True] * len(keys)

    def batch_get_v2(self, transfers, extra_info=None):
        self._enter("batch_get_v2")
        return {t.name: [True] * len(t.keys or []) for t in transfers}

    def batch_set_v1(self, keys, host_indices, extra_info=None):
        self._enter("batch_set_v1")
        return [True] * len(keys)

    def batch_set_v2(self, transfers, extra_info=None):
        self._enter("batch_set_v2")
        return {t.name: [True] * len(t.keys or []) for t in transfers}


def _controller(kind, backend, timeouts=None, swa_pool=None):
    cls = HiCacheController if kind == "base" else HybridCacheController
    cc = cls.__new__(cls)
    pool = LogicalHostPool(size=POOL_SIZE, page_size=PAGE_SIZE)
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
    cc.prefetch_hits_sync_groups = []
    cc.prefetch_completion_sync_groups = []
    cc.storage_sync_failed, cc.storage_sync_failure = False, None
    cc._storage_sync_lock = threading.Lock()
    cc._init_storage_worker_tracking()
    cc.backup_skip = False
    cc.page_get_func = cc._page_get_zero_copy
    cc.page_set_func = cc._page_set_zero_copy
    cc.storage_host_pool = pool
    if kind == "base":
        cc.mem_pool_host = pool
    else:
        entries = [
            PoolEntry(PoolName.KV, pool, None, None, is_primary_index_anchor=True)
        ]
        if swa_pool is not None:
            entries.append(PoolEntry(PoolName.SWA, swa_pool, None, None))
        cc.mem_pool_host = HostPoolGroup(entries)
        cc.pp_prefetch_command_thread = None
    cc.prefetch_threshold = PAGE_SIZE
    if timeouts is None:
        timeouts = StorageCallTimeouts(query_s=BOUND_S, get_s=BOUND_S, set_s=BOUND_S)
    cc.storage_call_guard = StorageCallGuard(timeouts, stop_event=cc.storage_stop_event)
    return cc, pool


def _prefetch_op(kind, rid, pool, num_pages=2):
    tokens = list(range(num_pages * PAGE_SIZE))
    if kind == "base":
        op = BasePrefetchOp(rid, tokens)
    else:
        op = HybridPrefetchOp(CacheRequestHandle(rid, 0), tokens)
    op.hash_value = [f"{rid}-{i}" for i in range(num_pages)]
    op.host_indices = pool.alloc(num_pages * PAGE_SIZE)
    return op


def _backup_op(pool, num_pages=2):
    host_indices = pool.alloc(num_pages * PAGE_SIZE)
    return BaseStorageOp(
        host_indices,
        list(range(num_pages * PAGE_SIZE)),
        hash_value=[f"b{i}" for i in range(num_pages)],
    )


def _start(target, name):
    thread = threading.Thread(target=target, daemon=True, name=name)
    thread.start()
    return thread


def _stop(cc, thread, queue):
    cc.storage_stop_event.set()
    queue.put(None)
    thread.join(5)


def _drain(queue, want, budget_s=5.0):
    got = []
    deadline = time.monotonic() + budget_s
    while len(got) < want and time.monotonic() < deadline:
        try:
            got.append(queue.get(timeout=0.1))
        except Empty:
            continue
    return got


def _orphans(cc, kind):
    return cc.storage_call_guard.take_stats().orphaned[kind]


def _wait_until(predicate, budget_s=5.0):
    deadline = time.monotonic() + budget_s
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


def _allocatable(pool):
    """Every slot the pool is still willing to hand out."""
    slots = []
    while True:
        got = pool.alloc(PAGE_SIZE)
        if got is None:
            break
        slots.extend(got.tolist())
    return sorted(slots)


class TestStorageCallTimeouts(CustomTestCase):
    def test_extra_config_keys_are_taken_and_zero_disables(self):
        extra = {
            "storage_query_timeout_s": 3,
            "storage_get_timeout_s": 0,
            "storage_set_timeout_s": None,
            "backend_name": "x",
        }
        timeouts = StorageCallTimeouts.pop_from_extra_config(extra)
        self.assertEqual(
            (timeouts.query_s, timeouts.get_s, timeouts.set_s), (3.0, None, None)
        )
        # Backends never see the controller's keys.
        self.assertEqual(extra, {"backend_name": "x"})
        defaults = StorageCallTimeouts.pop_from_extra_config(None)
        self.assertEqual(
            (defaults.query_s, defaults.get_s, defaults.set_s), (5.0, 20.0, 60.0)
        )
        # Scheduler-thread probes share the query deadline.
        self.assertEqual(timeouts.for_kind(StorageCallKind.PROBE), 3.0)
        for bad in ("5", True, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                StorageCallTimeouts.pop_from_extra_config(
                    {"storage_get_timeout_s": bad}
                )


class TestStorageCallGuard(CustomTestCase):
    def _guard(self, max_orphans=4):
        return StorageCallGuard(
            StorageCallTimeouts(
                query_s=BOUND_S, get_s=None, set_s=None, max_orphans=max_orphans
            )
        )

    @staticmethod
    def _query(guard, fn):
        return guard.run(
            StorageCallKind.QUERY, fn, miss=lambda: -1, op="batch_exists", num_keys=1
        )

    def test_refuses_new_calls_while_every_worker_is_orphaned(self):
        """With the orphan budget exhausted a call must fail fast as a miss
        rather than queue behind the hung workers or pin another thread, and
        the degraded state is reported once, not per call."""
        gate = threading.Event()
        guard = self._guard(max_orphans=1)
        self.assertEqual(self._query(guard, gate.wait), -1)
        with self.assertLogs(storage_call_guard_module.logger, level="ERROR") as logs:
            started = time.monotonic()
            for _ in range(3):
                self.assertEqual(self._query(guard, gate.wait), -1)
            self.assertLess(time.monotonic() - started, BOUND_S / 2)
        self.assertEqual(len([r for r in logs.records if r.levelname == "ERROR"]), 1)
        stats = guard.take_stats()
        self.assertEqual((stats.timeouts["query"], stats.errors["query"]), (1, 3))
        self.assertEqual(stats.orphaned["query"], 1)
        gate.set()
        self.assertTrue(_wait_until(lambda: guard.take_stats().orphaned["query"] == 0))
        self.assertEqual(self._query(guard, lambda: 7), 7)
        guard.shutdown()

    def test_backend_timeout_error_is_a_failure_not_a_deadline(self):
        """A backend raising TimeoutError (the same class as the future's wait
        timeout on 3.11+) must count as an error and keep the same worker,
        not retire it as an orphan."""
        guard = self._guard()
        workers = []

        def raise_timeout():
            workers.append(threading.current_thread())
            raise TimeoutError("rpc timed out")

        def record():
            workers.append(threading.current_thread())
            return 7

        self.assertEqual(self._query(guard, raise_timeout), -1)
        self.assertEqual(self._query(guard, record), 7)
        stats = guard.take_stats()
        self.assertEqual((stats.timeouts["query"], stats.errors["query"]), (0, 1))
        self.assertEqual(stats.orphaned["query"], 0)
        self.assertIs(workers[0], workers[1])
        guard.shutdown()

    def test_second_caller_on_one_kind_is_rejected(self):
        """A kind's worker is a FIFO with one caller; a second concurrent
        caller would otherwise inherit the first call's wait and could leave a
        queued job, its future and its pins stranded on an orphaned worker."""
        gate = threading.Event()
        guard = self._guard()
        first = threading.Thread(target=lambda: self._query(guard, gate.wait))
        first.start()
        try:
            self.assertTrue(
                _wait_until(lambda: guard._lanes[StorageCallKind.QUERY].busy)
            )
            with self.assertRaisesRegex(RuntimeError, "one thread"):
                self._query(guard, lambda: 1)
        finally:
            gate.set()
            first.join(5)
        guard.shutdown()


class TestBoundedControllerCalls(CustomTestCase):
    def test_hung_query_publishes_a_miss_and_later_queries_are_served(self):
        """One hung batch_exists must publish 0 hits for its operation while
        the next operation is still answered, on both controllers; a wedged
        query thread would otherwise starve every later prefetch and its peers."""
        for kind in ("base", "hybrid"):
            with self.subTest(kind=kind):
                backend = _Backend(hang={("batch_exists", 0)})
                cc, pool = _controller(kind, backend)
                worker = _start(cc.prefetch_thread_func, f"query-{kind}")
                ops = [_prefetch_op(kind, f"op{i}", pool) for i in range(2)]
                for op in ops:
                    op.host_indices = None
                    cc.prefetch_queue.put(op)
                published = _drain(cc.prefetch_hit_queue, 2)
                self.assertEqual(
                    [(op.request_id, op.storage_hit_count) for op in published],
                    [("op0", 0), ("op1", 2 * PAGE_SIZE)],
                )
                # The second query was answered while the first still hangs.
                self.assertFalse(backend.gate.is_set())
                self.assertTrue(worker.is_alive())
                backend.gate.set()
                self.assertTrue(_wait_until(lambda: _orphans(cc, "query") == 0))
                _stop(cc, worker, cc.prefetch_queue)

    def test_scheduler_probe_runs_on_its_own_lane(self):
        """The scheduler thread probes L3 (the PD-decode hit probe, PP0's
        ticket submission) while the prefetch thread's query for an earlier
        request may be in flight. The probe must have its own lane: a hung
        probe is a miss within the query bound and a healthy one is answered,
        never the one-caller error out of the scheduler loop."""
        for kind in ("base", "hybrid"):
            with self.subTest(kind=kind):
                backend = _Backend(hang={("batch_exists", 0), ("batch_exists", 1)})
                cc, pool = _controller(kind, backend)
                worker = _start(cc.prefetch_thread_func, f"query-{kind}")
                queued = _prefetch_op(kind, "queued", pool)
                queued.host_indices = None
                cc.prefetch_queue.put(queued)
                self.assertTrue(backend.hung.wait(5))
                probes = [_prefetch_op(kind, f"probe{i}", pool) for i in range(2)]
                started = time.monotonic()
                _, hit = cc._storage_hit_query(
                    probes[0], call_kind=StorageCallKind.PROBE
                )
                self.assertEqual(hit, 0)
                self.assertLess(time.monotonic() - started, 5 * BOUND_S)
                _, hit = cc._storage_hit_query(
                    probes[1], call_kind=StorageCallKind.PROBE
                )
                self.assertEqual(hit, 2 * PAGE_SIZE)
                self.assertEqual(
                    backend.threads["batch_exists"],
                    ["hicache-storage-query"] + ["hicache-storage-probe"] * 2,
                )
                stats = cc.storage_call_guard.take_stats()
                self.assertEqual(
                    (stats.orphaned["query"], stats.orphaned["probe"]), (1, 1)
                )
                self.assertEqual((stats.errors["query"], stats.errors["probe"]), (0, 0))
                self.assertFalse(backend.gate.is_set())
                backend.gate.set()
                self.assertTrue(
                    _wait_until(
                        lambda: (
                            sum(cc.storage_call_guard.take_stats().orphaned.values())
                            == 0
                        )
                    )
                )
                _stop(cc, worker, cc.prefetch_queue)

    def test_hung_transfer_quarantines_its_span_until_the_orphan_returns(self):
        """A timed-out page read is acked as 0 tokens so the scheduler frees
        the span, but the pool must not hand those slots out again while the
        orphaned read can still write them; a failing read is the same miss."""
        backend = _Backend(hang={("batch_get_v1", 0)}, fail={("batch_get_v1", 1)})
        cc, pool = _controller("base", backend)
        worker = _start(cc.prefetch_io_aux_func, "transfer")
        hung = _prefetch_op("base", "hung", pool)
        cc.prefetch_buffer.put(hung)
        acks = _drain(cc.prefetch_sync_queue, 2)
        self.assertEqual(
            [(ack.rid, ack.completed_tokens, ack.completed_req) for ack in acks],
            [("hung", 0, None), ("hung", None, True)],
        )
        # The ack drain releases the whole untransferred span ...
        pool.free(hung.host_indices)
        # ... yet nothing may be allocated from it while the read is in flight.
        quarantined = set(hung.host_indices.tolist())
        self.assertFalse(quarantined & set(_allocatable(pool)))
        self.assertEqual(pool.available_size(), 0)
        pool.clear()
        self.assertEqual(pool.available_size(), POOL_SIZE - len(quarantined))

        failed = _prefetch_op("base", "failed", pool)
        cc.prefetch_buffer.put(failed)
        acks = _drain(cc.prefetch_sync_queue, 2)
        self.assertEqual([ack.completed_tokens for ack in acks], [0, None])
        stats = cc.storage_call_guard.take_stats()
        self.assertEqual((stats.timeouts["get"], stats.errors["get"]), (1, 1))

        backend.gate.set()
        self.assertTrue(_wait_until(lambda: _orphans(cc, "get") == 0))
        self.assertEqual(pool.available_size(), POOL_SIZE - len(failed.host_indices))
        pool.free(failed.host_indices)
        served = _prefetch_op("base", "served", pool)
        cc.prefetch_buffer.put(served)
        acks = _drain(cc.prefetch_sync_queue, 2)
        self.assertEqual([ack.completed_tokens for ack in acks], [2 * PAGE_SIZE, None])
        self.assertTrue(worker.is_alive())
        _stop(cc, worker, cc.prefetch_buffer)

    def test_kv_miss_skips_the_derived_sidecar_read(self):
        """A batch whose KV read came back empty is a miss whatever its
        KV-derived sidecar holds; issuing the sidecar read anyway would cost a
        second bound on a hung backend and pin the same span twice."""
        backend = _Backend(fail={("batch_get_v1", 0)})
        cc, pool = _controller("base", backend)
        op = _prefetch_op("base", "op", pool)
        derived = [PoolTransfer(name=PoolName.INDEXER, indices_from_pool=PoolName.KV)]
        hits = cc._page_transfer_kv_batch(
            op, op.hash_value, op.host_indices, HiCacheStorageExtraInfo(), derived
        )
        self.assertEqual((hits, backend.calls["batch_get_v2"]), (0, 0))
        backend = _Backend()
        cc.storage_backend = backend
        hits = cc._page_transfer_kv_batch(
            op, op.hash_value, op.host_indices, HiCacheStorageExtraInfo(), derived
        )
        self.assertEqual((hits, backend.calls["batch_get_v2"]), (len(op.hash_value), 1))

    def test_hung_sidecar_read_acks_zero_pool_hits_and_pins_the_side_pool(self):
        """The hybrid sidecar read must still emit its pool_hits ack (the peer
        ranks reduce it) with 0 for the timed-out pool, and the side pool's
        own span stays quarantined."""
        swa_pool = LogicalHostPool(size=POOL_SIZE, page_size=PAGE_SIZE)
        backend = _Backend(hang={("batch_get_v2", 0)})
        cc, pool = _controller("hybrid", backend, swa_pool=swa_pool)
        op = _prefetch_op("hybrid", "swa", pool)
        swa = PoolTransfer(
            name=PoolName.SWA,
            host_indices=swa_pool.alloc(2 * PAGE_SIZE),
            keys=op.hash_value,
        )
        op.pool_transfers = [swa]
        cc._page_transfer_sidecar(op, kv_completed_pages=len(op.hash_value))
        ack = cc.prefetch_sync_queue.get_nowait()
        self.assertEqual(ack.pool_hits, {PoolName.SWA: 0})
        swa_pool.free(swa.host_indices)
        self.assertEqual(swa_pool.available_size(), POOL_SIZE - len(swa.host_indices))
        backend.gate.set()
        self.assertTrue(_wait_until(lambda: _orphans(cc, "get") == 0))
        self.assertEqual(swa_pool.available_size(), POOL_SIZE)

    def test_hung_backup_acks_zero_tokens_and_pins_its_source_span(self):
        """A timed-out write acks 0 tokens (count-stable for the peers) and
        its source span cannot be reused while the write may still read it."""
        backend = _Backend(hang={("batch_set_v1", 0)})
        cc, pool = _controller("base", backend)
        worker = _start(cc.backup_thread_func, "backup")
        hung = _backup_op(pool)
        cc.backup_queue.put(hung)
        (acked,) = _drain(cc.ack_backup_queue, 1)
        self.assertIs(acked, hung)
        self.assertEqual(hung.completed_tokens, 0)
        pool.free(hung.host_indices)
        self.assertFalse(set(hung.host_indices.tolist()) & set(_allocatable(pool)))
        backend.gate.set()
        self.assertTrue(_wait_until(lambda: _orphans(cc, "set") == 0))
        self.assertEqual(pool.available_size(), len(hung.host_indices))
        self.assertTrue(worker.is_alive())
        _stop(cc, worker, cc.backup_queue)

    def test_stop_request_drains_queued_calls_without_new_waits(self):
        """Once the stop event is set, the call in flight gets one grace period
        and every call queued behind it is published as a miss immediately, so
        a reset's thread join cannot be exceeded by a hung backend."""
        patcher = mock.patch.object(storage_call_guard_module, "_STOP_GRACE_S", 0.2)
        patcher.start()
        self.addCleanup(patcher.stop)
        backend = _Backend(hang={("batch_exists", 0)})
        cc, pool = _controller(
            "base", backend, StorageCallTimeouts(query_s=30, get_s=30, set_s=30)
        )
        worker = _start(cc.prefetch_thread_func, "query")
        ops = [_prefetch_op("base", f"op{i}", pool) for i in range(3)]
        for op in ops:
            op.host_indices = None
            cc.prefetch_queue.put(op)
        self.assertTrue(backend.hung.wait(5))
        started = time.monotonic()
        _stop(cc, worker, cc.prefetch_queue)
        self.assertFalse(worker.is_alive())
        self.assertLess(time.monotonic() - started, 2.0)
        published = _drain(cc.prefetch_hit_queue, 3, budget_s=1.0)
        self.assertEqual([op.storage_hit_count for op in published], [0, 0, 0])
        stats = cc.storage_call_guard.take_stats()
        self.assertEqual((stats.timeouts["query"], stats.orphaned["query"]), (1, 1))
        self.assertEqual(backend.calls["batch_exists"], 1)
        backend.gate.set()


class _SlotPool(HostKVCache):
    def get_size_per_token(self):
        return 1

    def init_kv_buffer(self):
        return None

    def load_to_device_per_layer(self, *args, **kwargs):
        pass

    def backup_from_device_all_layer(self, *args, **kwargs):
        pass

    def get_data_page(self, index, flat=True):
        raise NotImplementedError

    def get_dummy_flat_data_page(self):
        raise NotImplementedError

    def set_from_flat_data_page(self, index, data_page):
        raise NotImplementedError


class TestHostSlotQuarantine(CustomTestCase):
    def test_quarantined_slots_survive_free_and_clear_until_released(self):
        """The allocator contract behind the guard: a quarantined slot is not
        allocatable after its owner frees it or after a pool-wide clear, and
        comes back exactly once when released."""
        pool = _SlotPool(
            dtype=torch.uint8,
            device_capacity_tokens=8,
            size_per_token=1,
            start_layer=0,
            end_layer=1,
            host_to_device_ratio=1.0,
            host_size=0,
            page_size=PAGE_SIZE,
            layout="layer_first",
            pin_memory=False,
            device="cpu",
            is_dummy=True,
        )
        size = pool.available_size()
        span = pool.alloc(2 * PAGE_SIZE)
        pool.quarantine(span)
        pool.free(span)
        self.assertEqual(pool.available_size(), size - len(span))
        self.assertFalse(set(span.tolist()) & set(_allocatable(pool)))
        pool.clear()
        self.assertEqual(pool.available_size(), size - len(span))
        pool.release_quarantine(span)
        self.assertEqual(pool.available_size(), size)
        self.assertEqual(len(_allocatable(pool)), size)

    def test_overlapping_pins_are_counted(self):
        """Two calls can pin the same slots (a KV read and its sidecar read,
        a whole-operation write over its per-batch writes); the first one to
        return must not unprotect the slots the other still uses."""
        pool = LogicalHostPool(size=POOL_SIZE, page_size=PAGE_SIZE)
        span = pool.alloc(2 * PAGE_SIZE)
        pool.quarantine(span)
        pool.quarantine(span[:PAGE_SIZE])
        pool.free(span)
        pool.release_quarantine(span)
        self.assertEqual(pool.available_size(), POOL_SIZE - PAGE_SIZE)
        self.assertFalse(set(span[:PAGE_SIZE].tolist()) & set(_allocatable(pool)))
        pool.clear()
        self.assertEqual(pool.available_size(), POOL_SIZE - PAGE_SIZE)
        pool.release_quarantine(span[:PAGE_SIZE])
        self.assertEqual(pool.available_size(), POOL_SIZE)
        pool.release_quarantine(span)
        self.assertEqual(len(_allocatable(pool)), POOL_SIZE)

    def test_unified_page_envelope_pool_pins_pages_and_bounds_clear(self):
        """The shared page-envelope backing keeps a freed quarantined page out
        of its free list until released, and a clear that finds an orphaned
        call still holding the layout lease fails loudly instead of blocking
        the scheduler thread forever."""
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
        size = full.available_size()
        page = full.alloc(full.page_size)
        full.quarantine(page)
        full.free(page)
        self.assertEqual(full.available_size(), size - full.page_size)
        full.release_quarantine(page)
        self.assertEqual(full.available_size(), size)

        patcher = mock.patch(
            "sglang.srt.mem_cache.pool_host.unified._CLEAR_LEASE_WAIT_S", 0.2
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        leased = threading.Event()
        release = threading.Event()

        def hold_lease():
            with full._shared_backing.layout_lease():
                leased.set()
                release.wait()

        orphan = threading.Thread(target=hold_lease, daemon=True)
        orphan.start()
        self.assertTrue(leased.wait(5))
        try:
            with self.assertRaisesRegex(RuntimeError, "still leased"):
                full.clear()
        finally:
            release.set()
            orphan.join(5)
        full.clear()
        self.assertEqual(full.available_size(), size)


if __name__ == "__main__":
    unittest.main()
