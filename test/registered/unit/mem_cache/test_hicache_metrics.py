"""CPU unit tests for the HiCache write-path gauges and L3 backup failure counters."""

import threading
import unittest
from functools import partial
from queue import Queue
from types import SimpleNamespace
from unittest import mock

import torch
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from sglang.srt.managers.cache_controller import HiCacheController, StorageOperation
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    StorageOperation as HybridStorageOperation,
)
from sglang.srt.mem_cache.unified_cache.storage_attachment import StorageAttachment
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.observability.metrics_collector import StorageMetricsCollector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_LABELS = {
    "storage_backend": "file",
    "tp_rank": "0",
    "dp_rank": "0",
    "pp_rank": "0",
    "pp_size": "1",
    "attn_cp_rank": "0",
    "attn_cp_size": "1",
}

_GAUGE_NAMES = (
    "sglang:hicache_d2h_pending_ops",
    "sglang:hicache_d2h_ack_queue_depth",
    "sglang:hicache_l3_pending_ops",
    "sglang:hicache_l3_backup_queue_depth",
    "sglang:hicache_l3_ack_queue_depth",
    "sglang:hicache_l3_backup_worker_alive",
)


def _make_collector():
    # A private registry per collector avoids duplicate ``sglang:`` names.
    registry = CollectorRegistry()

    class _Collector(StorageMetricsCollector):
        _counter_cls = partial(Counter, registry=registry)
        _gauge_cls = partial(Gauge, registry=registry)
        _histogram_cls = partial(Histogram, registry=registry)

    return _Collector(labels=dict(_LABELS)), registry


def _gauges(registry):
    return tuple(registry.get_sample_value(name, _LABELS) for name in _GAUGE_NAMES)


def _sample(registry, name, **extra_labels):
    return registry.get_sample_value(name, {**_LABELS, **extra_labels})


class TestStorageMetricsCollectorHiCache(CustomTestCase):
    def setUp(self):
        self.collector, self.registry = _make_collector()

    def test_set_and_clear_gauges(self):
        self.collector.set_hicache_gauges(3, 2, 5, 4, 1, 1)
        self.assertEqual(_gauges(self.registry), (3, 2, 5, 4, 1, 1))

        self.collector.clear_hicache_gauges()
        self.assertEqual(_gauges(self.registry), (0,) * 6)

    def test_submitted_tokens(self):
        self.collector.log_backup_submitted_tokens(64)
        self.collector.log_backup_submitted_tokens(0)
        self.assertEqual(
            _sample(self.registry, "sglang:hicache_l3_backup_submitted_tokens_total"),
            64,
        )

    def test_failed_op_counts_kv_and_sidecar_pages_separately(self):
        self.collector.log_backup_failed_op(
            page_size=16,
            unwritten_pages=2,
            sidecar_unwritten_by_pool={"mamba": 1, "indexer": 3},
        )
        self.collector.log_backup_failed_op(
            page_size=16, unwritten_pages=0, sidecar_unwritten_by_pool={"mamba": 2}
        )

        self.assertEqual(
            _sample(self.registry, "sglang:hicache_l3_backup_failed_ops_total"), 2
        )
        self.assertEqual(
            _sample(self.registry, "sglang:hicache_l3_backup_unwritten_pages_total"),
            2,
        )
        self.assertEqual(
            _sample(self.registry, "sglang:hicache_l3_backup_unwritten_tokens_total"),
            32,
        )
        sidecar = "sglang:hicache_l3_backup_sidecar_unwritten_pages_total"
        self.assertEqual(_sample(self.registry, sidecar, pool="mamba"), 3)
        self.assertEqual(_sample(self.registry, sidecar, pool="indexer"), 3)


def _storage_operation(cls, num_pages, page_size, **kwargs):
    return cls(
        host_indices=torch.arange(num_pages * page_size),
        token_ids=list(range(num_pages * page_size)),
        hash_value=[f"h{i}" for i in range(num_pages)],
        **kwargs,
    )


def _run_backup_worker(controller, worker, operation):
    controller.storage_stop_event = threading.Event()
    controller.backup_queue = Queue()
    controller.ack_backup_queue = Queue()
    controller.backup_queue.put(operation)
    thread = threading.Thread(target=worker, args=(controller,), daemon=True)
    thread.start()
    try:
        return controller.ack_backup_queue.get(timeout=5)
    finally:
        controller.storage_stop_event.set()
        thread.join(timeout=5)


class TestBackupOutcome(CustomTestCase):
    """The backup worker records what a failed operation left unwritten."""

    page_size = 4

    def _base_controller(self, page_set_func, is_storage_writer=True):
        controller = HiCacheController.__new__(HiCacheController)
        controller.page_size = self.page_size
        controller.page_set_func = page_set_func
        controller.storage_config = SimpleNamespace(is_storage_writer=is_storage_writer)
        return controller

    def _hybrid_controller(self, page_set_func=None, is_storage_writer=True):
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.page_size = self.page_size
        controller.page_set_func = page_set_func
        controller.storage_config = SimpleNamespace(is_storage_writer=is_storage_writer)
        controller.storage_backend_type = "file"
        return controller

    def test_successful_backup_is_not_failed(self):
        controller = self._base_controller(mock.Mock(return_value=True))
        operation = _storage_operation(StorageOperation, 3, self.page_size)

        acked = _run_backup_worker(
            controller, HiCacheController.backup_thread_func, operation
        )

        self.assertIs(acked, operation)
        self.assertFalse(operation.failed)
        self.assertEqual(operation.unwritten_pages, 0)
        self.assertEqual(operation.completed_tokens, 3 * self.page_size)

    def test_backend_refusal_records_remaining_pages(self):
        controller = self._base_controller(mock.Mock(side_effect=[True, False]))
        operation = _storage_operation(StorageOperation, 4, self.page_size)

        with mock.patch("sglang.srt.managers.cache_controller.STORAGE_BATCH_SIZE", 1):
            HiCacheController._page_backup(controller, operation)

        self.assertTrue(operation.failed)
        self.assertEqual(operation.unwritten_pages, 3)
        self.assertEqual(operation.completed_tokens, self.page_size)

    def test_backend_exception_is_acked_with_remaining_pages(self):
        controller = self._base_controller(
            mock.Mock(side_effect=[True, RuntimeError("storage I/O error")])
        )
        operation = _storage_operation(StorageOperation, 4, self.page_size)

        with (
            mock.patch("sglang.srt.managers.cache_controller.STORAGE_BATCH_SIZE", 2),
            self.assertLogs(level="ERROR"),
        ):
            acked = _run_backup_worker(
                controller, HiCacheController.backup_thread_func, operation
            )

        self.assertIs(acked, operation)
        self.assertTrue(operation.failed)
        self.assertEqual(operation.unwritten_pages, 2)

    def test_hybrid_kv_refusal_records_remaining_pages(self):
        controller = self._hybrid_controller(mock.Mock(return_value=False))
        operation = _storage_operation(HybridStorageOperation, 2, self.page_size)

        controller._page_backup_kv_with_trace(operation)

        self.assertTrue(operation.failed)
        self.assertEqual(operation.unwritten_pages, 2)

    def test_partial_sidecar_write_records_unwritten_pages_by_pool(self):
        # A non-writer rank still backs up its own rank-sharded Mamba state.
        controller = self._hybrid_controller(is_storage_writer=False)
        controller.storage_backend = mock.Mock()
        controller.storage_backend.batch_set_v2.return_value = {
            PoolName.MAMBA.value: [True, False, False]
        }
        operation = _storage_operation(
            HybridStorageOperation,
            3,
            self.page_size,
            pool_transfers=[
                PoolTransfer(
                    name=PoolName.MAMBA,
                    host_indices=torch.arange(3),
                    keys=["m0", "m1", "m2"],
                )
            ],
        )

        controller._page_backup_with_stable_layout(operation)

        self.assertTrue(operation.failed)
        self.assertEqual(operation.sidecar_unwritten_by_pool, {"mamba": 2})
        # The primary KV pool is not this rank's to write.
        self.assertEqual(operation.unwritten_pages, 0)
        self.assertEqual(operation.completed_tokens, 0)

    def test_full_sidecar_write_is_not_failed(self):
        controller = self._hybrid_controller(is_storage_writer=False)
        controller.storage_backend = mock.Mock()
        controller.storage_backend.batch_set_v2.return_value = {
            PoolName.MAMBA.value: [True, True]
        }
        operation = _storage_operation(
            HybridStorageOperation,
            2,
            self.page_size,
            pool_transfers=[
                PoolTransfer(
                    name=PoolName.MAMBA,
                    host_indices=torch.arange(2),
                    keys=["m0", "m1"],
                )
            ],
        )

        controller._page_backup_with_stable_layout(operation)

        self.assertFalse(operation.failed)
        self.assertEqual(operation.sidecar_unwritten_by_pool, {})
        self.assertEqual(operation.completed_tokens, 2 * self.page_size)

    def test_exception_on_non_writer_rank_leaves_kv_pages_alone(self):
        controller = self._hybrid_controller(is_storage_writer=False)
        controller._page_backup = mock.Mock(side_effect=RuntimeError("sidecar error"))
        operation = _storage_operation(HybridStorageOperation, 2, self.page_size)

        with self.assertLogs(level="ERROR"):
            acked = _run_backup_worker(
                controller, HybridCacheController.backup_thread_func, operation
            )

        self.assertIs(acked, operation)
        self.assertTrue(operation.failed)
        self.assertEqual(operation.unwritten_pages, 0)


class _AliveThread:
    def __init__(self, alive=True):
        self.alive = alive

    def is_alive(self):
        return self.alive


def _make_cache():
    """A UnifiedRadixCache with just the state the storage metrics read."""
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.tree_core = mock.Mock()
    cache.enable_storage = True
    cache.enable_storage_metrics = True
    cache.storage_metrics_collector, registry = _make_collector()
    cache._hicache_gauge_last_state = None
    cache._hicache_gauge_last_ts = 0.0
    cache.host_memory_mode = "cache"
    cache.buffer_pipeline = None
    cache.ongoing_write_through = {}
    cache.ongoing_backup = {}
    cache.ongoing_prefetch = {}
    cache.cache_controller = SimpleNamespace(
        page_size=4,
        ack_write_queue=[],
        prefetch_hit_queue=Queue(),
        ack_prefetch_queue=Queue(),
        backup_queue=Queue(),
        ack_backup_queue=Queue(),
        host_mem_release_queue=Queue(),
        backup_thread=_AliveThread(),
    )
    return cache, registry


class TestUnifiedRadixCacheHiCacheGauges(CustomTestCase):
    def test_publishes_queue_depths(self):
        cache, registry = _make_cache()
        cc = cache.cache_controller
        cache.ongoing_write_through = {1: object(), 2: object()}
        cc.ack_write_queue.append(object())
        cache.ongoing_backup = {3: object()}
        for _ in range(4):
            cc.backup_queue.put(object())
        cc.ack_backup_queue.put(object())

        cache._update_hicache_gauges()

        self.assertEqual(_gauges(registry), (2, 1, 1, 4, 1, 1))

    def test_dead_backup_worker_reports_zero(self):
        cache, registry = _make_cache()
        cache.cache_controller.backup_thread = _AliveThread(alive=False)

        cache._update_hicache_gauges()

        self.assertEqual(
            registry.get_sample_value(_GAUGE_NAMES[-1], _LABELS),
            0,
        )

    def test_buffer_mode_reads_the_pipeline(self):
        cache, registry = _make_cache()
        cache.buffer_pipeline = SimpleNamespace(
            ongoing_write_through={-1: object()},
            ongoing_backup={7: object(), 8: object()},
        )

        cache._update_hicache_gauges()

        self.assertEqual(_gauges(registry)[:3], (1, 0, 2))

    def test_unchanged_state_is_republished_only_after_a_second(self):
        cache, _ = _make_cache()
        cache._update_hicache_gauges()

        with mock.patch.object(
            cache.storage_metrics_collector, "set_hicache_gauges"
        ) as set_gauges:
            cache._update_hicache_gauges()
            set_gauges.assert_not_called()

            cache.ongoing_backup = {1: object()}
            cache._update_hicache_gauges()
            set_gauges.assert_called_once()

            cache._hicache_gauge_last_ts -= 2.0
            cache._update_hicache_gauges()
            self.assertEqual(set_gauges.call_count, 2)

    def test_clear_resets_gauges_and_throttle(self):
        cache, registry = _make_cache()
        cache.ongoing_write_through = {1: object()}
        cache._update_hicache_gauges()

        cache.clear_hicache_gauges()

        self.assertEqual(_gauges(registry), (0,) * 6)
        self.assertIsNone(cache._hicache_gauge_last_state)
        # The next sample is published even though the state is unchanged.
        cache._update_hicache_gauges()
        self.assertEqual(_gauges(registry)[0], 1)

    def test_backup_submission_counts_kv_tokens(self):
        cache, registry = _make_cache()
        cache.hicache_storage_pass_prefix_keys = False
        cache.tree_core.build_storage_backup_spec.return_value = SimpleNamespace(
            host_value=torch.arange(12),
            hash_value=["h0", "h1", "h2"],
            token_ids=list(range(12)),
            prefix_keys=None,
            comp_xfers={},
        )
        cache._build_sidecar_transfers = mock.Mock(return_value=[])
        cache.inc_host_lock_ref = mock.Mock()
        cache.cache_controller.write_storage = mock.Mock(return_value=5)

        cache.write_backup_storage(node_id=1)

        self.assertIn(5, cache.ongoing_backup)
        self.assertEqual(
            _sample(registry, "sglang:hicache_l3_backup_submitted_tokens_total"), 12
        )

    def _drain(self, cache):
        cache._drain_storage_control_queues_impl(
            n_storage_hit=0,
            n_ack_prefetch=0,
            n_backup=None,
            n_release=0,
            extra_release_counts=None,
            log_metrics=True,
        )

    def test_drain_counts_failed_backups(self):
        cache, registry = _make_cache()
        cache.dec_host_lock_ref = mock.Mock()
        ok = _storage_operation(HybridStorageOperation, 2, 4)
        ok.completed_tokens = 8
        failed = _storage_operation(HybridStorageOperation, 3, 4)
        failed.completed_tokens = 4
        failed.failed = True
        failed.unwritten_pages = 2
        failed.sidecar_unwritten_by_pool = {"swa": 1}
        for operation in (ok, failed):
            cache.ongoing_backup[operation.id] = (operation.id, None)
            cache.cache_controller.ack_backup_queue.put(operation)

        self._drain(cache)

        self.assertEqual(cache.ongoing_backup, {})
        self.assertEqual(_sample(registry, "sglang:backuped_tokens_total"), 12)
        self.assertEqual(
            _sample(registry, "sglang:hicache_l3_backup_failed_ops_total"), 1
        )
        self.assertEqual(
            _sample(registry, "sglang:hicache_l3_backup_unwritten_pages_total"), 2
        )
        self.assertEqual(
            _sample(registry, "sglang:hicache_l3_backup_unwritten_tokens_total"), 8
        )
        self.assertEqual(
            _sample(
                registry,
                "sglang:hicache_l3_backup_sidecar_unwritten_pages_total",
                pool="swa",
            ),
            1,
        )

    def test_detach_clears_gauges(self):
        cache, registry = _make_cache()
        cache.ongoing_write_through = {1: object()}
        cache._update_hicache_gauges()
        self.assertEqual(_gauges(registry)[0], 1)
        cache.cache_controller.detach_storage_backend = mock.Mock()
        cache.drain_storage_control_queues_local = mock.Mock()
        attachment = StorageAttachment(cache)
        attachment._release_pending_storage_ops = mock.Mock()

        ok, message = attachment.detach()

        self.assertTrue(ok, message)
        self.assertFalse(cache.enable_storage_metrics)
        self.assertEqual(_gauges(registry), (0,) * 6)
        self.assertIsNone(cache._hicache_gauge_last_state)

    def test_detach_without_a_collector(self):
        cache, _ = _make_cache()
        cache.enable_storage_metrics = False
        cache.storage_metrics_collector = None
        cache.cache_controller.detach_storage_backend = mock.Mock()
        cache.drain_storage_control_queues_local = mock.Mock()
        attachment = StorageAttachment(cache)
        attachment._release_pending_storage_ops = mock.Mock()

        ok, message = attachment.detach()

        self.assertTrue(ok, message)


if __name__ == "__main__":
    unittest.main()
