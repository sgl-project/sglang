"""CPU checks for HiCache pending-work gauges and backup outcome metrics."""

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from functools import partial
from queue import Queue
from types import SimpleNamespace
from unittest import mock

import prometheus_client

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.observability.metrics_collector import StorageMetricsCollector


def _read_metric(metric, labels: dict):
    """Read the current value of a labeled prometheus metric."""
    return metric.labels(**labels)._value.get()


def _make_labels():
    return {
        "storage_backend": "test",
        "tp_rank": "0",
        "dp_rank": "0",
        "pp_rank": "0",
        "pp_size": "1",
        "attn_cp_rank": "0",
        "attn_cp_size": "1",
    }


class MetricTestCase(CustomTestCase):
    def setUp(self):
        super().setUp()
        registry = prometheus_client.CollectorRegistry()
        for name in ("Counter", "Gauge", "Histogram"):
            factory = partial(getattr(prometheus_client, name), registry=registry)
            self.enterContext(mock.patch.object(prometheus_client, name, factory))


class TestStorageMetricsCollectorGauges(MetricTestCase):
    def setUp(self):
        super().setUp()
        self.labels = _make_labels()
        self.collector = StorageMetricsCollector(labels=self.labels)

    def test_all_gauges_created(self):
        for attr in [
            "hicache_d2h_pending_ops",
            "hicache_d2h_ack_queue_depth",
            "hicache_l3_pending_ops",
            "hicache_l3_backup_queue_depth",
            "hicache_l3_ack_queue_depth",
            "hicache_l3_backup_worker_alive",
        ]:
            self.assertTrue(hasattr(self.collector, attr))

    def test_all_counters_created(self):
        for attr in [
            "hicache_l3_backup_submitted_tokens_total",
            "hicache_l3_backup_failed_ops_total",
            "hicache_l3_backup_unwritten_pages_total",
            "hicache_l3_backup_unwritten_tokens_total",
            "hicache_l3_backup_sidecar_unwritten_pages_total",
        ]:
            self.assertTrue(hasattr(self.collector, attr))

    def test_gauges_use_mostrecent_multiprocess_mode(self):
        """All 6 state gauges must use multiprocess_mode='mostrecent'."""
        for attr in [
            "hicache_d2h_pending_ops",
            "hicache_d2h_ack_queue_depth",
            "hicache_l3_pending_ops",
            "hicache_l3_backup_queue_depth",
            "hicache_l3_ack_queue_depth",
            "hicache_l3_backup_worker_alive",
        ]:
            gauge = getattr(self.collector, attr)
            self.assertEqual(
                gauge._multiprocess_mode,
                "mostrecent",
                f"{attr} should use mostrecent multiprocess_mode",
            )

    def test_set_hicache_gauges(self):
        self.collector.set_hicache_gauges(3, 2, 5, 4, 1, 1)
        L = self.labels
        self.assertEqual(_read_metric(self.collector.hicache_d2h_pending_ops, L), 3)
        self.assertEqual(_read_metric(self.collector.hicache_d2h_ack_queue_depth, L), 2)
        self.assertEqual(_read_metric(self.collector.hicache_l3_pending_ops, L), 5)
        self.assertEqual(
            _read_metric(self.collector.hicache_l3_backup_queue_depth, L), 4
        )
        self.assertEqual(_read_metric(self.collector.hicache_l3_ack_queue_depth, L), 1)
        self.assertEqual(
            _read_metric(self.collector.hicache_l3_backup_worker_alive, L), 1
        )

    def test_clear_hicache_gauges(self):
        self.collector.set_hicache_gauges(3, 2, 5, 4, 1, 1)
        self.collector.clear_hicache_gauges()
        L = self.labels
        self.assertEqual(_read_metric(self.collector.hicache_d2h_pending_ops, L), 0)
        self.assertEqual(
            _read_metric(self.collector.hicache_l3_backup_worker_alive, L), 0
        )

    def test_log_backup_submitted_tokens(self):
        self.collector.log_backup_submitted_tokens(100)
        L = self.labels
        self.assertEqual(
            _read_metric(self.collector.hicache_l3_backup_submitted_tokens_total, L),
            100,
        )
        self.collector.log_backup_submitted_tokens(0)
        self.assertEqual(
            _read_metric(self.collector.hicache_l3_backup_submitted_tokens_total, L),
            100,
        )

    def test_log_backup_failed_op_kv_only(self):
        page_size = 16
        unwritten_pages = 3
        self.collector.log_backup_failed_op(page_size, unwritten_pages)
        L = self.labels
        self.assertEqual(
            _read_metric(self.collector.hicache_l3_backup_failed_ops_total, L), 1
        )
        self.assertEqual(
            _read_metric(self.collector.hicache_l3_backup_unwritten_pages_total, L),
            unwritten_pages,
        )
        self.assertEqual(
            _read_metric(self.collector.hicache_l3_backup_unwritten_tokens_total, L),
            unwritten_pages * page_size,
        )

    def test_log_backup_failed_op_with_sidecar(self):
        """KV and sidecar unwritten are tracked separately."""
        page_size = 16
        kv_unwritten = 2
        sidecar = {PoolName.INDEXER: 3, PoolName.MAMBA: 1}
        self.collector.log_backup_failed_op(page_size, kv_unwritten, sidecar)
        L = self.labels
        # failed_ops increments once
        self.assertEqual(
            _read_metric(self.collector.hicache_l3_backup_failed_ops_total, L), 1
        )
        # KV unwritten pages/tokens are KV-only
        self.assertEqual(
            _read_metric(self.collector.hicache_l3_backup_unwritten_pages_total, L),
            kv_unwritten,
        )
        self.assertEqual(
            _read_metric(self.collector.hicache_l3_backup_unwritten_tokens_total, L),
            kv_unwritten * page_size,
        )
        # Sidecar unwritten pages tracked per-pool
        indexer_labels = dict(L)
        indexer_labels["pool_name"] = PoolName.INDEXER
        self.assertEqual(
            _read_metric(
                self.collector.hicache_l3_backup_sidecar_unwritten_pages_total,
                indexer_labels,
            ),
            3,
        )
        mamba_labels = dict(L)
        mamba_labels["pool_name"] = PoolName.MAMBA
        self.assertEqual(
            _read_metric(
                self.collector.hicache_l3_backup_sidecar_unwritten_pages_total,
                mamba_labels,
            ),
            1,
        )

    def test_log_backup_failed_op_sidecar_only(self):
        """Operation with only sidecar failure (KV succeeded)."""
        page_size = 16
        sidecar = {PoolName.INDEXER: 2}
        self.collector.log_backup_failed_op(page_size, 0, sidecar)
        L = self.labels
        self.assertEqual(
            _read_metric(self.collector.hicache_l3_backup_failed_ops_total, L), 1
        )
        # KV unwritten is 0
        self.assertEqual(
            _read_metric(self.collector.hicache_l3_backup_unwritten_pages_total, L), 0
        )
        self.assertEqual(
            _read_metric(self.collector.hicache_l3_backup_unwritten_tokens_total, L), 0
        )
        # Sidecar counter has the pages
        pool_labels = dict(L)
        pool_labels["pool_name"] = PoolName.INDEXER
        self.assertEqual(
            _read_metric(
                self.collector.hicache_l3_backup_sidecar_unwritten_pages_total,
                pool_labels,
            ),
            2,
        )


class TestUpdateHicacheGaugesProduction(MetricTestCase):
    """Test _update_hicache_gauges on a real UnifiedRadixCache instance.

    We create a UnifiedRadixCache via __new__ (bypassing __init__) and stub
    just enough attributes to call the production _update_hicache_gauges
    method. This catches regressions if the production method changes.
    (main-0923 port: HiRadixCache no longer exists; the detach cases live in
    TestUnifiedCacheDetachClearsGauges.)
    """

    def _make_partial_cache(self):
        """Create a UnifiedRadixCache instance without __init__."""
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        # enable_storage is a property backed by tree_core.
        cache.tree_core = mock.MagicMock()
        cache.buffer_pipeline = None
        labels = _make_labels()
        cache.enable_storage_metrics = True
        cache.enable_storage = True
        cache.storage_metrics_collector = StorageMetricsCollector(labels=labels)
        cache.ongoing_write_through = {}
        cache.ongoing_backup = {}

        # Create a fake controller with the needed queues/threads
        cc = mock.MagicMock()
        cc.ack_write_queue = []
        cc.backup_queue = Queue()
        cc.ack_backup_queue = Queue()
        cache.cache_controller = cc
        return cache, labels

    def test_worker_alive_when_thread_missing(self):
        """When backup_thread attr doesn't exist, gauge reports 0."""
        cache, labels = self._make_partial_cache()
        del cache.cache_controller.backup_thread

        cache._update_hicache_gauges()
        self.assertEqual(
            _read_metric(
                cache.storage_metrics_collector.hicache_l3_backup_worker_alive,
                labels,
            ),
            0,
        )

    def test_worker_alive_when_thread_none(self):
        cache, labels = self._make_partial_cache()
        cache.cache_controller.backup_thread = None

        cache._update_hicache_gauges()
        self.assertEqual(
            _read_metric(
                cache.storage_metrics_collector.hicache_l3_backup_worker_alive,
                labels,
            ),
            0,
        )

    def test_worker_alive_when_thread_alive(self):
        cache, labels = self._make_partial_cache()
        thread = mock.MagicMock()
        thread.is_alive.return_value = True
        cache.cache_controller.backup_thread = thread

        cache._update_hicache_gauges()
        self.assertEqual(
            _read_metric(
                cache.storage_metrics_collector.hicache_l3_backup_worker_alive,
                labels,
            ),
            1,
        )

    def test_worker_alive_when_thread_dead(self):
        cache, labels = self._make_partial_cache()
        thread = mock.MagicMock()
        thread.is_alive.return_value = False
        cache.cache_controller.backup_thread = thread

        cache._update_hicache_gauges()
        self.assertEqual(
            _read_metric(
                cache.storage_metrics_collector.hicache_l3_backup_worker_alive,
                labels,
            ),
            0,
        )

    def test_gauge_throttle_no_change_within_one_second(self):
        """When state is unchanged and <1s elapsed, gauge should not be re-set."""
        cache, labels = self._make_partial_cache()
        cache._update_hicache_gauges()

        with mock.patch.object(
            cache.storage_metrics_collector, "set_hicache_gauges"
        ) as mock_set:
            cache._update_hicache_gauges()
            mock_set.assert_not_called()

    def test_gauge_updates_on_state_change(self):
        """When state changes, gauge is immediately updated."""
        cache, labels = self._make_partial_cache()
        cache._update_hicache_gauges()  # initial call, sets state

        # Change state
        cache.ongoing_backup = {1: "fake_node"}
        cache._update_hicache_gauges()

        self.assertEqual(
            _read_metric(
                cache.storage_metrics_collector.hicache_l3_pending_ops, labels
            ),
            1,
        )

    def test_gauge_periodic_refresh_after_one_second(self):
        """Even with no state change, after 1s gauges are refreshed."""
        cache, labels = self._make_partial_cache()
        cache._update_hicache_gauges()

        # Simulate >1s elapsed by backdating the timestamp
        cache._hicache_gauge_last_ts -= 2.0

        with mock.patch.object(
            cache.storage_metrics_collector, "set_hicache_gauges"
        ) as mock_set:
            cache._update_hicache_gauges()
            mock_set.assert_called_once()

    def test_reset_clears_gauge_cache(self):
        """After calling production UnifiedRadixCache.reset(), gauge cache is cleared."""

        cache, labels = self._make_partial_cache()
        cache._update_hicache_gauges()

        # Set up attributes needed by reset()
        cache.linker = None
        cache._pending_restore_backups = []
        cache.session_refs = mock.MagicMock()
        cache.session = mock.MagicMock()

        cache.reset()

        # Gauge cache should be cleared
        self.assertIsNone(cache._hicache_gauge_last_state)
        self.assertEqual(cache._hicache_gauge_last_ts, 0.0)

        # Next _update_hicache_gauges should publish (state is None != current)
        with mock.patch.object(
            cache.storage_metrics_collector, "set_hicache_gauges"
        ) as mock_set:
            cache._update_hicache_gauges()
            mock_set.assert_called_once()

    def test_buffer_mode_l3_pending_reads_the_pipeline(self):
        """Buffer mode tracks host->L3 writes on the pipeline, not ongoing_backup."""
        cache, labels = self._make_partial_cache()
        cache.buffer_pipeline = mock.MagicMock()
        cache.buffer_pipeline.ongoing_backup = {7: object(), 8: object()}

        cache._update_hicache_gauges()

        self.assertEqual(
            _read_metric(
                cache.storage_metrics_collector.hicache_l3_pending_ops, labels
            ),
            2,
        )

    def test_update_hicache_gauges_when_metrics_disabled(self):
        """_update_hicache_gauges is a no-op when metrics disabled."""
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.enable_storage_metrics = False

        # Should not raise even though collector/controller don't exist
        cache._update_hicache_gauges()


class TestBackupAckMetrics(CustomTestCase):
    def test_ack_drain_accepts_legacy_and_detailed_failure_outcomes(self):
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        for detailed in (False, True):
            with self.subTest(detailed=detailed):
                operation = SimpleNamespace(id=7, completed_tokens=4)
                if detailed:
                    operation.failed = True
                    operation.unwritten_pages = 2
                    operation.sidecar_unwritten_by_pool = {"indexer": 1}
                queues = {
                    name: Queue()
                    for name in (
                        "prefetch_hit_queue",
                        "ack_prefetch_queue",
                        "ack_backup_queue",
                        "host_mem_release_queue",
                    )
                }
                queues["ack_backup_queue"].put(operation)
                controller = SimpleNamespace(
                    **queues, page_size=4, extra_pool_host_mem_release_queues={}
                )
                cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
                cache.buffer_pipeline = None
                cache.host_memory_mode = "cache"
                cache.cache_controller = controller
                cache.enable_storage_metrics = True
                cache.storage_metrics_collector = mock.Mock()
                cache.ongoing_backup = {7: (11, None)}
                cache.dec_host_lock_ref = mock.Mock()
                cache._drain_storage_control_queues_impl(
                    n_storage_hit=0,
                    n_ack_prefetch=0,
                    n_backup=1,
                    n_release=0,
                    extra_release_counts={},
                    log_metrics=True,
                )
                cache.storage_metrics_collector.log_backuped_tokens.assert_called_once_with(
                    4
                )
                if detailed:
                    cache.storage_metrics_collector.log_backup_failed_op.assert_called_once_with(
                        4, 2, {"indexer": 1}
                    )
                else:
                    cache.storage_metrics_collector.log_backup_failed_op.assert_not_called()
                self.assertEqual(cache.ongoing_backup, {})


class TestUnifiedCacheDetachClearsGauges(MetricTestCase):
    """UnifiedRadixCache's detach path must clear the HiCache gauges.

    A gauge keeps its last published value until something overwrites it, so a
    detach that skips the clear leaves a permanently non-zero queue depth and a
    worker_alive=1 on the dashboard of a backend that is no longer attached.
    reset() and detach() are two separate paths; this covers detach().
    """

    def _make_cache_and_attachment(self):
        from sglang.srt.mem_cache.unified_cache.storage_attachment import (
            StorageAttachment,
        )
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        labels = _make_labels()
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        # enable_storage is a property backed by tree_core.
        cache.tree_core = mock.MagicMock()
        cache.enable_storage = True
        cache.enable_storage_metrics = True
        cache.storage_metrics_collector = StorageMetricsCollector(labels=labels)
        # Non-empty so the published gauge is non-zero before the detach.
        cache.ongoing_write_through = {"req-a": object()}
        cache.ongoing_backup = {}
        cc = mock.MagicMock()
        cc.ack_write_queue = []
        cc.backup_queue = Queue()
        cc.ack_backup_queue = Queue()
        cache.cache_controller = cc
        cache.drain_storage_control_queues_local = mock.MagicMock()
        attachment = StorageAttachment(cache)
        attachment._release_pending_storage_ops = mock.MagicMock()
        return cache, attachment, labels

    def test_detach_clears_gauges_and_the_throttle_cache(self):
        cache, attachment, labels = self._make_cache_and_attachment()
        collector = cache.storage_metrics_collector
        cache._update_hicache_gauges()
        self.assertEqual(_read_metric(collector.hicache_d2h_pending_ops, labels), 1)

        ok, _ = attachment.detach()

        self.assertTrue(ok)
        self.assertFalse(cache.enable_storage)
        self.assertFalse(cache.enable_storage_metrics)
        self.assertEqual(_read_metric(collector.hicache_d2h_pending_ops, labels), 0)
        self.assertEqual(
            _read_metric(collector.hicache_l3_backup_worker_alive, labels), 0
        )
        # Throttle cache too: a later re-attach must publish its first sample
        # instead of matching the stale state and staying silent.
        self.assertIsNone(cache._hicache_gauge_last_state)
        self.assertEqual(cache._hicache_gauge_last_ts, 0.0)

    def test_detach_without_a_collector_does_not_raise(self):
        cache, attachment, _ = self._make_cache_and_attachment()
        cache.enable_storage_metrics = False
        cache.storage_metrics_collector = None

        ok, _ = attachment.detach()

        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
