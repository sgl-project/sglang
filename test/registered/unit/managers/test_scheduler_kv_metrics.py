import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    SchedulerMetricsReporter,
)
from sglang.srt.managers.scheduler_components.pool_stats_observer import PoolStats
from sglang.srt.observability.metrics_collector import SchedulerStats

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _FakeBatch:
    def __init__(
        self,
        initial_size: int,
        filtered_size: int,
        *,
        decode_mem_available: bool = True,
        retracted_size: int | None = None,
    ):
        self.reqs = [object() for _ in range(initial_size)]
        self.filtered_size = filtered_size
        self.decode_mem_available = decode_mem_available
        self.retracted_size = retracted_size
        self.batch_is_full = True
        self.prepared_for_decode = False

    def batch_size(self):
        return len(self.reqs)

    def filter_batch(self):
        self.reqs = self.reqs[: self.filtered_size]

    def is_empty(self):
        return not self.reqs

    def check_decode_mem(self):
        return self.decode_mem_available

    def retract_decode(self, _server_args):
        retracted_reqs = self.reqs[self.retracted_size :]
        self.reqs = self.reqs[: self.retracted_size]
        return retracted_reqs, 0.5, []

    def prepare_for_decode(self):
        self.prepared_for_decode = True


class TestSchedulerKVMetrics(unittest.TestCase):
    def setUp(self):
        self.scheduler = Scheduler.__new__(Scheduler)
        self.scheduler.forward_ct = 1
        self.scheduler.new_token_ratio_tracker = SimpleNamespace(
            current=1.0, decay_step=MagicMock()
        )
        self.scheduler.server_args = SimpleNamespace()
        self.scheduler.tree_cache = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(mamba_allocator=None)
        )
        self.scheduler.token_to_kv_pool_allocator = SimpleNamespace(
            available_size=MagicMock(side_effect=[0, 1])
        )
        self.scheduler._add_request_to_queue = MagicMock()

        self.reporter = SchedulerMetricsReporter.__new__(SchedulerMetricsReporter)
        self.reporter.scheduler = self.scheduler
        self.reporter.current_scheduler_metrics_enabled = True
        self.reporter.enable_kv_cache_events = True
        self.reporter.enable_metrics = False
        self.reporter.stats = SchedulerStats(token_usage=0.9)
        self.scheduler.metrics_reporter = self.reporter

        pool_stats = PoolStats(
            full_num_used=10,
            full_token_usage=0.1,
            full_available_size=90,
            full_evictable_size=0,
        )
        self.scheduler.pool_stats_observer = MagicMock()
        self.scheduler.pool_stats_observer.get_pool_stats.return_value = pool_stats

        self.emitted_usage = []
        self.scheduler.kv_events_publisher = SimpleNamespace(
            emit_kv_metrics=MagicMock(
                side_effect=lambda: self.emitted_usage.append(
                    self.reporter.stats.token_usage
                )
            )
        )

    def update_running_batch(self, batch):
        with patch("sglang.srt.managers.scheduler.TEST_RETRACT", False):
            return Scheduler.update_running_batch(self.scheduler, batch)

    def test_partial_batch_shrink_emits_fresh_kv_metrics(self):
        batch = _FakeBatch(initial_size=2, filtered_size=1)

        result = self.update_running_batch(batch)

        self.assertIs(result, batch)
        self.assertEqual(self.emitted_usage, [0.1])
        self.scheduler.pool_stats_observer.get_pool_stats.assert_called_once_with()
        self.scheduler.kv_events_publisher.emit_kv_metrics.assert_called_once_with()
        self.assertTrue(batch.prepared_for_decode)

    def test_full_batch_shrink_emits_fresh_kv_metrics(self):
        batch = _FakeBatch(initial_size=2, filtered_size=0)

        result = self.update_running_batch(batch)

        self.assertIs(result, batch)
        self.assertEqual(self.emitted_usage, [0.1])
        self.scheduler.pool_stats_observer.get_pool_stats.assert_called_once_with()
        self.scheduler.kv_events_publisher.emit_kv_metrics.assert_called_once_with()
        self.assertFalse(batch.prepared_for_decode)

    def test_unchanged_batch_does_not_emit_kv_metrics(self):
        batch = _FakeBatch(initial_size=2, filtered_size=2)

        result = self.update_running_batch(batch)

        self.assertIs(result, batch)
        self.assertEqual(self.emitted_usage, [])
        self.scheduler.pool_stats_observer.get_pool_stats.assert_not_called()
        self.scheduler.kv_events_publisher.emit_kv_metrics.assert_not_called()
        self.assertTrue(batch.prepared_for_decode)

    def test_retraction_emits_fresh_kv_metrics(self):
        batch = _FakeBatch(
            initial_size=2,
            filtered_size=2,
            decode_mem_available=False,
            retracted_size=1,
        )

        result = self.update_running_batch(batch)

        self.assertIs(result, batch)
        self.assertEqual(self.emitted_usage, [0.1])
        self.scheduler.pool_stats_observer.get_pool_stats.assert_called_once_with()
        self.scheduler.kv_events_publisher.emit_kv_metrics.assert_called_once_with()
        self.scheduler._add_request_to_queue.assert_called_once()
        self.assertTrue(batch.prepared_for_decode)

    def test_batch_shrink_without_kv_events_does_not_sample_pool(self):
        self.reporter.enable_kv_cache_events = False
        batch = _FakeBatch(initial_size=2, filtered_size=0)

        self.update_running_batch(batch)

        self.scheduler.pool_stats_observer.get_pool_stats.assert_not_called()
        self.scheduler.kv_events_publisher.emit_kv_metrics.assert_not_called()

    def test_batch_shrink_without_scheduler_metrics_does_not_sample_pool(self):
        self.reporter.current_scheduler_metrics_enabled = False
        batch = _FakeBatch(initial_size=2, filtered_size=0)

        self.update_running_batch(batch)

        self.scheduler.pool_stats_observer.get_pool_stats.assert_not_called()
        self.scheduler.kv_events_publisher.emit_kv_metrics.assert_not_called()


if __name__ == "__main__":
    unittest.main()
