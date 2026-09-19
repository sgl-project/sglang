"""Pure-CPU unit tests for the queue-depth gauges on the scheduler metrics path.

``sglang:prefill_queue_depth`` / ``sglang:decode_queue_depth`` split the single
non-PD ``waiting_queue`` by ``Req.is_retracted`` (computed inside the existing
``QueueCount.from_reqs`` pass), and ``sglang:num_prefill_inflight_reqs`` reflects
``scheduler.chunked_req``.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import types
import unittest
from unittest.mock import patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    SchedulerMetricsReporter,
)
from sglang.srt.observability.metrics_collector import (
    QueueCount,
    SchedulerMetricsCollector,
    SchedulerMetricsCollectorContext,
    SchedulerStats,
)
from sglang.test.test_utils import CustomTestCase


class _FakeReq:
    def __init__(self, is_retracted: bool = False, priority=None, retracted_stain=False):
        self.is_retracted = is_retracted
        self.priority = priority
        self.retracted_stain = retracted_stain


class _BoundRecordingMetric:
    def __init__(self, metric, labels):
        self.metric = metric
        self.labels = labels

    def set(self, value):
        self.metric.values[tuple(sorted(self.labels.items()))] = value

    def inc(self, value=1):
        pass

    def observe(self, value):
        pass


class _RecordingMetric:
    def __init__(self, *args, name=None, labelnames=(), **kwargs):
        self.name = name if name is not None else args[0]
        self.labelnames = tuple(labelnames)
        self.values = {}

    def labels(self, *values, **labels):
        if values:
            labels = dict(zip(self.labelnames, values, strict=True))
        return _BoundRecordingMetric(self, labels)


def _make_reporter(scheduler) -> SchedulerMetricsReporter:
    context = SchedulerMetricsCollectorContext(
        enable_metrics=False,
        is_stats_logging_rank=True,
        current_scheduler_metrics_enabled=False,
        enable_kv_cache_events=False,
        collector=None,
    )
    with patch.object(SchedulerMetricsReporter, "__init__", return_value=None):
        reporter = SchedulerMetricsReporter()
    reporter.scheduler = scheduler
    reporter.metrics_collector_context = context
    reporter.metrics_collector = None
    reporter.stats = SchedulerStats()
    return reporter


class TestQueueCountRetracted(CustomTestCase):
    def test_counts_retracted_in_same_pass(self):
        reqs = [_FakeReq(), _FakeReq(is_retracted=True), _FakeReq()]
        qc = QueueCount.from_reqs(reqs, count_retracted=True)
        self.assertEqual(qc.total, 3)
        self.assertEqual(qc.num_retracted, 1)
        self.assertIsNone(qc.by_priority)

    def test_priority_breakdown_and_retracted_together(self):
        reqs = [
            _FakeReq(priority=0),
            _FakeReq(is_retracted=True, priority=1),
            _FakeReq(priority=1),
        ]
        qc = QueueCount.from_reqs(
            reqs, enable_priority_scheduling=True, count_retracted=True
        )
        self.assertEqual(qc.by_priority, {0: 1, 1: 2})
        self.assertEqual(qc.num_retracted, 1)

    def test_default_does_not_count_retracted(self):
        qc = QueueCount.from_reqs([_FakeReq(is_retracted=True)])
        self.assertEqual((qc.total, qc.num_retracted), (1, 0))

    def test_empty_queue(self):
        qc = QueueCount.from_reqs([], count_retracted=True)
        self.assertEqual((qc.total, qc.num_retracted), (0, 0))


class TestQueueDepths(CustomTestCase):
    def _scheduler(self, waiting, chunked_req=None, mode=DisaggregationMode.NULL):
        return types.SimpleNamespace(
            waiting_queue=waiting,
            chunked_req=chunked_req,
            disaggregation_mode=mode,
        )

    def test_non_pd_split_by_retraction(self):
        waiting = [_FakeReq(), _FakeReq(is_retracted=True), _FakeReq(), _FakeReq()]
        reporter = _make_reporter(self._scheduler(waiting))
        reporter.stats.num_queue_reqs = QueueCount.from_reqs(
            waiting, count_retracted=True
        )
        reporter._update_queue_depths()
        self.assertEqual(reporter.stats.prefill_queue_depth, 3)
        self.assertEqual(reporter.stats.decode_queue_depth, 1)
        self.assertEqual(
            reporter.stats.prefill_queue_depth + reporter.stats.decode_queue_depth,
            reporter.stats.num_queue_reqs.total,
        )
        self.assertEqual(reporter.stats.num_prefill_inflight_reqs, 0)

    def test_chunked_prefill_inflight(self):
        reporter = _make_reporter(self._scheduler([], chunked_req=_FakeReq()))
        reporter.stats.num_queue_reqs = QueueCount.from_reqs([], count_retracted=True)
        reporter._update_queue_depths()
        self.assertEqual(reporter.stats.prefill_queue_depth, 0)
        self.assertEqual(reporter.stats.decode_queue_depth, 0)
        self.assertEqual(reporter.stats.num_prefill_inflight_reqs, 1)

    def test_pd_prefill_engine(self):
        scheduler = self._scheduler([_FakeReq()], mode=DisaggregationMode.PREFILL)
        scheduler.disagg_prefill_bootstrap_queue = types.SimpleNamespace(queue=[1, 2])
        # Prefill-completed KV transfers are excluded: already prefilled.
        scheduler.disagg_prefill_inflight_queue = [1]
        reporter = _make_reporter(scheduler)
        reporter._update_queue_depths()
        self.assertEqual(reporter.stats.prefill_queue_depth, 3)
        self.assertEqual(reporter.stats.decode_queue_depth, 0)

    def test_pd_decode_engine(self):
        reboot_waiting = _FakeReq()
        reboot_waiting.pd_rebootstrap_in_progress = True
        both_markers = _FakeReq(retracted_stain=True)
        both_markers.pd_rebootstrap_in_progress = True
        scheduler = self._scheduler(
            [_FakeReq(), _FakeReq(retracted_stain=True), reboot_waiting, both_markers],
            mode=DisaggregationMode.DECODE,
        )
        fresh = types.SimpleNamespace(is_rebootstrap=False)
        reboot = types.SimpleNamespace(is_rebootstrap=True)
        # Retracted, rebootstrap-held, in-flight rebootstraps, and restored
        # retractions still in waiting_queue count (both markers -> once);
        # ordinary first-decode handoffs in prealloc/transfer/waiting do not.
        scheduler.disagg_decode_prealloc_queue = types.SimpleNamespace(
            queue=[fresh, reboot], retracted_queue=[1, 2, 3, 4], held_rebootstrap_reqs=[1]
        )
        scheduler.disagg_decode_transfer_queue = types.SimpleNamespace(
            queue=[fresh, reboot]
        )
        reporter = _make_reporter(scheduler)
        reporter._update_queue_depths()
        self.assertEqual(reporter.stats.prefill_queue_depth, 0)
        self.assertEqual(reporter.stats.decode_queue_depth, 10)


class _RecordingCollector(SchedulerMetricsCollector):
    _counter_cls = _RecordingMetric
    _gauge_cls = _RecordingMetric
    _histogram_cls = _RecordingMetric
    _summary_cls = _RecordingMetric


class TestCollectorGauges(CustomTestCase):
    LABELS = {
        "model_name": "m",
        "engine_type": "unified",
        "tp_rank": 0,
        "pp_rank": 0,
        "moe_ep_rank": 0,
    }

    # Built once: the collector also registers non-DI'd metrics (GaugeHistogram)
    # on the global prometheus registry, which rejects duplicates.
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Upstream reads prefill-delayer buckets from the runtime context
        # (get_schedule()) and gates the EPLB balancedness summary on
        # exports_expert_balancedness_to_prometheus(); neither is published in
        # a pure unit test, so stub both at the module import site.
        schedule = types.SimpleNamespace(
            prefill_delayer_max_delay_passes=30,
            prefill_delayer_forward_passes_buckets=None,
            prefill_delayer_wait_seconds_buckets=None,
        )
        with (
            patch(
                "sglang.srt.observability.metrics_collector.get_schedule",
                return_value=schedule,
            ),
            patch(
                "sglang.srt.observability.metrics_collector.exports_expert_balancedness_to_prometheus",
                return_value=False,
            ),
        ):
            cls.collector = _RecordingCollector(
                labels=dict(cls.LABELS), server_args=types.SimpleNamespace()
            )

    def _collector(self) -> _RecordingCollector:
        for gauge in (
            self.collector.prefill_queue_depth,
            self.collector.decode_queue_depth,
            self.collector.num_prefill_inflight_reqs,
        ):
            gauge.values.clear()
        return self.collector

    def _value(self, gauge, **extra):
        return gauge.values[tuple(sorted({**self.LABELS, **extra}.items()))]

    def test_gauge_names(self):
        c = self._collector()
        self.assertEqual(c.prefill_queue_depth.name, "sglang:prefill_queue_depth")
        self.assertEqual(c.decode_queue_depth.name, "sglang:decode_queue_depth")
        self.assertEqual(
            c.num_prefill_inflight_reqs.name, "sglang:num_prefill_inflight_reqs"
        )

    def test_log_stats_emits_queue_depths(self):
        c = self._collector()
        stats = SchedulerStats(
            prefill_queue_depth=3, decode_queue_depth=1, num_prefill_inflight_reqs=1
        )
        c.log_stats(stats)
        self.assertEqual(self._value(c.prefill_queue_depth), 3)
        self.assertEqual(self._value(c.decode_queue_depth), 1)
        self.assertEqual(self._value(c.num_prefill_inflight_reqs), 1)


if __name__ == "__main__":
    unittest.main()
