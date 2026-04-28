import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import GetLoadsReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.observability.metrics_collector import (
    SchedulerMetricsCollector,
    SchedulerStats,
)
from sglang.srt.observability.scheduler_metrics_mixin import SchedulerMetricsMixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class _FakeMetric:
    def __init__(self, *args, **kwargs):
        self.name = kwargs.get("name")
        self.value = None
        self.observations = []
        self.increments = []

    def labels(self, **kwargs):
        return self

    def set(self, value):
        self.value = value

    def observe(self, value):
        self.observations.append(value)

    def inc(self, value=1):
        self.increments.append(value)


def _scheduler_labels():
    return {
        "model_name": "test",
        "engine_type": "unified",
        "tp_rank": 0,
        "pp_rank": 0,
        "moe_ep_rank": 0,
    }


def _server_args():
    return SimpleNamespace(
        prefill_delayer_max_delay_passes=8,
        prefill_delayer_forward_passes_buckets=None,
        prefill_delayer_wait_seconds_buckets=None,
    )


def _make_collector():
    return SchedulerMetricsCollector(
        labels=_scheduler_labels(),
        server_args=_server_args(),
    )


def _adaptive_metrics():
    return {
        "enabled": 1,
        "current_steps": 3,
        "previous_steps": 1,
        "num_tier_switches": 2,
        "ema_accept_len": 2.25,
        "last_batch_accept_len": 2.5,
        "wasted_draft_ratio": 0.5,
    }


def _make_scheduler_stub():
    return SimpleNamespace(
        running_batch=SimpleNamespace(reqs=[]),
        waiting_queue=[],
        disaggregation_mode=DisaggregationMode.NULL,
        get_pool_stats=lambda: SimpleNamespace(get_kv_token_stats=lambda: (8, 0.125)),
        _get_adaptive_spec_metrics=_adaptive_metrics,
        spec_algorithm=SimpleNamespace(is_none=lambda: False),
        spec_total_num_forward_ct=0,
        spec_total_num_accepted_tokens=0,
        stats=SchedulerStats(spec_accept_rate=0.75),
        dp_rank=0,
        max_total_num_tokens=1024,
        max_running_requests=16,
        last_gen_throughput=0.0,
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                weight_load_mem_usage=0.0,
                graph_mem_usage=0.0,
            )
        ),
        token_to_kv_pool_allocator=SimpleNamespace(
            get_kvcache=lambda: SimpleNamespace(mem_usage=0.0)
        ),
    )


@patch("prometheus_client.Summary", _FakeMetric)
@patch("prometheus_client.Histogram", _FakeMetric)
@patch("prometheus_client.Counter", _FakeMetric)
@patch("prometheus_client.Gauge", _FakeMetric)
class TestAdaptiveSpecMetricsCollector(CustomTestCase):
    def test_log_stats_emits_adaptive_spec_gauges(self):
        collector = _make_collector()
        stats = SchedulerStats(
            adaptive_spec_enabled=1,
            adaptive_spec_current_steps=3,
            adaptive_spec_previous_steps=1,
            adaptive_spec_num_tier_switches=2,
            adaptive_spec_ema_accept_length=2.25,
            adaptive_spec_last_batch_accept_length=2.5,
            adaptive_spec_wasted_draft_ratio=0.5,
        )

        collector.log_stats(stats)

        self.assertEqual(collector.adaptive_spec_enabled.value, 1)
        self.assertEqual(collector.adaptive_spec_current_steps.value, 3)
        self.assertEqual(collector.adaptive_spec_previous_steps.value, 1)
        self.assertEqual(collector.adaptive_spec_num_tier_switches.value, 2)
        self.assertEqual(collector.adaptive_spec_ema_accept_length.value, 2.25)
        self.assertEqual(
            collector.adaptive_spec_last_batch_accept_length.value,
            2.5,
        )
        self.assertEqual(collector.adaptive_spec_wasted_draft_ratio.value, 0.5)

    def test_log_stats_marks_adaptive_spec_disabled_by_default(self):
        collector = _make_collector()

        collector.log_stats(SchedulerStats())

        self.assertEqual(collector.adaptive_spec_enabled.value, 0)
        self.assertEqual(collector.adaptive_spec_current_steps.value, 0)
        self.assertEqual(collector.adaptive_spec_wasted_draft_ratio.value, 0.0)

    def test_get_loads_includes_adaptive_spec_wasted_draft_ratio(self):
        scheduler = _make_scheduler_stub()

        loads = SchedulerMetricsMixin.get_loads(
            scheduler, GetLoadsReqInput(include=["spec"])
        )

        self.assertIsNotNone(loads.speculative)
        self.assertEqual(loads.speculative.adaptive_enabled, 1)
        self.assertEqual(loads.speculative.adaptive_wasted_draft_ratio, 0.5)

    @patch("sglang.srt.managers.scheduler.get_global_server_args")
    def test_internal_state_includes_adaptive_spec_wasted_draft_ratio(
        self, get_global_server_args
    ):
        scheduler = _make_scheduler_stub()
        get_global_server_args.return_value = SimpleNamespace(model_config=object())

        output = Scheduler.get_internal_state(scheduler, recv_req=None)

        self.assertEqual(output.internal_state["adaptive_spec_enabled"], 1)
        self.assertEqual(output.internal_state["adaptive_spec_wasted_draft_ratio"], 0.5)


if __name__ == "__main__":
    unittest.main()
