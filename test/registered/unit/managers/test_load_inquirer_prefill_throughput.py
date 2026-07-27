"""Unit coverage for scheduler-side Prefill throughput load snapshots."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.load_inquirer import (
    SchedulerLoadInquirer,
)

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def _new_load_inquirer(
    *,
    mode: DisaggregationMode,
    fully_idle: bool,
    prefill_throughput: float,
) -> SchedulerLoadInquirer:
    """Build a minimal scheduler load inquirer for throughput tests.

    Args:
        mode: Engine disaggregation role under test.
        fully_idle: Value returned by the scheduler idle predicate.
        prefill_throughput: Existing metrics-reporter throughput value.

    Returns:
        An inquirer with deterministic empty queues and load statistics.
    """
    empty_queue = SimpleNamespace(queue=[], retracted_queue=[])
    stats = SimpleNamespace(
        gen_throughput=0.0,
        cache_hit_rate=0.0,
        utilization=0.0,
        spec_accept_rate=0.0,
        num_grammar_queue_reqs=0,
        num_paused_reqs=0,
        num_retracted_reqs=0,
        kv_transfer_speed_gb_s=0.0,
        kv_transfer_latency_ms=0.0,
    )
    return SchedulerLoadInquirer(
        disaggregation_mode=mode,
        ps=SimpleNamespace(dp_rank=0),
        server_args=SimpleNamespace(enable_lora=False),
        max_total_num_tokens=4096,
        max_running_requests=128,
        pool_stats_observer=SimpleNamespace(
            get_pool_stats=lambda: SimpleNamespace(get_kv_token_stats=lambda: (0, 0.0))
        ),
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                weight_load_mem_usage=0.0,
                graph_mem_usage=0.0,
            )
        ),
        token_to_kv_pool_allocator=SimpleNamespace(
            get_kvcache=lambda: SimpleNamespace(mem_usage=0.0)
        ),
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        get_running_batch=lambda: SimpleNamespace(reqs=[]),
        get_waiting_queue=lambda: [],
        get_stats=lambda: stats,
        get_prefill_throughput=lambda: prefill_throughput,
        is_fully_idle=lambda: fully_idle,
        get_chunked_req=lambda: None,
        get_disagg_prefill_bootstrap_queue=lambda: empty_queue,
        get_disagg_prefill_inflight_queue=lambda: [],
        get_disagg_decode_prealloc_queue=lambda: empty_queue,
        get_disagg_decode_transfer_queue=lambda: empty_queue,
        get_spec_total_num_accept_tokens=lambda: 0,
        get_spec_total_num_forward_ct=lambda: 0,
    )


class TestPrefillThroughputSnapshot(CustomTestCase):
    """Verify Prefill throughput mode and idle gating."""

    def test_only_active_pd_prefill_exposes_throughput(self) -> None:
        """Report the rounded value only for a non-idle PD Prefill Engine.

        Returns:
            None.
        """
        cases = (
            (DisaggregationMode.PREFILL, False, 123.46),
            (DisaggregationMode.PREFILL, True, 0.0),
            (DisaggregationMode.NULL, False, 0.0),
            (DisaggregationMode.DECODE, False, 0.0),
        )

        for mode, fully_idle, expected in cases:
            with self.subTest(mode=mode, fully_idle=fully_idle):
                snapshot = _new_load_inquirer(
                    mode=mode,
                    fully_idle=fully_idle,
                    prefill_throughput=123.456,
                ).get_loads()
                self.assertEqual(snapshot.prefill_throughput, expected)


class TestSchedulerSnapshotPublication(CustomTestCase):
    """Verify snapshot publication follows Prefill result accounting."""

    def test_prefill_result_processing_precedes_snapshot_publication(self) -> None:
        """Publish only after the processor updates Prefill statistics.

        Returns:
            None.
        """
        scheduler = Scheduler.__new__(Scheduler)
        order = Mock()
        scheduler.batch_result_processor = SimpleNamespace(
            process_batch_result_prefill=order.process_prefill
        )
        scheduler.publish_load_snapshot = order.publish
        scheduler.metrics_reporter = MagicMock()
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.enable_fpm = False
        scheduler._maybe_clear_mm_inputs = MagicMock()
        scheduler.maybe_send_health_check_signal = MagicMock()

        batch = MagicMock()
        batch.forward_mode.is_decode.return_value = False
        batch.forward_mode.is_extend.return_value = True
        batch.forward_mode.is_prebuilt.return_value = False
        batch.forward_mode.is_idle.return_value = False
        batch.is_dllm.return_value = False
        result = object()

        Scheduler.process_batch_result(scheduler, batch, result)

        self.assertEqual(
            order.mock_calls[:2],
            [call.process_prefill(batch, result), call.publish(force=True)],
        )


if __name__ == "__main__":
    unittest.main()
