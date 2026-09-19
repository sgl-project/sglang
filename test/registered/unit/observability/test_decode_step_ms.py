"""Pure-CPU unit tests for the decode-log ``step-ms``/``gap-ms`` accounting.

``step-ms`` must reflect decode-step wall time only; idle loop iterations and
prefill work between decode logs go to ``gap-ms`` instead of inflating the next
``step-ms`` reading.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import logging
import re
import types
import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    PrefillStats,
    SchedulerMetricsReporter,
)
from sglang.srt.observability.metrics_collector import (
    SchedulerMetricsCollectorContext,
)
from sglang.srt.runtime_context import get_context
from sglang.test.test_utils import CustomTestCase

_LOGGER_NAME = "sglang.srt.managers.scheduler_components.metrics_reporter"


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def set(self, t):
        self.t = t


def _make_scheduler():
    scheduler = MagicMock()
    scheduler.device = "cuda"
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.waiting_queue = []
    scheduler.forward_ct = 0
    scheduler.spec_algorithm = types.SimpleNamespace(is_none=lambda: True)
    scheduler.pool_stats_observer.get_pool_stats.return_value = types.SimpleNamespace(
        get_decode_usage_msg_parts=lambda: [],
        get_prefill_usage_msg_parts=lambda: [],
    )
    return scheduler


def _make_reporter() -> SchedulerMetricsReporter:
    context = SchedulerMetricsCollectorContext(
        enable_metrics=False,
        is_stats_logging_rank=True,
        current_scheduler_metrics_enabled=False,
        enable_kv_cache_events=False,
        collector=None,
    )
    with patch.object(SchedulerMetricsReporter, "__init__", return_value=None):
        reporter = SchedulerMetricsReporter()
    reporter.scheduler = _make_scheduler()
    reporter.metrics_collector_context = context
    reporter.metrics_collector = None
    reporter.enable_metrics = False
    reporter.is_stats_logging_rank = True
    reporter.current_scheduler_metrics_enabled = False
    reporter.enable_kv_cache_events = False
    reporter._init_metrics(0, 0, None)
    return reporter


def _fake_batch():
    return types.SimpleNamespace(
        reqs=[object()],
        batch_size=lambda: 1,
        forward_iter=None,
        dp_cooperation_info=None,
    )


def _decode_lines(records):
    return [r.getMessage() for r in records if "Decode batch" in r.getMessage()]


def _metric(line, name):
    m = re.search(rf"{name}: ([0-9.]+)", line)
    return float(m.group(1)) if m else None


class _ReporterTestBase(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(decode_log_interval=1)
        override.install()
        self.addCleanup(override.restore)
        self.reporter = _make_reporter()
        self.clock = _Clock()
        self._patcher = patch(
            "sglang.srt.managers.scheduler_components.metrics_reporter."
            "time.perf_counter",
            new=self.clock,
        )
        self._patcher.start()
        self.addCleanup(self._patcher.stop)
        # Fresh accounting origin at t=0.
        self.reporter._step_tic = 0.0
        self.reporter.last_decode_stats_tic = 0.0
        self.reporter.last_prefill_stats_tic = 0.0

    def _decode(self, t):
        self.reporter.forward_ct_decode = (self.reporter.forward_ct_decode + 1) % (
            1 << 30
        )
        self.clock.set(t)
        self.reporter.report_decode_stats(
            can_run_cuda_graph=True, running_batch=_fake_batch()
        )

    def _idle(self, t):
        self.clock.set(t)
        self.reporter.mark_idle()

    def _prefill(self, t):
        self.clock.set(t)
        self.reporter.report_prefill_stats(
            batch=None,
            prefill_stats=PrefillStats(
                log_input_tokens=1,
                log_hit_tokens=0,
                new_token_ratio=1.0,
                num_running_reqs=types.SimpleNamespace(total=0),
                num_new_seqs=1,
            ),
            can_run_cuda_graph=False,
        )


class TestIdleGapExcluded(_ReporterTestBase):
    def test_idle_and_prefill_time_go_to_gap_not_step(self):
        self.clock.set(1.0)
        self._decode(1.0)
        self._idle(2.0)
        self._idle(13.0)
        self._prefill(13.05)
        logger = logging.getLogger(_LOGGER_NAME)
        with self.assertLogs(logger, level="INFO") as cm:
            self._decode(13.07)

        lines = _decode_lines(cm.records)
        self.assertEqual(len(lines), 1)
        line = lines[0]
        m = re.search(r"step-ms: ([0-9.]+)", line)
        self.assertIsNotNone(m)
        self.assertAlmostEqual(float(m.group(1)), 20.0, delta=0.1)
        self.assertAlmostEqual(_metric(line, "gap-ms"), 12050.0, delta=0.1)
        # Field order kept: gen throughput, step-ms, gap-ms, #queue-req.
        idx_throughput = line.index("gen throughput (token/s):")
        idx_step = line.index("step-ms:")
        idx_queue = line.index("#queue-req:")
        self.assertLess(idx_throughput, idx_step)
        self.assertLess(idx_step, idx_queue)


class TestBackToBackDecode(_ReporterTestBase):
    def test_back_to_back_decodes_unchanged(self):
        logger = logging.getLogger(_LOGGER_NAME)
        self._decode(1.000)
        with self.assertLogs(logger, level="INFO") as cm2:
            self._decode(1.013)
        with self.assertLogs(logger, level="INFO") as cm3:
            self._decode(1.026)

        for records in (cm2.records, cm3.records):
            lines = _decode_lines(records)
            self.assertEqual(len(lines), 1)
            self.assertAlmostEqual(_metric(lines[0], "step-ms"), 13.0, delta=0.1)
            self.assertAlmostEqual(_metric(lines[0], "gap-ms"), 0.0, delta=0.1)


class TestDecodeLogInterval(_ReporterTestBase):
    def setUp(self):
        super().setUp()
        self.reporter.decode_log_interval = 4

    def test_interval_averages_steps_and_accumulates_gap(self):
        logger = logging.getLogger(_LOGGER_NAME)
        for t in (0.010, 0.020, 0.030):
            with self.assertNoLogs(logger, level="INFO"):
                self._decode(t)
        with self.assertLogs(logger, level="INFO") as cm:
            self._decode(0.040)
        lines = _decode_lines(cm.records)
        self.assertEqual(len(lines), 1)
        self.assertAlmostEqual(_metric(lines[0], "step-ms"), 10.0, delta=0.1)
        self.assertAlmostEqual(_metric(lines[0], "gap-ms"), 0.0, delta=0.1)

        self._prefill(0.100)
        for t in (0.110, 0.120, 0.130):
            with self.assertNoLogs(logger, level="INFO"):
                self._decode(t)
        with self.assertLogs(logger, level="INFO") as cm2:
            self._decode(0.140)
        lines = _decode_lines(cm2.records)
        self.assertEqual(len(lines), 1)
        self.assertAlmostEqual(_metric(lines[0], "step-ms"), 10.0, delta=0.1)
        self.assertAlmostEqual(_metric(lines[0], "gap-ms"), 60.0, delta=0.1)


class TestIdleBatchAccounting(_ReporterTestBase):
    """DP-attention idle batches never reach ``on_idle()``; they route through
    ``process_batch_result_idle``, which must mark the interval as gap."""

    def test_idle_interval_between_decodes_is_gap(self):
        self._decode(0.013)
        self._idle(0.513)
        logger = logging.getLogger(_LOGGER_NAME)
        with self.assertLogs(logger, level="INFO") as cm:
            self._decode(0.526)
        lines = _decode_lines(cm.records)
        self.assertEqual(len(lines), 1)
        self.assertAlmostEqual(_metric(lines[0], "step-ms"), 13.0, delta=0.1)
        self.assertAlmostEqual(_metric(lines[0], "gap-ms"), 500.0, delta=0.1)

    def test_process_batch_result_idle_marks_idle(self):
        with patch.object(SchedulerBatchResultProcessor, "__init__", return_value=None):
            processor = SchedulerBatchResultProcessor()
        # Frozen dataclass: bypass __setattr__ to inject the two collaborators.
        object.__setattr__(processor, "metrics_reporter", MagicMock())
        object.__setattr__(processor, "output_streamer", MagicMock())
        batch = types.SimpleNamespace(reqs=[], return_logprob=False)
        result = types.SimpleNamespace(copy_done=None)
        processor.process_batch_result_idle(batch, result)
        processor.metrics_reporter.mark_idle.assert_called_once_with()


class TestDisaggPrebuiltAdmission(CustomTestCase):
    """Transferred-req admission on an idle disagg-decode worker must close the
    gap interval, but only when no decode is in flight."""

    def _run(self, running_is_empty, pp_drained=True):
        from sglang.srt.disaggregation.decode import (
            SchedulerDisaggregationDecodeMixin,
        )

        mock_self = MagicMock()
        mock_self.enable_hisparse = False
        mock_self.chunked_req = None
        mock_self._pp_microbatches_drained.return_value = pp_drained
        prebuilt = MagicMock()
        prebuilt.is_empty.return_value = False
        mock_self.get_new_prebuilt_batch.return_value = prebuilt
        running_batch = MagicMock()
        running_batch.is_empty.return_value = running_is_empty
        updated = MagicMock()
        updated.is_empty.return_value = False
        mock_self.update_running_batch.return_value = updated
        mock_self.dp_attn_adapter.maybe_prepare_mlp_sync_batch.side_effect = (
            lambda ret: ret
        )

        SchedulerDisaggregationDecodeMixin.get_next_disagg_decode_batch_to_run(
            mock_self, running_batch
        )
        return mock_self.metrics_reporter.mark_idle

    def test_marks_idle_when_running_batch_empty(self):
        mark_idle = self._run(running_is_empty=True)
        mark_idle.assert_called_once_with()

    def test_no_mark_when_decode_in_flight(self):
        mark_idle = self._run(running_is_empty=False)
        mark_idle.assert_not_called()

    def test_no_mark_when_other_pp_microbatches_active(self):
        mark_idle = self._run(running_is_empty=True, pp_drained=False)
        mark_idle.assert_not_called()


class TestResetMetrics(_ReporterTestBase):
    def test_reset_metrics_zeroes_accumulators(self):
        self._decode(1.0)
        self._idle(2.0)
        self.reporter.reset_metrics()
        self.assertEqual(self.reporter.decode_step_time_acc, 0.0)
        self.assertEqual(self.reporter.decode_gap_time_acc, 0.0)



class TestEnginePauseAccounting(_ReporterTestBase):
    def test_engine_pause_counts_as_gap_not_step(self):
        """A paused engine must charge pause wall time to gap-ms, not let the
        whole pause inflate the next decode's step-ms."""
        from sglang.srt.managers.scheduler import Scheduler

        logger = logging.getLogger(_LOGGER_NAME)
        self.clock.set(10.0)
        self._decode(10.0)
        self.clock.set(70.0)
        sched = MagicMock()
        sched.metrics_reporter = self.reporter
        sched.is_fully_idle = MagicMock(return_value=True)
        sched.metrics_reporter.record_scheduler_idle = MagicMock()
        Scheduler._record_scheduler_state_for_paused_engine(sched)
        with self.assertLogs(logger, level="INFO") as cm:
            self._decode(70.015)

        line = _decode_lines(cm.records)[-1]
        self.assertAlmostEqual(_metric(line, "step-ms"), 15.0, delta=0.1)
        self.assertAlmostEqual(_metric(line, "gap-ms"), 60000.0, delta=1.0)
        sched.metrics_reporter.record_scheduler_idle.assert_called_once()


if __name__ == "__main__":
    unittest.main()
