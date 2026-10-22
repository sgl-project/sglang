"""PD workers log idle steps so a plain average over stats lines tracks a time average."""

import re
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    PrefillStats,
    SchedulerMetricsReporter,
    _ForwardOccupancyLogWindow,
    _SchedulerTimeAccountingSnapshot,
)
from sglang.srt.observability.metrics_collector import QueueCount
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, enter_scope, published_topology

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_MODULE = "sglang.srt.managers.scheduler_components.metrics_reporter"
_QUEUE_RE = re.compile(r"#queue-req: (\d+)")


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _make_scheduler(mode: DisaggregationMode, spec_enabled: bool = False):
    pool_stats = SimpleNamespace(
        get_prefill_usage_msg_parts=lambda: ["token usage: 0.00"],
        get_decode_usage_msg_parts=lambda: ["token usage: 0.00"],
        update_scheduler_stats=lambda stats: None,
    )
    return SimpleNamespace(
        disaggregation_mode=mode,
        forward_ct=0,
        waiting_queue=[],
        grammar_manager=[],
        enable_lora=False,
        enable_hierarchical_cache=False,
        enable_fpm=False,
        running_batch=SimpleNamespace(
            reqs=[], forward_iter=None, seq_lens_cpu=None, dp_cooperation_info=None
        ),
        disagg_prefill_bootstrap_queue=SimpleNamespace(queue=[]),
        disagg_prefill_inflight_queue=[],
        disagg_decode_prealloc_queue=SimpleNamespace(
            queue=[], retracted_queue=[], num_tokens_pre_allocated=0
        ),
        disagg_decode_transfer_queue=SimpleNamespace(
            queue=[], num_pending_deferred_releases=lambda: 0
        ),
        max_total_num_tokens=1,
        enable_priority_scheduling=False,
        spec_algorithm=SimpleNamespace(
            is_none=lambda: not spec_enabled, is_dspark=lambda: False
        ),
        load_inquirer=SimpleNamespace(_get_num_pending_tokens=lambda: 0),
        pool_stats_observer=SimpleNamespace(
            get_pool_stats=lambda: pool_stats,
            streaming_session_count=lambda: 0,
            session_held_tokens=lambda: 0,
        ),
        kv_events_publisher=SimpleNamespace(
            publish_kv_events=lambda: None, emit_kv_metrics=lambda: None
        ),
    )


def _make_reporter(scheduler, metrics_enabled: bool) -> SchedulerMetricsReporter:
    reporter = SchedulerMetricsReporter.__new__(SchedulerMetricsReporter)
    reporter.scheduler = scheduler
    reporter.enable_metrics = False
    reporter._init_metrics()
    reporter.is_stats_logging_rank = True
    reporter.current_scheduler_metrics_enabled = metrics_enabled
    reporter.metrics_collector = MagicMock()
    return reporter


def _idle_forward(reporter: SchedulerMetricsReporter) -> None:
    processor = SimpleNamespace(
        output_streamer=SimpleNamespace(
            _stream_output_generation=lambda *args, **kwargs: None
        ),
        metrics_reporter=reporter,
    )
    SchedulerBatchResultProcessor.process_batch_result_idle(
        processor,
        batch=SimpleNamespace(
            reqs=[],
            return_logprob=False,
            forward_iter=None,
            extend_lens=None,
            seq_lens_cpu=None,
        ),
        result=SimpleNamespace(copy_done=None, can_run_cuda_graph=True),
    )


class TestIdleBatchStatsLines(CustomTestCase):
    def setUp(self):
        super().setUp()
        enter_scope(self, published_topology(role="scheduler", decode_log_interval=2))
        self.clock = _Clock()
        enter_scope(self, patch(f"{_MODULE}.time.perf_counter", self.clock))

    def _prefill_timeline_queue_mean(self, log_idle: bool) -> float:
        """Busy for 1 s with three queued requests, then idle for 1 s."""
        self.clock.now = 0.0
        scheduler = _make_scheduler(DisaggregationMode.PREFILL)
        with patch(f"{_MODULE}.LOG_IDLE_BATCH_STATS", log_idle):
            reporter = _make_reporter(scheduler, metrics_enabled=False)
            with self.assertLogs(f"{_MODULE}", level="INFO") as logs:
                scheduler.waiting_queue = [object()] * 3
                for step in range(1, 11):
                    self.clock.now = step * 0.1
                    reporter.report_prefill_stats(
                        batch=SimpleNamespace(reqs=[], forward_iter=None),
                        prefill_stats=PrefillStats(
                            log_input_tokens=100,
                            log_hit_tokens=0,
                            new_token_ratio=0.0,
                            num_running_reqs=QueueCount(),
                            num_new_seqs=1,
                        ),
                        can_run_cuda_graph=False,
                    )
                scheduler.waiting_queue = []
                for poll in range(1, 1001):
                    self.clock.now = 1.0 + poll * 0.001
                    reporter.maybe_report_idle_stats()
        queue = [
            int(m.group(1)) for line in logs.output if (m := _QUEUE_RE.search(line))
        ]
        return sum(queue) / len(queue)

    def test_prefill_line_average_tracks_time_average(self):
        # The time average of #queue-req over the 2 s timeline is 1.5.
        self.assertAlmostEqual(
            self._prefill_timeline_queue_mean(log_idle=True), 1.5, delta=0.1
        )
        # Without idle lines the idle second is invisible to a line average.
        self.assertEqual(self._prefill_timeline_queue_mean(log_idle=False), 3.0)

    def test_attention_dp_idle_forward_lines(self):
        with patch(f"{_MODULE}.LOG_IDLE_BATCH_STATS", True):
            prefill = _make_reporter(
                _make_scheduler(DisaggregationMode.PREFILL), metrics_enabled=True
            )
            # Speculative decode with a window of idle forwards has no acceptance.
            decode = _make_reporter(
                _make_scheduler(DisaggregationMode.DECODE, spec_enabled=True),
                metrics_enabled=True,
            )
            # The acceptance exported for the rank's last busy window.
            decode.stats.spec_accept_length = 2.5
            with self.assertLogs(f"{_MODULE}", level="INFO") as logs:
                for _ in range(2):
                    self.clock.now += 0.1
                    _idle_forward(prefill)
                    _idle_forward(decode)

        prefill_lines = [line for line in logs.output if "Prefill batch," in line]
        decode_lines = [line for line in logs.output if "Decode batch," in line]
        self.assertEqual(len(prefill_lines), 2)
        self.assertTrue(all("#new-seq: 0, #new-token: 0" in l for l in prefill_lines))
        self.assertEqual(prefill.metrics_collector.method_calls, [])
        # One decode line per decode-log-interval forwards, as on a busy rank.
        self.assertEqual(len(decode_lines), 1)
        self.assertIn("#running-req: 0", decode_lines[0])
        self.assertNotIn("accept len", decode_lines[0])
        # The decode line exports its window, which keeps the acceptance gauge.
        self.assertEqual(decode.metrics_collector.log_stats.call_count, 1)
        self.assertEqual(decode.stats.gen_throughput, 0)
        self.assertEqual(decode.stats.spec_accept_length, 2.5)

    def test_decode_timed_line_keeps_dp_ranks_on_the_same_forwards(self):
        """Attention-DP decode ranks time idle lines on their own clocks; a timed
        line on one rank must not shift which forwards its batch lines land on."""
        with patch(f"{_MODULE}.LOG_IDLE_BATCH_STATS", True):
            ranks = []
            for gap_ema_s in (0.800, 0.806):
                rank = _make_reporter(
                    _make_scheduler(DisaggregationMode.DECODE), metrics_enabled=False
                )
                rank._decode_idle_cadence.gap_ema_s = gap_ema_s
                # Traffic stopped one forward into a decode-log-interval of 2.
                rank.forward_ct_decode = 1
                ranks.append(rank)
            logged_at = [[], []]
            with self.assertLogs(f"{_MODULE}", level="INFO") as logs:
                # The first idle step closes the open forward on both ranks.
                self.clock.now = 0.01
                for rank in ranks:
                    rank.maybe_report_idle_stats()
                self.assertEqual(len(logs.output), 2)
                # Past the first rank's idle interval only.
                self.clock.now = 0.813
                for rank in ranks:
                    rank.maybe_report_idle_stats()
                self.assertEqual(len(logs.output), 3)
                # Traffic resumes: every rank runs the same forwards.
                for forward in range(1, 5):
                    self.clock.now += 0.01
                    for i, rank in enumerate(ranks):
                        before = len(logs.output)
                        _idle_forward(rank)
                        if len(logs.output) > before:
                            logged_at[i].append(forward)
        self.assertEqual(logged_at[0], logged_at[1])

    def test_every_decode_line_exports_its_window(self):
        """Each decode line, idle or not, closes the window since the previous
        line and exports that window's rate. last_gen_throughput, which
        /server_info reports after a run, keeps the last busy window's rate."""
        with patch(f"{_MODULE}.LOG_IDLE_BATCH_STATS", True):
            reporter = _make_reporter(
                _make_scheduler(DisaggregationMode.DECODE), metrics_enabled=True
            )
            log_stats = reporter.metrics_collector.log_stats
            reporter.last_gen_throughput = 7.0

            def assert_exported(count: int, throughput: float) -> None:
                self.assertEqual(log_stats.call_count, count)
                self.assertAlmostEqual(reporter.stats.gen_throughput, throughput)

            with self.assertLogs(f"{_MODULE}", level="INFO"):
                # The rank's last request finished after one real forward of 8
                # tokens; a busy peer's next forward makes it idle-forward.
                reporter.num_generated_tokens = 8
                reporter.forward_ct_decode = 1
                self.clock.now = 0.25
                _idle_forward(reporter)
                assert_exported(count=1, throughput=32.0)

                # A burst shorter than the interval, closed by the first idle step.
                reporter.num_generated_tokens = 4
                reporter.forward_ct_decode += 1
                self.clock.now = 1.5
                reporter.maybe_report_idle_stats()
                assert_exported(count=2, throughput=3.2)

                # A window without work.
                self.clock.now = 3.0
                reporter.maybe_report_idle_stats()
                assert_exported(count=3, throughput=0.0)
                self.assertEqual(reporter.last_gen_throughput, 7.0)

                # A batch line on the last forward of its window.
                reporter.num_generated_tokens = 6
                reporter.forward_ct_decode = reporter.decode_log_interval
                self.clock.now = 3.5
                reporter.report_decode_stats(
                    can_run_cuda_graph=False,
                    running_batch=reporter.scheduler.running_batch,
                )
                assert_exported(count=4, throughput=12.0)
        self.assertAlmostEqual(reporter.last_gen_throughput, 12.0)

    def test_going_idle_exports_the_tail_window_then_zero(self):
        """The first idle step exports the window holding the last burst's tail
        before the fully idle refresh runs; without the device timer the refresh
        then finds nothing stale, and the throughput gauge reads the tail's
        rate, then 0."""
        with patch(f"{_MODULE}.LOG_IDLE_BATCH_STATS", True):
            reporter = _make_reporter(
                _make_scheduler(DisaggregationMode.DECODE), metrics_enabled=True
            )
            # A busy window exported one running request, which then finished
            # after one more forward of 4 tokens.
            reporter.stats.num_running_reqs = QueueCount.from_reqs([object()])
            reporter.num_generated_tokens = 4
            reporter.forward_ct_decode = 1
            reporter.metrics_collector.last_log_time = 0.0
            exported = []
            with self.assertLogs(f"{_MODULE}", level="INFO"):
                for now in (0.5, 1.5):
                    self.clock.now = now
                    # on_idle's order: idle stats line, then the fully idle refresh.
                    reporter.maybe_report_idle_stats()
                    reporter._maybe_log_idle_metrics()
                    exported.append(reporter.stats.gen_throughput)
        self.assertEqual(reporter.metrics_collector.log_stats.call_count, 2)
        self.assertEqual(reporter.stats.num_running_reqs.total, 0)
        self.assertEqual(exported, [8.0, 0.0])

    def test_device_timer_refresh_keeps_only_the_tail_rate(self):
        """With the device timer, a finite occupancy makes the fully idle refresh
        export right after the line that closed the last window. After an
        idle-start line it keeps that line's tail rate; after a batch line
        (decode-log-interval 1, no idle-start line) the window was busy, so it
        zeroes the rate."""
        for tail in (True, False):
            with (
                self.subTest(tail=tail),
                patch(f"{_MODULE}.LOG_IDLE_BATCH_STATS", True),
                patch(f"{_MODULE}.ENABLE_METRICS_DEVICE_TIMER", True),
            ):
                self.clock.now = 0.0
                reporter = _make_reporter(
                    _make_scheduler(DisaggregationMode.DECODE), metrics_enabled=True
                )
                reporter.forward_pass_device_timer = SimpleNamespace(
                    _report=lambda: None
                )
                reporter.metrics_collector.last_log_time = 0.0
                reporter.fwd_occupancy = 50.0
                reporter.num_generated_tokens = 4
                reporter.forward_ct_decode = 1
                with self.assertLogs(f"{_MODULE}", level="INFO"):
                    self.clock.now = 0.5
                    if tail:
                        reporter.maybe_report_idle_stats()
                    else:
                        reporter.decode_log_interval = 1
                        reporter.report_decode_stats(
                            can_run_cuda_graph=False,
                            running_batch=reporter.scheduler.running_batch,
                        )
                    reporter._maybe_log_idle_metrics()
                self.assertEqual(reporter.metrics_collector.log_stats.call_count, 2)
                self.assertEqual(reporter.stats.gen_throughput, 8.0 if tail else 0)

    def test_fully_idle_refresh_zeroes_throughput(self):
        """A stale running-request gauge means the last export was a busy window;
        once fully idle, its throughput is stale too."""
        cases = (
            (False, DisaggregationMode.DECODE),
            (True, DisaggregationMode.NULL),
            (True, DisaggregationMode.DECODE),
        )
        for log_idle, mode in cases:
            with (
                self.subTest(log_idle=log_idle, mode=mode),
                patch(f"{_MODULE}.LOG_IDLE_BATCH_STATS", log_idle),
            ):
                reporter = _make_reporter(_make_scheduler(mode), metrics_enabled=True)
                reporter.stats.num_running_reqs = QueueCount.from_reqs([object()])
                reporter.stats.gen_throughput = 4.0
                reporter._maybe_log_idle_metrics()
                self.assertEqual(reporter.metrics_collector.log_stats.call_count, 1)
                self.assertEqual(reporter.stats.gen_throughput, 0)

    def test_flag_leaves_non_pd_workers_unchanged(self):
        """Idle lines exist only on PD workers, so elsewhere the flag must not
        change how decode and occupancy windows close."""
        with (
            patch(f"{_MODULE}.LOG_IDLE_BATCH_STATS", True),
            patch(f"{_MODULE}.ENABLE_METRICS_DEVICE_TIMER", True),
        ):
            reporter = _make_reporter(
                _make_scheduler(DisaggregationMode.NULL), metrics_enabled=False
            )
            reporter.forward_ct_decode = 3
            reporter.reset_metrics()
            self.assertEqual(reporter.forward_ct_decode, 0)

            window = _ForwardOccupancyLogWindow(start_ns=0, gpu_seconds=0.25)
            reporter._forward_occupancy_log_window = window
            reporter.forward_pass_device_timer = SimpleNamespace(_report=lambda: None)
            self.clock.now = 1.0
            reporter.forward_occupancy_log_string()
            self.assertEqual((window.start_ns, window.gpu_seconds), (0, 0.25))
            # Every decode-log-interval batches and the turn to idle still close it.
            for _ in range(reporter.decode_log_interval):
                reporter.update_device_timer()
            self.assertGreater(window.start_ns, 0)
            reporter._scheduler_time_accounting = _SchedulerTimeAccountingSnapshot.init(
                now_ns=0, now_process_cpu_ns=0, is_idle=False
            )
            reporter._record_scheduler_time(is_idle=True, event_ns=5)
            self.assertEqual(window.start_ns, 5)

    def test_decode_sol_hook_gets_the_time_per_forward(self):
        """The decode SoL hook takes per-iteration time, so a window closed at an
        idle step divides by the forwards it holds, and one without any skips
        the hook."""
        with patch(f"{_MODULE}.LOG_IDLE_BATCH_STATS", True):
            reporter = _make_reporter(
                _make_scheduler(DisaggregationMode.DECODE), metrics_enabled=False
            )
            reporter.enable_mfu_metrics = True
            reporter._mfu_log_flops = 0.0
            reporter._mfu_log_read_bytes = 0.0
            reporter._mfu_log_write_bytes = 0.0
            reporter._decode_sol_suffix = MagicMock(return_value="")
            with self.assertLogs(f"{_MODULE}", level="INFO") as logs:
                # Traffic stopped one forward into a decode-log-interval of 2.
                reporter.forward_ct_decode = 1
                for now in (1.0, 2.0):
                    self.clock.now = now
                    reporter.maybe_report_idle_stats()
        self.assertEqual(len(logs.output), 2)
        reporter._decode_sol_suffix.assert_called_once()
        self.assertAlmostEqual(reporter._decode_sol_suffix.call_args.args[1], 1.0)

    def test_idle_export_of_an_emptied_batch_sums_no_seq_lens(self):
        """A running batch whose requests all finished keeps stale seq_lens."""
        with patch(f"{_MODULE}.LOG_IDLE_BATCH_STATS", True):
            reporter = _make_reporter(
                _make_scheduler(DisaggregationMode.DECODE), metrics_enabled=True
            )
            reporter.scheduler.running_batch.seq_lens_cpu = torch.tensor([100, 200])
            reporter.stats.decode_sum_seq_lens = 300
            with self.assertLogs(f"{_MODULE}", level="INFO"):
                self.clock.now = 1.0
                reporter.maybe_report_idle_stats()
        self.assertEqual(reporter.metrics_collector.log_stats.call_count, 1)
        self.assertEqual(reporter.stats.decode_sum_seq_lens, 0)

    def test_metrics_only_rank_exports_idle_decode_windows(self):
        """With metrics on every scheduler, a rank that does not log still counts
        idle forwards, so it closes and exports windows on its logging rank's
        forwards, and while idle."""
        with patch(f"{_MODULE}.LOG_IDLE_BATCH_STATS", True):
            reporter = _make_reporter(
                _make_scheduler(DisaggregationMode.DECODE), metrics_enabled=True
            )
            reporter.is_stats_logging_rank = False
            log_stats = reporter.metrics_collector.log_stats
            with self.assertNoLogs(f"{_MODULE}", level="INFO"):
                for _ in range(2):
                    self.clock.now += 0.1
                    _idle_forward(reporter)
                self.assertEqual(log_stats.call_count, 1)
                self.clock.now += 2.0
                reporter.maybe_report_idle_stats()
        self.assertEqual(log_stats.call_count, 2)

    def test_cache_flush_keeps_dp_ranks_on_the_same_forwards(self):
        """A cache flush lands on attention-DP ranks at different forwards, e.g.
        on an idle rank while its busy peer refuses; it must not shift which
        forwards their decode lines land on."""
        with patch(f"{_MODULE}.LOG_IDLE_BATCH_STATS", True):
            ranks = []
            for _ in range(2):
                rank = _make_reporter(
                    _make_scheduler(DisaggregationMode.DECODE), metrics_enabled=False
                )
                rank.forward_ct_decode = 1
                ranks.append(rank)
            ranks[0].reset_metrics()
            logged_at = [[], []]
            with self.assertLogs(f"{_MODULE}", level="INFO") as logs:
                for forward in range(1, 5):
                    self.clock.now += 0.01
                    for i, rank in enumerate(ranks):
                        before = len(logs.output)
                        _idle_forward(rank)
                        if len(logs.output) > before:
                            logged_at[i].append(forward)
        self.assertEqual(logged_at[0], logged_at[1])

    def test_step_time_skips_a_window_shortened_by_an_idle_line(self):
        """An idle line closes a window without advancing the forward count, so
        the next batch line closes a window of fewer than decode-log-interval
        forwards; dividing its gap by the interval would understate the step."""
        with (
            patch(f"{_MODULE}.LOG_IDLE_BATCH_STATS", True),
            patch(f"{_MODULE}.RECORD_STEP_TIME", True),
        ):
            reporter = _make_reporter(
                _make_scheduler(DisaggregationMode.DECODE), metrics_enabled=False
            )

            def forward() -> None:
                reporter.forward_ct_decode += 1
                reporter.report_decode_stats(
                    can_run_cuda_graph=False,
                    running_batch=reporter.scheduler.running_batch,
                )

            with self.assertLogs(f"{_MODULE}", level="INFO"):
                # Traffic stopped one forward into a decode-log-interval of 2.
                reporter.forward_ct_decode = 1
                self.clock.now = 1.0
                reporter.maybe_report_idle_stats()
                self.clock.now = 1.1
                forward()
                self.assertEqual(dict(reporter.step_time_dict), {})
                for _ in range(2):
                    self.clock.now += 0.1
                    forward()
        self.assertEqual(list(reporter.step_time_dict), [0])
        self.assertEqual(len(reporter.step_time_dict[0]), 1)
        self.assertAlmostEqual(reporter.step_time_dict[0][0], 0.1)

    def test_decode_windows_count_real_forwards_when_off(self):
        """Without idle lines, a decode window closes every decode-log-interval
        real forwards, whatever idle forwards and polls run in between."""
        with patch(f"{_MODULE}.LOG_IDLE_BATCH_STATS", False):
            reporter = _make_reporter(
                _make_scheduler(DisaggregationMode.DECODE), metrics_enabled=True
            )
            log_stats = reporter.metrics_collector.log_stats
            exported = []
            with self.assertLogs(f"{_MODULE}", level="INFO") as logs:
                for _ in range(4):
                    self.clock.now += 0.1
                    reporter.num_generated_tokens += 3
                    reporter.forward_ct_decode += 1
                    reporter.report_decode_stats(
                        can_run_cuda_graph=False,
                        running_batch=reporter.scheduler.running_batch,
                    )
                    if log_stats.call_count > len(exported):
                        exported.append(reporter.stats.gen_throughput)
                    _idle_forward(reporter)
                    self.clock.now += 2.0
                    reporter.maybe_report_idle_stats()
        self.assertEqual(len(logs.output), 2)
        self.assertEqual(reporter.forward_ct_decode, 4)
        # Each window spans two real forwards plus the idle time between them.
        self.assertEqual(len(exported), 2)
        self.assertAlmostEqual(exported[0], 6 / 2.2)
        self.assertAlmostEqual(exported[1], 6 / 4.2)

    def test_line_occupancy_covers_the_time_since_the_previous_line(self):
        """Occupancy on a line covers the same span as its other fields: a burst
        followed by idle time stays in the next line, and each line's idle time
        is counted once."""

        def clock_ns() -> int:
            return round(self.clock.now * 1e9)

        with patch(f"{_MODULE}.LOG_IDLE_BATCH_STATS", True):
            reporter = _make_reporter(
                _make_scheduler(DisaggregationMode.DECODE), metrics_enabled=False
            )
            reporter.forward_pass_device_timer = SimpleNamespace(_report=lambda: None)
            reporter._forward_occupancy_log_window = _ForwardOccupancyLogWindow(
                start_ns=0
            )
            reporter._scheduler_time_accounting = _SchedulerTimeAccountingSnapshot.init(
                now_ns=0, now_process_cpu_ns=0, is_idle=False
            )
            with (
                patch(f"{_MODULE}.ENABLE_METRICS_DEVICE_TIMER", True),
                patch(f"{_MODULE}.time.monotonic_ns", clock_ns),
                self.assertLogs(f"{_MODULE}", level="INFO") as logs,
            ):
                # A burst of fewer than decode-log-interval forwards, 0.25 s on the
                # GPU, then idle from 0.5 s until a timed line.
                reporter._forward_occupancy_log_window.gpu_seconds += 0.25
                self.clock.now = 0.5
                reporter._record_scheduler_time(is_idle=True, event_ns=clock_ns())
                self.clock.now = 1.0
                reporter.maybe_report_idle_stats()
                # Idle until a request at 1.5 s, 0.1 s on the GPU, then a stall
                # without forwards until the next timed line.
                self.clock.now = 1.5
                reporter._record_scheduler_time(is_idle=False, event_ns=clock_ns())
                reporter._forward_occupancy_log_window.gpu_seconds += 0.1
                self.clock.now = 2.0
                reporter.maybe_report_idle_stats()
        self.assertEqual(len(logs.output), 2)
        self.assertIn(
            "fwd occupancy: 25.00%, fwd+idle occupancy: 75.00%", logs.output[0]
        )
        self.assertIn(
            "fwd occupancy: 10.00%, fwd+idle occupancy: 60.00%", logs.output[1]
        )

    def test_prefill_idle_lines_skip_the_sol_hook(self):
        with patch(f"{_MODULE}.LOG_IDLE_BATCH_STATS", True):
            reporter = _make_reporter(
                _make_scheduler(DisaggregationMode.PREFILL), metrics_enabled=False
            )
            reporter.enable_mfu_metrics = True
            reporter._prefill_sol_suffix = MagicMock(return_value="")
            with self.assertLogs(f"{_MODULE}", level="INFO") as logs:
                self.clock.now = 0.1
                _idle_forward(reporter)
                self.clock.now = 2.0
                reporter.maybe_report_idle_stats()
        self.assertEqual(len(logs.output), 2)
        reporter._prefill_sol_suffix.assert_not_called()
        self.assertTrue(
            all("est. prefill TFLOPS/s (per GPU): 0.00" in l for l in logs.output)
        )

    def test_decode_idle_boundary_lines_bound_the_idle_stretch(self):
        """With decode-log-interval > 1, the first idle step closes the forwards
        since the previous line, and the line right before the next forward
        ends the idle stretch; the busy windows on either side then hold no
        idle time, and neither line moves the forward count. The end line is
        logged once the forward's requests are admitted, but none of them ran
        in its window."""
        with patch(f"{_MODULE}.LOG_IDLE_BATCH_STATS", True):
            reporter = _make_reporter(
                _make_scheduler(DisaggregationMode.DECODE), metrics_enabled=True
            )
            exported = []
            running = []

            def log_stats(stats) -> None:
                exported.append(stats.gen_throughput)
                running.append(stats.num_running_reqs.total)

            reporter.metrics_collector.log_stats.side_effect = log_stats

            def forward_at(now: float) -> None:
                self.clock.now = now
                reporter.num_generated_tokens += 3
                reporter.forward_ct_decode += 1
                reporter.report_decode_stats(
                    can_run_cuda_graph=False,
                    running_batch=reporter.scheduler.running_batch,
                )

            def idle_step_at(now: float) -> None:
                self.clock.now = now
                reporter.maybe_report_idle_stats()

            def launch_at(now: float) -> None:
                self.clock.now = now
                reporter.maybe_report_idle_end()

            with self.assertLogs(f"{_MODULE}", level="INFO") as logs:
                forward_at(0.1)
                forward_at(0.2)  # a batch line closes [0, 0.2]
                forward_at(0.3)
                idle_step_at(0.35)  # idle starts: closes [0.2, 0.35]
                idle_step_at(0.5)
                idle_step_at(1.35)  # a timed line
                reporter.scheduler.running_batch.reqs = [
                    SimpleNamespace(seqlen=8, routing_key=None)
                ] * 2
                launch_at(1.8)  # idle ends: closes [1.35, 1.8]
                launch_at(1.81)  # one end line per idle stretch
                forward_at(1.9)  # a batch line closes [1.8, 1.9]
                forward_at(2.0)
                idle_step_at(2.05)  # idle starts: closes [1.9, 2.05]
                launch_at(2.1)  # no timed line in this short stretch, so no end line
                forward_at(2.2)  # a batch line closes [2.05, 2.2]
        self.assertEqual(len(logs.output), 7)
        self.assertEqual(reporter.forward_ct_decode, 6)
        expected = [6 / 0.2, 3 / 0.15, 0.0, 0.0, 3 / 0.1, 3 / 0.15, 3 / 0.15]
        self.assertEqual(len(exported), len(expected))
        for got, want in zip(exported, expected):
            self.assertAlmostEqual(got, want)
        self.assertIn("#running-req: 0,", logs.output[3])
        self.assertEqual(running[3:5], [0, 2])

    def test_decode_idle_end_counts_admitted_requests_by_routing_key(self):
        """No request ran in an idle-end line's window, but its forward already
        admitted some; the running+waiting routing distribution still counts
        them, since they are in neither queue any more."""
        with patch(f"{_MODULE}.LOG_IDLE_BATCH_STATS", True):
            reporter = _make_reporter(
                _make_scheduler(DisaggregationMode.DECODE), metrics_enabled=True
            )
            reporter.scheduler.running_batch.reqs = [
                SimpleNamespace(seqlen=8, routing_key="a")
            ] * 2
            reporter.scheduler.waiting_queue = [SimpleNamespace(routing_key="b")]
            with self.assertLogs(f"{_MODULE}", level="INFO") as logs:
                self.clock.now = 0.3
                reporter.maybe_report_idle_end()
        self.assertIn("#running-req: 0,", logs.output[0])
        self.assertEqual(reporter.stats.num_running_reqs.total, 0)
        self.assertEqual(reporter.stats.num_unique_running_routing_keys, 0)
        self.assertEqual(sorted(reporter.stats.routing_key_all_req_counts), [1, 2])

    def test_first_forward_after_startup_gets_an_idle_end_line(self):
        """Startup counts as an idle stretch covered by timed lines, so the first
        forward closes it even before any timed line was logged."""
        with patch(f"{_MODULE}.LOG_IDLE_BATCH_STATS", True):
            reporter = _make_reporter(
                _make_scheduler(DisaggregationMode.DECODE), metrics_enabled=False
            )
            with self.assertLogs(f"{_MODULE}", level="INFO") as logs:
                self.clock.now = 0.3
                reporter.maybe_report_idle_end()
                reporter.maybe_report_idle_end()
        self.assertEqual(len(logs.output), 1)

    def test_prefill_idle_end_line_leaves_the_batch_window_busy(self):
        """The first prefill after timed lines measures its input throughput
        over its own forward, not the idle time before it."""
        with patch(f"{_MODULE}.LOG_IDLE_BATCH_STATS", True):
            reporter = _make_reporter(
                _make_scheduler(DisaggregationMode.PREFILL), metrics_enabled=False
            )
            with self.assertLogs(f"{_MODULE}", level="INFO") as logs:
                self.clock.now = 1.0
                reporter.maybe_report_idle_stats()
                self.clock.now = 1.5
                reporter.maybe_report_idle_end()
                reporter.maybe_report_idle_end()
                self.clock.now = 1.7
                reporter.report_prefill_stats(
                    batch=SimpleNamespace(reqs=[], forward_iter=None),
                    prefill_stats=PrefillStats(
                        log_input_tokens=100,
                        log_hit_tokens=0,
                        new_token_ratio=0.0,
                        num_running_reqs=QueueCount(),
                        num_new_seqs=1,
                    ),
                    can_run_cuda_graph=False,
                )
        self.assertEqual(len(logs.output), 3)
        self.assertIn("input throughput (token/s): 500.00", logs.output[2])

    def test_idle_lines_are_off_by_default(self):
        reporter = _make_reporter(
            _make_scheduler(DisaggregationMode.PREFILL), metrics_enabled=False
        )
        self.clock.now = 60.0
        with self.assertNoLogs(f"{_MODULE}", level="INFO"):
            reporter.maybe_report_idle_stats()
            reporter.maybe_report_idle_end()
            _idle_forward(reporter)


if __name__ == "__main__":
    unittest.main()
