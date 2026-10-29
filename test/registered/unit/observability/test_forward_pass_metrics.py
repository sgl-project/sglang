from sglang.srt.runtime_context import get_context, get_observability, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

import math
import queue
import types
import unittest
from collections import deque
from contextlib import nullcontext
from functools import partial
from itertools import count
from types import SimpleNamespace
from unittest.mock import Mock, patch

import prometheus_client
import torch

from sglang.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    PrefillStats,
    SchedulerMetricsReporter,
    _CacheHitRateWindow,
)
from sglang.srt.managers.scheduler_pp_mixin import PPBatchMetadata, SchedulerPPMixin
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.forward_pass_metrics import (
    ForwardPassMetrics,
    FpmTiming,
    _FpmPublisherThread,
    decode,
    wrap_forward_with_fpm,
)
from sglang.srt.observability.metrics_collector import (
    DPBalanceStats,
    SchedulerMetricsCollector,
)
from sglang.srt.observability.scheduler_stage_metrics import (
    SCHEDULER_STAGE_RECV_REQUESTS,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils.device_timer import DeviceTimer, _TimingInterval, device_timer_ctx
from sglang.test.test_utils import CustomTestCase, enter_scope


class FakeInterval(_TimingInterval):
    def __init__(self, milliseconds, ready=False, start=0, stream=0):
        super().__init__(
            start_event=SimpleNamespace(
                elapsed_time=lambda end: end.timestamp - start, query=lambda: True
            ),
            end_event=self,
            stream=stream,
        )
        self.timestamp = start + milliseconds
        self.ready = ready

    def end(self, metadata):
        self.metadata = metadata

    def query(self):
        return self.ready

    def elapsed_time(self):
        return self.start_event.elapsed_time(self)


def capture_timing(timer, *intervals):
    timing = FpmTiming()
    with patch.object(_TimingInterval, "create", side_effect=intervals):
        with timer.capture(timing):
            for _ in intervals:
                with device_timer_ctx(timer, "test"):
                    pass
    timing.seal()
    return timing


class TestDeviceTimerCapture(unittest.TestCase):
    def test_intervals_use_selected_device(self):
        for track_stream in (False, True):
            with (
                self.subTest(track_stream=track_stream),
                patch(
                    "sglang.srt.utils.device_timer.torch.get_device_module"
                ) as module,
            ):
                device = module.return_value
                interval = _TimingInterval.create(track_stream=track_stream)
                interval.end({})
            self.assertEqual(device.Event.call_count, 2)
            device.Event.assert_called_with(enable_timing=True)
            self.assertEqual(device.Event.return_value.record.call_count, 2)
            self.assertEqual(device.current_stream.call_count, int(track_stream))
            self.assertIs(
                interval.stream,
                device.current_stream.return_value if track_stream else None,
            )

    def test_wrapper_cleans_up_after_exception(self):
        timer = DeviceTimer()
        forward = Mock(side_effect=ValueError("forward failed"))
        wrapped = wrap_forward_with_fpm(forward, timer)
        batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
        with self.assertRaisesRegex(ValueError, "forward failed"):
            wrapped(batch)
        self.assertIsNone(timer._observer)

    def test_overlapping_groups_keep_their_own_spans(self):
        reporter = Mock()
        timer = DeviceTimer(reporter)
        pending = [FakeInterval(5, start=4), FakeInterval(3, start=12)]
        first = capture_timing(timer, FakeInterval(2, ready=True), pending[0])
        second = capture_timing(timer, pending[1])
        # A later unobserved interval must not extend either captured span.
        with patch.object(
            _TimingInterval, "create", return_value=FakeInterval(99, True)
        ):
            with timer.wrap({}):
                pass
        callbacks = [Mock(), Mock()]
        first.when_ready(callbacks[0])
        second.when_ready(callbacks[1])
        timer._report()
        callbacks[0].assert_not_called()
        callbacks[1].assert_not_called()
        pending[0].ready = True
        self.assertTrue(timer.is_active())
        callbacks[0].assert_not_called()
        timer._report()
        callbacks[0].assert_called_once_with(0.009)  # Includes the 2 ms gap.
        callbacks[1].assert_not_called()
        pending[1].ready = True
        self.assertFalse(timer.is_active())
        callbacks[1].assert_not_called()
        timer._report()
        callbacks[1].assert_called_once_with(0.003)
        self.assertTrue(all(interval.observer is None for interval in pending))
        self.assertEqual(reporter.call_count, 4)
        timer._report()
        callbacks[0].assert_called_once()
        callbacks[1].assert_called_once()

    def test_empty_zero_and_cross_stream_spans(self):
        for intervals, expected in (
            ([], 0.0),
            ([FakeInterval(0, True)], 0.0),
            ([FakeInterval(7, True)], 0.007),
            ([FakeInterval(2, True, stream=1), FakeInterval(5, True, stream=2)], None),
        ):
            with self.subTest(expected=expected, intervals=len(intervals)):
                for interval in intervals:
                    interval.elapsed_time = Mock(side_effect=AssertionError)
                timing = capture_timing(DeviceTimer(), *intervals)
                self.assertEqual(timing.num_intervals, len(intervals))
                callback = Mock()
                timing.when_ready(callback)
                callback.assert_called_once_with(expected)


class TestSpecDecodeLengthSnapshot(CustomTestCase):
    def test_snapshot_precedes_settlement_and_survives_retraction(self):
        for algorithm in (
            SpeculativeAlgorithm.EAGLE,
            SpeculativeAlgorithm.EAGLE3,
            SpeculativeAlgorithm.DFLASH,
            SpeculativeAlgorithm.DSPARK,
        ):
            with self.subTest(algorithm=algorithm):
                reporter = _make_reporter(self, types.SimpleNamespace())
                reporter.scheduler.enable_fpm = True
                reqs = [
                    types.SimpleNamespace(
                        seqlen=length + 1,
                        is_retracted=length == 200,
                        finished=lambda length=length: length == 300,
                        grammar=None,
                        spec_verify_ct=0,
                        spec_num_correct_drafts=0,
                        update_spec_correct_drafts_histogram=Mock(),
                        kv=types.SimpleNamespace(kv_committed_len=0),
                    )
                    for length in (100, 200, 300)
                ]
                batch = types.SimpleNamespace(
                    reqs=reqs,
                    spec_algorithm=algorithm,
                    forward_mode=ForwardMode.DECODE,
                    seq_lens_cpu=None,
                )
                result = GenerationBatchResult(
                    next_token_ids=torch.arange(24),
                    accept_lens=torch.tensor([1, 8, 3]),
                    speculative_output_stride=8,
                )

                def settle_grammar(result, batch):
                    # Mutating request state must not affect the earlier snapshot.
                    for req in batch.reqs:
                        req.seqlen += 20

                processor = types.SimpleNamespace(
                    metrics_reporter=reporter,
                    model_worker=types.SimpleNamespace(on_verify_complete_cpu=Mock()),
                    advance_grammar_fsm=settle_grammar,
                )
                tokens = SchedulerBatchResultProcessor._resolve_spec_v2_tokens(
                    processor, result, batch
                )
                self.assertEqual([len(ids) for ids in tokens], [1, 8, 3])
                self.assertEqual([req.kv.kv_committed_len for req in reqs], [1, 0, 0])
                metrics = reporter._build_scheduled_request_metrics(batch, result)
                self.assertEqual(metrics.num_decode_requests, 3)
                self.assertEqual(metrics.sum_decode_kv_tokens, 600)
                self.assertAlmostEqual(metrics.var_decode_kv_tokens, 20000 / 3)

    def test_disabled_and_unsupported_do_not_read_requests(self):
        reporter = _make_reporter(self, types.SimpleNamespace())
        result = GenerationBatchResult()
        reporter.snapshot_spec_decode_metrics(None, result)
        self.assertIsNone(result.fpm_scheduled_requests)
        reporter.scheduler.enable_fpm = True
        for algorithm in (SpeculativeAlgorithm.UNO, SpeculativeAlgorithm.FROZEN_KV_MTP):
            reporter.snapshot_spec_decode_metrics(
                types.SimpleNamespace(spec_algorithm=algorithm), result
            )
            self.assertIsNone(result.fpm_scheduled_requests)


class _FakeReq:
    def __init__(
        self,
        prompt_len: int,
        output_len: int = 0,
    ):
        self.origin_input_ids = list(range(prompt_len))
        self.output_ids = list(range(output_len))
        self.seqlen = prompt_len + output_len


class _FakeForwardMode:
    def __init__(self, *, is_mixed: bool = False, is_extend: bool = False):
        self._is_mixed = is_mixed
        self._is_extend = is_extend

    def is_mixed(self):
        return self._is_mixed

    def is_extend(self, include_draft_extend_v2: bool = False):
        return self._is_extend

    def is_decode(self):
        return not self._is_mixed and not self._is_extend


class _CollectingPublisher:
    def __init__(self):
        self.metrics = []

    def publish(self, metrics):
        self.metrics.append(metrics)

    def set_idle(self, idle):
        self.idle = idle


class _DummyPublisherThread:
    def __init__(self, endpoint: str, worker_id: str, dp_rank: int, **_: object):
        self.endpoint = endpoint
        self.worker_id = worker_id
        self.dp_rank = dp_rank

    def shutdown(self):
        pass


def _publish_server_args(test, **fields):
    """Install reporter configuration and rank overrides, with test cleanup."""
    fields.setdefault("decode_log_interval", 40)
    override = get_context().override_server_args(**fields)
    server_args = override.install()
    test.addCleanup(override.restore)
    enter_scope(
        test,
        get_parallel().override(
            tp_rank=0,
            attn_tp_rank=0,
            attn_cp_rank=0,
            moe_ep_rank=0,
            attn_dp_rank=0,
            dp_rank=0,
            moe_ep_size=1,
            moe_dp_size=1,
            moe_tp_size=1,
            pp_rank=0,
            pp_size=1,
        ),
    )
    return server_args


class _CollectingMetricsCollector:
    def __init__(self):
        self.forward_pass_interference = []
        self.schedule_to_result = []

    def observe_forward_pass_interference(self, **kwargs):
        self.forward_pass_interference.append(kwargs)

    def observe_schedule_to_result_latency(self, **kwargs):
        self.schedule_to_result.append(kwargs)

    def increment_forward_execution_seconds(self, **kwargs):
        pass


class _FakeDeviceTimer:
    """CPU stand-in for the CUDA-event device timer.

    `record` starts one forward segment with the given device time; `_report`
    hands completed segments to every reporter in start order and stops at the
    first one still running, as the real timer does with its end events.
    """

    def __init__(self, reporter):
        self._reporters = [reporter]
        self._segments = deque()
        self.num_started = 0

    def add_reporter(self, reporter):
        self._reporters.append(reporter)

    def record(self, seconds, category="decode", done=True):
        self._segments.append([seconds, category, done])
        self.num_started += 1

    def complete_all(self):
        for segment in self._segments:
            segment[2] = True

    def _report(self):
        while self._segments and self._segments[0][2]:
            seconds, category, _ = self._segments.popleft()
            for reporter in self._reporters:
                reporter(t=seconds, category=category)


def _launch(reporter, batch, *segment_seconds, done=True):
    """Launch `batch` the way `Scheduler.run_batch` brackets its forward."""
    begin = reporter.forward_timer_ordinal()
    for seconds in segment_seconds:
        reporter.forward_pass_device_timer.record(seconds, done=done)
    reporter.stamp_forward_timer_span(batch, begin)
    return batch


def _make_reporter(
    test,
    scheduler,
    *,
    metrics_collector=None,
    current_scheduler_metrics_enabled=False,
    is_stats_logging_rank=True,
) -> SchedulerMetricsReporter:
    if not hasattr(scheduler, "device"):
        scheduler.device = "cpu"
    if not hasattr(scheduler, "run_batch"):
        scheduler.run_batch = lambda *args, **kwargs: types.SimpleNamespace()
    if not hasattr(scheduler, "server_args"):
        scheduler.server_args = _publish_server_args(
            test,
            enable_metrics=False,
            enable_metrics_for_all_schedulers=False,
            kv_events_config=None,
            enable_mfu_metrics=False,
            enable_forward_pass_metrics=False,
        )
    if not hasattr(scheduler, "kv_events_publisher"):
        scheduler.kv_events_publisher = types.SimpleNamespace(
            init_kv_events=lambda *a, **kw: None,
        )
    if not hasattr(scheduler, "tp_workers"):
        scheduler.tp_workers = []
    if not hasattr(scheduler, "tp_worker"):
        scheduler.tp_worker = types.SimpleNamespace(
            model_runner=types.SimpleNamespace(),
        )
    if not hasattr(scheduler, "draft_worker"):
        scheduler.draft_worker = None
    if not hasattr(scheduler, "enable_hierarchical_cache"):
        scheduler.enable_hierarchical_cache = False
    context = types.SimpleNamespace(
        enable_metrics=current_scheduler_metrics_enabled,
        is_stats_logging_rank=is_stats_logging_rank,
        current_scheduler_metrics_enabled=current_scheduler_metrics_enabled,
        enable_kv_cache_events=False,
        collector=metrics_collector,
    )
    return SchedulerMetricsReporter(
        scheduler=scheduler,
        metrics_collector_context=context,
        metrics_collector=metrics_collector,
    )


class TestForwardPassMetrics(unittest.TestCase):
    def setUp(self):
        self.scheduler = types.SimpleNamespace()
        self.scheduler._fpm_worker_id = "worker-7"
        self.scheduler._fpm_dp_rank = 0
        self.scheduler._fpm_publisher = _CollectingPublisher()
        self.scheduler.waiting_queue = []
        self.scheduler.disaggregation_mode = DisaggregationMode.NULL
        self.reporter = _make_reporter(self, self.scheduler)
        self.scheduler.enable_fpm = True

    def test_cache_hit_rate_window_keeps_last_15s_of_tokens(self):
        window = _CacheHitRateWindow()
        self.assertEqual(window.add(hit_tokens=20, total_tokens=100, now=0.0), 0.2)
        self.assertEqual(window.add(hit_tokens=80, total_tokens=100, now=10.0), 0.5)
        self.assertEqual(window.add(hit_tokens=90, total_tokens=100, now=15.0), 0.85)

    def _make_batch(self, **overrides):
        defaults = dict(
            forward_mode=_FakeForwardMode(),
            reqs=[],
            decoding_reqs=[],
            prefix_lens=[],
            prefill_stats=None,
            seq_lens_cpu=[],
            fpm_start_time=100.0,
            forward_timer_span=None,
        )
        defaults.update(overrides)
        return types.SimpleNamespace(**defaults)

    def _emit_ready(self, batch, milliseconds=1):
        timing = capture_timing(DeviceTimer(), FakeInterval(milliseconds, ready=True))
        self.reporter._emit_forward_pass_metrics(
            batch, GenerationBatchResult(fpm_timing=timing)
        )

    def test_emit_mixed_batch_separates_prefill_and_decode(self):
        self.scheduler._fpm_dp_rank = 3
        self.scheduler.waiting_queue = [_FakeReq(6), _FakeReq(4, output_len=2)]

        prefill_a = _FakeReq(10)
        prefill_b = _FakeReq(14)
        decode_req = _FakeReq(8, output_len=3)
        batch = self._make_batch(
            forward_mode=_FakeForwardMode(is_mixed=True, is_extend=True),
            reqs=[prefill_a, prefill_b, decode_req],
            decoding_reqs=[decode_req],
            prefix_lens=[2, 3, decode_req.seqlen - 1],
            prefill_stats=PrefillStats(
                log_input_tokens=12,
                log_hit_tokens=5,
                new_token_ratio=1.0,
                num_running_reqs=types.SimpleNamespace(),
                num_new_seqs=2,
            ),
            seq_lens_cpu=[decode_req.seqlen],
        )

        self._emit_ready(batch, milliseconds=4500)

        self.assertEqual(len(self.scheduler._fpm_publisher.metrics), 1)
        metrics = self.scheduler._fpm_publisher.metrics[0]
        self.assertEqual(metrics.worker_id, "worker-7")
        self.assertEqual(metrics.dp_rank, 3)
        self.assertEqual(metrics.wall_time, 4.5)
        self.assertEqual(metrics.scheduled_requests.num_prefill_requests, 2)
        self.assertEqual(metrics.scheduled_requests.sum_prefill_tokens, 12)
        self.assertEqual(metrics.scheduled_requests.sum_prefill_kv_tokens, 5)
        self.assertEqual(metrics.scheduled_requests.num_decode_requests, 1)
        self.assertEqual(
            metrics.scheduled_requests.sum_decode_kv_tokens, decode_req.seqlen
        )
        self.assertEqual(metrics.queued_requests.num_prefill_requests, 1)
        self.assertEqual(metrics.queued_requests.num_decode_requests, 1)

    def test_dummy_rank_keeps_gpu_time_without_local_scheduled_requests(self):
        self.scheduler.waiting_queue = [_FakeReq(128), _FakeReq(256)]
        batch = self._make_batch(forward_mode=ForwardMode.IDLE)
        self._emit_ready(batch, milliseconds=23)
        metric = self.scheduler._fpm_publisher.metrics[-1]
        self.assertEqual(metric.scheduled_requests.num_prefill_requests, 0)
        self.assertEqual(metric.scheduled_requests.num_decode_requests, 0)
        self.assertAlmostEqual(metric.wall_time, 0.023)
        self.assertEqual(metric.queued_requests.num_prefill_requests, 2)
        self.assertEqual(metric.queued_requests.sum_prefill_tokens, 384)

    def test_scheduler_state_controls_heartbeat_with_metrics_disabled(self):
        self.reporter.record_scheduler_active(event_ns=0)
        self.assertFalse(self.scheduler._fpm_publisher.idle)
        self.reporter.record_scheduler_idle()
        self.assertTrue(self.scheduler._fpm_publisher.idle)
        self.reporter.record_scheduler_active(event_ns=1)
        self.assertFalse(self.scheduler._fpm_publisher.idle)

    def test_emit_skips_uninstrumented_iteration(self):
        for timing in (None, capture_timing(DeviceTimer())):
            with self.subTest(timing=timing):
                self.reporter._emit_forward_pass_metrics(
                    self._make_batch(), GenerationBatchResult(fpm_timing=timing)
                )
                self.assertEqual(self.scheduler._fpm_publisher.metrics, [])

    def test_delayed_timing_uses_frozen_batch_and_queue_stats(self):
        timer = self.reporter.forward_pass_device_timer = DeviceTimer()
        interval = FakeInterval(7)
        timing = capture_timing(timer, interval)
        batch = self._make_batch(seq_lens_cpu=[100, 200])
        self.scheduler.waiting_queue = [_FakeReq(300)]
        result = types.SimpleNamespace(fpm_timing=timing)
        self.reporter._emit_forward_pass_metrics(batch, result)
        self.assertEqual(self.scheduler._fpm_publisher.metrics, [])

        # Later iterations may mutate both sources before timing becomes ready.
        batch.seq_lens_cpu = [900]
        self.scheduler.waiting_queue.clear()
        interval.ready = True
        with patch(
            "sglang.srt.managers.scheduler_components.metrics_reporter.ENABLE_METRICS_DEVICE_TIMER",
            False,
        ):
            self.reporter.update_device_timer()
        metrics = self.scheduler._fpm_publisher.metrics[0]
        self.assertAlmostEqual(metrics.wall_time, 0.007)
        self.assertEqual(metrics.scheduled_requests.num_decode_requests, 2)
        self.assertEqual(metrics.scheduled_requests.sum_decode_kv_tokens, 300)
        self.assertEqual(metrics.queued_requests.sum_prefill_tokens, 300)
        timer._report()
        self.assertEqual(len(self.scheduler._fpm_publisher.metrics), 1)

    def _make_mixed_batch(self, fpm_start_time=100.0):
        prefill_req = _FakeReq(10)
        decode_req = _FakeReq(8, output_len=3)
        return self._make_batch(
            forward_mode=_FakeForwardMode(is_mixed=True, is_extend=True),
            reqs=[prefill_req, decode_req],
            decoding_reqs=[decode_req],
            prefix_lens=[2, decode_req.seqlen - 1],
            prefill_stats=PrefillStats(
                log_input_tokens=1200,
                log_hit_tokens=0,
                new_token_ratio=1.0,
                num_running_reqs=types.SimpleNamespace(),
                num_new_seqs=1,
            ),
            seq_lens_cpu=[decode_req.seqlen],
            fpm_start_time=fpm_start_time,
        )

    def _make_prefill_only_batch(self, num_waiting_decode_reqs):
        return self._make_batch(
            forward_mode=_FakeForwardMode(is_extend=True),
            reqs=[_FakeReq(3118)],
            prefill_stats=PrefillStats(
                log_input_tokens=3118,
                log_hit_tokens=0,
                new_token_ratio=1.0,
                num_running_reqs=types.SimpleNamespace(),
                num_new_seqs=1,
                num_waiting_decode_reqs=num_waiting_decode_reqs,
            ),
        )

    def test_prefill_only_pass_reports_waiting_decode_requests(self):
        for waiting in (0, 4):
            with self.subTest(waiting=waiting):
                metrics_collector = _CollectingMetricsCollector()
                reporter = self._make_device_timed_reporter(metrics_collector)
                batch = _launch(reporter, self._make_prefill_only_batch(waiting), 0.020)

                reporter.observe_forward_pass_interference(batch)

                observation = metrics_collector.forward_pass_interference[0]
                self.assertAlmostEqual(observation["duration_seconds"], 0.020)
                self.assertEqual(observation["phase"], "prefill_only")
                self.assertEqual(observation["decode_reqs"], 0)
                self.assertEqual(observation["waiting_decode_reqs"], waiting)

    def _prefill_stats(self, running_reqs):
        adder = types.SimpleNamespace(
            log_input_tokens=3118,
            log_replay_tokens=0,
            log_hit_tokens=0,
            reprocessed_log_input_tokens=0,
            reprocessed_log_hit_tokens=0,
            log_device_hit_tokens=0,
            log_host_hit_tokens=0,
            log_storage_hit_tokens=0,
            new_token_ratio=1.0,
            can_run_list=[_FakeReq(3118)],
        )

        return PrefillStats.from_adder(adder, running_reqs)

    def test_waiting_decode_requests_exclude_finished_running_requests(self):
        # Finished requests stay in the running batch until the next decode.
        running_reqs = [
            types.SimpleNamespace(finished=lambda done=done: done)
            for done in (False, False, True)
        ]

        stats = self._prefill_stats(running_reqs)

        self.assertEqual(stats.num_running_reqs.total, 3)
        self.assertEqual(stats.num_waiting_decode_reqs, 2)

    def test_prefill_wait_snapshot_survives_overlap_result_processing(self):
        state = {"finished": False}
        running_reqs = [SimpleNamespace(finished=lambda: state["finished"])]
        # Overlap selects prefill before settling the preceding decode result.
        stats = self._prefill_stats(running_reqs)
        state["finished"] = True
        batch = self._make_prefill_only_batch(0)
        batch.prefill_stats = stats
        collector = _CollectingMetricsCollector()
        reporter = self._make_device_timed_reporter(collector)
        reporter.observe_forward_pass_interference(_launch(reporter, batch, 0.020))

        self.assertEqual(
            collector.forward_pass_interference[0]["waiting_decode_reqs"], 1
        )
        # A request finishing in flight can be counted once, never indefinitely.
        self.assertEqual(self._prefill_stats(running_reqs).num_waiting_decode_reqs, 0)

    def _make_device_timed_reporter(self, metrics_collector, **reporter_kwargs):
        """Build a reporter whose forward timing comes from a fake device timer."""
        self.scheduler.enable_fpm = False
        with (
            patch(
                "sglang.srt.managers.scheduler_components.metrics_reporter.ENABLE_METRICS_DEVICE_TIMER",
                True,
            ),
            patch(
                "sglang.srt.managers.scheduler_components.metrics_reporter.DeviceTimer",
                _FakeDeviceTimer,
            ),
        ):
            return _make_reporter(
                self,
                self.scheduler,
                metrics_collector=metrics_collector,
                current_scheduler_metrics_enabled=True,
                **reporter_kwargs,
            )

    def test_schedule_to_result_observation_does_not_require_fpm(self):
        self.scheduler.enable_fpm = False
        metrics_collector = _CollectingMetricsCollector()
        self.reporter = _make_reporter(
            self,
            self.scheduler,
            metrics_collector=metrics_collector,
            current_scheduler_metrics_enabled=True,
        )
        batch = self._make_mixed_batch()

        with patch(
            "sglang.srt.managers.scheduler_components.metrics_reporter.time.monotonic",
            return_value=100.075,
        ):
            self.reporter.observe_forward_pass_interference(batch)

        self.assertEqual(len(metrics_collector.schedule_to_result), 1)
        observation = metrics_collector.schedule_to_result[0]
        self.assertAlmostEqual(observation["duration_seconds"], 0.075, places=4)
        self.assertEqual(observation["phase"], "mixed")
        self.assertEqual(observation["prefill_tokens"], 1200)
        self.assertEqual(observation["decode_reqs"], 1)
        # No device timer, so there is no interval bounded by this batch's own
        # forward pass; nothing is exported rather than a loop-bounded stand-in.
        self.assertEqual(metrics_collector.forward_pass_interference, [])

    def test_forward_duration_excludes_next_batch_scheduling_and_launch(self):
        """`event_loop_overlap` selects (and on the steady-state branch also
        launches) batch n+1 before it processes n's result, so a wall interval
        started at n's selection absorbs n+1's scheduler work. The forward-pass
        duration must stay pinned to n's own forward; only schedule-to-result
        latency may grow with that extra work.
        """

        def run_step(next_batch_overhead: float) -> _CollectingMetricsCollector:
            metrics_collector = _CollectingMetricsCollector()
            reporter = self._make_device_timed_reporter(metrics_collector)
            # Batch n's own forward: 10 ms on device.
            batch = _launch(
                reporter, self._make_mixed_batch(fpm_start_time=100.0), 0.010
            )
            # The loop then picks and launches batch n+1 before popping n's
            # result off the queue, which is what the extra overhead stands for.
            with patch(
                "sglang.srt.managers.scheduler_components.metrics_reporter.time.monotonic",
                return_value=100.020 + next_batch_overhead,
            ):
                reporter.observe_forward_pass_interference(batch)
            return metrics_collector

        without_overhead = run_step(0.0)
        with_overhead = run_step(0.050)

        for metrics_collector in (without_overhead, with_overhead):
            self.assertEqual(len(metrics_collector.forward_pass_interference), 1)

        # The forward pass took 10 ms in both steps; the next batch's work must
        # not show up here.
        self.assertAlmostEqual(
            without_overhead.forward_pass_interference[0]["duration_seconds"],
            0.010,
            places=6,
        )
        self.assertAlmostEqual(
            with_overhead.forward_pass_interference[0]["duration_seconds"],
            0.010,
            places=6,
        )

        # It shows up here instead, under a name that says so.
        for metrics_collector in (without_overhead, with_overhead):
            self.assertEqual(len(metrics_collector.schedule_to_result), 1)
        self.assertAlmostEqual(
            without_overhead.schedule_to_result[0]["duration_seconds"],
            0.020,
            places=6,
        )
        self.assertAlmostEqual(
            with_overhead.schedule_to_result[0]["duration_seconds"],
            0.070,
            places=6,
        )

    def test_prometheus_drain_still_feeds_fpm(self):
        """Prometheus draining a shared timer must also complete FPM's observer."""
        metrics_collector = _CollectingMetricsCollector()
        with patch(
            "sglang.srt.managers.scheduler_components.metrics_reporter.ENABLE_METRICS_DEVICE_TIMER",
            True,
        ):
            reporter = _make_reporter(
                self,
                self.scheduler,
                metrics_collector=metrics_collector,
                current_scheduler_metrics_enabled=True,
            )
        self.scheduler.enable_fpm = True
        timer = reporter.forward_pass_device_timer
        begin = reporter.forward_timer_ordinal()
        interval = FakeInterval(12)
        timing = capture_timing(timer, interval)
        batch = self._make_mixed_batch()
        reporter.stamp_forward_timer_span(batch, begin)

        interval.ready = True
        reporter.observe_forward_pass_interference(batch)
        reporter._emit_forward_pass_metrics(
            batch, GenerationBatchResult(fpm_timing=timing)
        )

        self.assertEqual(len(metrics_collector.forward_pass_interference), 1)
        self.assertAlmostEqual(
            metrics_collector.forward_pass_interference[0]["duration_seconds"], 0.012
        )
        published = self.scheduler._fpm_publisher.metrics
        self.assertEqual(len(published), 1)
        self.assertAlmostEqual(published[0].wall_time, 0.012)
        timer._report()
        self.assertEqual(len(published), 1)

    def test_forward_pass_device_time_is_never_borrowed_across_batches(self):
        """A batch exports the sum of its own segments once all have completed.

        Speculative decoding launches several segments per batch. A batch with
        a segment still running, or with no timed segment, exports no duration;
        it does not take another batch's time, and a segment that completes
        late is not charged to the batch processed after it.
        """
        metrics_collector = _CollectingMetricsCollector()
        reporter = self._make_device_timed_reporter(metrics_collector)
        timer = reporter.forward_pass_device_timer

        first = _launch(reporter, self._make_mixed_batch(), 0.004, 0.006)
        reporter.observe_forward_pass_interference(first)
        # Second segment still running on the device when its result is processed.
        late = self._make_mixed_batch()
        begin = reporter.forward_timer_ordinal()
        timer.record(0.100)
        timer.record(0.500, done=False)
        reporter.stamp_forward_timer_span(late, begin)
        reporter.observe_forward_pass_interference(late)
        timer.complete_all()
        # Went through run_batch but launched no timed segment.
        untimed = _launch(reporter, self._make_mixed_batch())
        reporter.observe_forward_pass_interference(untimed)
        last = _launch(reporter, self._make_mixed_batch(), 0.020)
        reporter.observe_forward_pass_interference(last)
        # Never launched, e.g. a disaggregated-decode placeholder.
        reporter.observe_forward_pass_interference(self._make_mixed_batch())

        durations = [
            o["duration_seconds"] for o in metrics_collector.forward_pass_interference
        ]
        self.assertEqual(len(durations), 2)
        self.assertAlmostEqual(durations[0], 0.010)
        self.assertAlmostEqual(durations[1], 0.020)
        self.assertEqual(len(metrics_collector.schedule_to_result), 5)

    def test_real_device_timer_ordinals_match_reported_segments(self):
        """`run_batch` stamps spans from `DeviceTimer.num_started` while the
        reporter numbers segments by counting reports; the two must agree on
        the real timer, not only on the fake one.
        """
        clock = {"ms": 0.0}

        class _CpuEvent:
            def __init__(self, enable_timing=False):
                self.ms = None

            def record(self):
                self.ms = clock["ms"]

            def query(self):
                return True

            def elapsed_time(self, end_event):
                return end_event.ms - self.ms

        metrics_collector = _CollectingMetricsCollector()
        with patch(
            "sglang.srt.managers.scheduler_components.metrics_reporter.ENABLE_METRICS_DEVICE_TIMER",
            True,
        ):
            reporter = _make_reporter(
                self,
                self.scheduler,
                metrics_collector=metrics_collector,
                current_scheduler_metrics_enabled=True,
            )
        timer = reporter.forward_pass_device_timer
        self.assertIs(type(timer), DeviceTimer)

        def launch(*segment_ms):
            batch = self._make_mixed_batch()
            begin = reporter.forward_timer_ordinal()
            for ms in segment_ms:
                with timer.wrap(metadata={"category": "decode"}):
                    clock["ms"] += ms
            reporter.stamp_forward_timer_span(batch, begin)
            return batch

        with patch(
            "sglang.srt.utils.device_timer.torch.get_device_module",
            return_value=SimpleNamespace(Event=_CpuEvent),
        ):
            # Overlap order: n+1 launches before n's result is processed.
            first = launch(4.0, 6.0)
            second = launch(30.0)
            reporter.observe_forward_pass_interference(first)
            reporter.observe_forward_pass_interference(second)

        durations = [
            o["duration_seconds"] for o in metrics_collector.forward_pass_interference
        ]
        self.assertEqual(len(durations), 2)
        self.assertAlmostEqual(durations[0], 0.010)
        self.assertAlmostEqual(durations[1], 0.030)

    def test_forward_pass_composition_does_not_need_seq_lens_cpu(self):
        """Decode requests are counted from the batch, not from `seq_lens_cpu`,
        which spec-v2 overlap may leave as None. A decode batch that DP
        attention runs as a 1-token extend still counts as decode.
        """
        metrics_collector = _CollectingMetricsCollector()
        self.reporter = _make_reporter(
            self,
            self.scheduler,
            metrics_collector=metrics_collector,
            current_scheduler_metrics_enabled=True,
        )
        reqs = [_FakeReq(8, output_len=3), _FakeReq(5, output_len=1)]
        spec_v2_decode = self._make_batch(reqs=reqs, seq_lens_cpu=None)
        dp_converted_decode = self._make_batch(
            forward_mode=_FakeForwardMode(is_extend=True),
            reqs=reqs,
            decoding_reqs=reqs,
            seq_lens_cpu=None,
        )

        for batch in (spec_v2_decode, dp_converted_decode):
            self.reporter.observe_forward_pass_interference(batch)

        self.assertEqual(
            [
                (o["phase"], o["prefill_tokens"], o["decode_reqs"])
                for o in metrics_collector.schedule_to_result
            ],
            [("decode_only", 0, 2)] * 2,
        )

    def _drive_event_loop(self, event_loop, *, disable_overlap):
        """Run three decode batches through a real scheduler loop.

        The real `Scheduler.run_batch` brackets each forward, and the model
        worker records the segment the model runner's device-timer wrap would.
        Every forward has finished on the device before any result is
        processed. Returns the launch/process order (by forward iteration) and
        the exported forward durations in result-processing order.

        `run_batch` takes its non-overlap branch even for the overlap loop: the
        span is stamped after both branches, and the loop order is what differs.
        """
        metrics_collector = _CollectingMetricsCollector()
        reporter = self._make_device_timed_reporter(metrics_collector)
        device_seconds = iter([0.010, 0.030, 0.020])
        events = []

        def forward_batch_generation(batch, **_kwargs):
            events.append(("launch", batch.forward_iter))
            reporter.forward_pass_device_timer.record(next(device_seconds))
            return GenerationBatchResult()

        def process_batch_result(batch, _result):
            events.append(("process", batch.forward_iter))
            reporter.observe_forward_pass_interference(batch)

        schedule = [
            ScheduleBatch(
                reqs=[],
                forward_mode=ForwardMode.DECODE,
                spec_algorithm=SpeculativeAlgorithm.NONE,
                seq_lens_cpu=[8],
            )
            for _ in range(3)
        ] + [None]

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.gracefully_exit = False
        scheduler._engine_paused = False
        scheduler._sched_idled = False
        scheduler.forward_ct = 0
        scheduler.is_generation = True
        scheduler.enable_overlap = False
        scheduler.enable_pdmux = False
        scheduler.enable_unified_memory = False
        scheduler.enable_unified_cache_external_linker = False
        scheduler.enable_dp_attention = False
        scheduler.attn_dp_enabled = False
        scheduler.spec_algorithm = SpeculativeAlgorithm.NONE
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.scheduler_stage_metrics = None
        scheduler.scripted_scheduler_hook = None
        scheduler.forward_sleep_time = None
        scheduler.profiler_manager = types.SimpleNamespace(
            _profile_batch_predicate=Mock()
        )
        scheduler.future_map = None
        scheduler.model_worker = types.SimpleNamespace(
            forward_batch_generation=forward_batch_generation
        )
        scheduler.update_cache_from_scheduler = Mock()
        scheduler._copy_auxiliary_output_to_cpu = Mock()
        scheduler.metrics_reporter = types.SimpleNamespace(
            record_scheduler_active=Mock(),
            forward_timer_ordinal=reporter.forward_timer_ordinal,
            stamp_forward_timer_span=reporter.stamp_forward_timer_span,
        )
        scheduler.running_batch = None
        scheduler.last_batch = None
        scheduler.ingest_requests = Mock(
            side_effect=[None] * len(schedule) + [StopIteration]
        )
        scheduler.get_next_batch_to_run = Mock(
            side_effect=[
                types.SimpleNamespace(running_batch=None, batch_to_run=batch)
                for batch in schedule
            ]
        )
        scheduler.is_disable_overlap_for_batch = lambda batch, last_batch: (
            disable_overlap and batch is not None and last_batch is not None
        )
        scheduler._apply_war_barrier = Mock()
        scheduler.launch_batch_sample_if_needed = Mock()
        scheduler.on_idle = Mock()
        scheduler.process_batch_result = process_batch_result

        with (
            patch("sglang.srt.managers.scheduler.resolve_forward_inputs"),
            self.assertRaises(StopIteration),
        ):
            event_loop(scheduler)

        durations = [
            o["duration_seconds"] for o in metrics_collector.forward_pass_interference
        ]
        return events, durations

    def test_forward_duration_is_attributed_to_its_own_batch_in_every_loop(self):
        """On the steady-state overlap branch `event_loop_overlap` launches n+1
        before it processes n. If n+1's forward has already finished by then,
        draining the device timer at n's result would charge n+1's forward to
        n. Each sample must be its own batch's forward on that branch, on the
        disable-overlap branch (n is processed before n+1 launches), and in
        the normal loop.
        """
        sequential = [
            ("launch", 1),
            ("process", 1),
            ("launch", 2),
            ("process", 2),
            ("launch", 3),
            ("process", 3),
        ]
        cases = [
            (
                "overlap",
                Scheduler.event_loop_overlap,
                False,
                [
                    ("launch", 1),
                    ("launch", 2),
                    ("process", 1),
                    ("launch", 3),
                    ("process", 2),
                    ("process", 3),
                ],
            ),
            ("overlap_disabled", Scheduler.event_loop_overlap, True, sequential),
            ("normal", Scheduler.event_loop_normal, False, sequential),
        ]
        for name, event_loop, disable_overlap, expected_order in cases:
            with self.subTest(name):
                events, durations = self._drive_event_loop(
                    event_loop, disable_overlap=disable_overlap
                )
                self.assertEqual(events, expected_order)
                self.assertEqual(durations, [0.010, 0.030, 0.020])

    def test_forward_pass_metrics_emit_from_one_canonical_pp_rank(self):
        """One batch must produce exactly one sample.

        The attention TP, CP and PP ranks of a batch all run it. FPM and the
        Prometheus series both gate to attention TP and CP rank 0 on the final
        PP stage; otherwise every one of those ranks exports its own timing of
        the same batch under the same series name.
        """
        cases = [
            # pp_rank, pp_size, attn_tp_rank, attn_cp_rank, emits
            (0, 1, 0, 0, True),  # no pipeline parallelism: unchanged
            (0, 2, 0, 0, False),  # earlier PP stage stays silent
            (1, 2, 0, 0, True),  # final PP stage is the canonical rank
            (1, 2, 1, 0, False),  # non-zero attention TP rank stays silent
            (1, 2, 0, 1, False),  # non-zero attention CP rank stays silent
            (2, 4, 0, 0, False),
            (3, 4, 0, 0, True),
        ]
        for pp_rank, pp_size, attn_tp_rank, attn_cp_rank, emits in cases:
            with (
                self.subTest(
                    pp_rank=pp_rank,
                    pp_size=pp_size,
                    attn_tp=attn_tp_rank,
                    attn_cp=attn_cp_rank,
                ),
                get_parallel().override(
                    pp_rank=pp_rank,
                    pp_size=pp_size,
                    tp_rank=attn_cp_rank * 2 + attn_tp_rank,
                    tp_size=4,
                    attn_tp_rank=attn_tp_rank,
                    attn_tp_size=2,
                    attn_cp_rank=attn_cp_rank,
                    attn_cp_size=2,
                    moe_tp_size=4,
                ),
            ):
                metrics_collector = _CollectingMetricsCollector()
                reporter = self._make_device_timed_reporter(
                    metrics_collector, is_stats_logging_rank=attn_tp_rank == 0
                )
                batch = _launch(reporter, self._make_mixed_batch(), 0.010)
                with patch(
                    "sglang.srt.managers.scheduler_components.metrics_reporter.time.monotonic",
                    return_value=100.075,
                ):
                    reporter.observe_forward_pass_interference(batch)

                expected = 1 if emits else 0
                self.assertEqual(
                    len(metrics_collector.forward_pass_interference), expected
                )
                self.assertEqual(len(metrics_collector.schedule_to_result), expected)
                self.assertEqual(reporter.forward_pass_metrics_enabled, emits)

    def test_forward_pass_metrics_are_off_under_pdmux(self):
        """PD multiplexing runs decode and split prefill concurrently on one
        device timer, so a batch's own forward time is not attributable."""
        with get_context().override_server_args(enable_pdmux=True):
            reporter = self._make_device_timed_reporter(_CollectingMetricsCollector())
        self.assertFalse(reporter.forward_pass_metrics_enabled)

    def test_disaggregated_batch_selectors_stamp_schedule_start(self):
        """schedule_to_result is sampled only from batches stamped with their
        selection start, so the PD-disaggregation selectors must stamp it too.
        """
        running = ScheduleBatch(reqs=[_FakeReq(4)])
        prefill = ScheduleBatch(reqs=[_FakeReq(4)])
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.scheduler_stage_metrics = None
        scheduler.get_new_prebuilt_batch = Mock(return_value=None)
        scheduler.update_running_batch = lambda batch: batch
        scheduler.dp_attn_adapter = types.SimpleNamespace(
            maybe_prepare_mlp_sync_batch=lambda batch: batch
        )
        scheduler.process_pending_chunked_abort = Mock()
        scheduler._process_hicache_events = Mock()
        scheduler.resolve_waiting_queue_bootstrap = Mock()
        scheduler.process_prefill_chunk = Mock()
        scheduler.get_new_batch_prefill = Mock(
            return_value=types.SimpleNamespace(
                batch_to_run=prefill, running_batch=running
            )
        )

        selectors = [
            ("decode", lambda: scheduler.get_next_disagg_decode_batch_to_run(running)),
            (
                "prefill",
                lambda: scheduler.get_next_disagg_prefill_batch_to_run(running, None),
            ),
        ]
        for name, select in selectors:
            with self.subTest(name), patch("time.monotonic", return_value=42.0):
                self.assertEqual(select().batch_to_run.fpm_start_time, 42.0)

    def test_disagg_prefill_queued_metrics_include_compute_waiting_queue(self):
        self.scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        self.scheduler.disagg_prefill_bootstrap_queue = types.SimpleNamespace(
            queue=[_FakeReq(100)],
        )
        self.scheduler.waiting_queue = [_FakeReq(200), _FakeReq(50)]
        batch = self._make_batch()

        self._emit_ready(batch)

        metrics = self.scheduler._fpm_publisher.metrics[0]
        self.assertEqual(metrics.queued_requests.num_prefill_requests, 3)
        self.assertEqual(metrics.queued_requests.sum_prefill_tokens, 350)
        self.assertEqual(metrics.queued_requests.num_decode_requests, 0)

    def test_disagg_decode_queued_metrics(self):
        self.scheduler.disaggregation_mode = DisaggregationMode.DECODE
        self.scheduler.disagg_decode_prealloc_queue = types.SimpleNamespace(
            queue=[_FakeReq(10, output_len=5), _FakeReq(20, output_len=10)],
        )
        self.scheduler.disagg_decode_transfer_queue = types.SimpleNamespace(
            queue=[_FakeReq(30, output_len=15)],
        )
        batch = self._make_batch()

        self._emit_ready(batch)

        metrics = self.scheduler._fpm_publisher.metrics[0]
        self.assertEqual(metrics.queued_requests.num_prefill_requests, 0)
        self.assertEqual(metrics.queued_requests.num_decode_requests, 3)
        self.assertEqual(metrics.queued_requests.sum_decode_kv_tokens, 15 + 30 + 45)

    def test_init_metrics_uses_server_worker_id(self):
        scheduler = types.SimpleNamespace()
        original_forward = scheduler.run_batch = lambda batch: types.SimpleNamespace()
        scheduler.server_args = _publish_server_args(
            self,
            enable_metrics=False,
            enable_metrics_for_all_schedulers=False,
            extra_metric_labels=None,
            enable_forward_pass_metrics=True,
            forward_pass_metrics_worker_id="endpoint-42",
            forward_pass_metrics_ipc_name=None,
            kv_events_config=None,
        )
        enter_scope(self, get_parallel().override(pp_rank=0, pp_size=1, dp_rank=2))
        scheduler.enable_kv_cache_events = False

        with patch(
            "sglang.srt.observability.forward_pass_metrics._FpmPublisherThread",
            _DummyPublisherThread,
        ):
            reporter = _make_reporter(self, scheduler)

        self.assertTrue(scheduler.enable_fpm)
        self.assertIs(scheduler.run_batch.__wrapped__, original_forward)
        batch = types.SimpleNamespace(forward_mode=ForwardMode.DECODE)
        self.assertEqual(scheduler.run_batch(batch).fpm_timing.num_intervals, 0)
        self.assertEqual(scheduler._fpm_worker_id, "endpoint-42")
        self.assertEqual(scheduler._fpm_dp_rank, 2)
        self.assertEqual(scheduler._fpm_publisher.worker_id, "endpoint-42")
        self.assertEqual(scheduler._fpm_publisher.dp_rank, 2)
        self.assertTrue(scheduler._fpm_publisher.endpoint.startswith("ipc://"))
        # The bag is what makes the write a bag write: an instance mutation
        # would still show up in the resolved dict through its ServerArgs base.
        endpoint = get_observability().forward_pass_metrics_ipc_name
        self.assertTrue(endpoint.startswith("ipc://"))
        self.assertEqual(
            get_context().resolved_server_args_dict()["forward_pass_metrics_ipc_name"],
            endpoint,
        )
        self.assertIsNone(scheduler.server_args.forward_pass_metrics_ipc_name)

    def test_init_fpm_disabled_on_non_last_pp_rank(self):
        scheduler = types.SimpleNamespace()
        original_forward = scheduler.run_batch = lambda: types.SimpleNamespace()
        scheduler.server_args = _publish_server_args(
            self,
            enable_metrics=False,
            enable_metrics_for_all_schedulers=False,
            extra_metric_labels=None,
            enable_forward_pass_metrics=True,
            forward_pass_metrics_worker_id="endpoint-42",
            forward_pass_metrics_ipc_name=None,
            kv_events_config=None,
        )
        enter_scope(self, get_parallel().override(pp_rank=0, pp_size=2))
        scheduler.enable_kv_cache_events = False

        with patch(
            "sglang.srt.observability.forward_pass_metrics._FpmPublisherThread",
            _DummyPublisherThread,
        ):
            reporter = _make_reporter(self, scheduler)

        self.assertFalse(scheduler.enable_fpm)
        self.assertIs(scheduler.run_batch, original_forward)

    def test_fpm_off_keeps_original_forward(self):
        class Scheduler:
            def run_batch(self):
                return types.SimpleNamespace()

        scheduler = Scheduler()
        _make_reporter(self, scheduler)
        self.assertFalse(scheduler.enable_fpm)
        self.assertNotIn("run_batch", vars(scheduler))
        self.assertIs(scheduler.run_batch.__func__, Scheduler.run_batch)

    def test_prebuilt_recursive_idle_forward_owns_timing_once(self):
        timer = DeviceTimer()
        scheduler = types.SimpleNamespace()
        result = GenerationBatchResult()

        def forward(batch):
            if batch.forward_mode.is_prebuilt():
                return SchedulerDisaggregationDecodeMixin._run_batch_prebuilt(
                    scheduler, batch
                )
            with timer.wrap({"category": "idle"}):
                pass
            return result

        scheduler.run_batch = wrap_forward_with_fpm(forward, timer)
        batch = types.SimpleNamespace(
            forward_mode=ForwardMode.PREBUILT,
            inner_idle_batch=types.SimpleNamespace(forward_mode=ForwardMode.IDLE),
        )
        with patch.object(
            _TimingInterval, "create", return_value=FakeInterval(5, ready=True)
        ) as create:
            actual = scheduler.run_batch(batch)
        self.assertIs(actual, result)
        self.assertIsNone(batch.inner_idle_batch)
        self.assertEqual(actual.fpm_timing.num_intervals, 1)
        published = []
        actual.fpm_timing.when_ready(published.append)
        self.assertEqual(published, [0.005])
        create.assert_called_once()
        # A prebuilt with no inner forward does not invent a timing group.
        self.assertIsNone(scheduler.run_batch(batch).fpm_timing)

    def test_pp_launch_and_relay_keep_rank_local_timing(self):
        for speculative in (False, True):
            with self.subTest(speculative=speculative):
                self.scheduler._fpm_publisher.metrics.clear()
                timer, interval = DeviceTimer(), FakeInterval(9)
                receiver = SchedulerPPMixin()
                receiver._pp_spec_relay = speculative
                receiver.forward_stream_ctx = nullcontext()
                receiver.forward_stream = Mock()
                receiver.schedule_stream = object()
                receiver.pp_send_done_event = None
                receiver.pp_proxy_recv_event = None
                receiver.device_module = SimpleNamespace(
                    Event=Mock, current_stream=Mock()
                )
                receiver.pp_group = SimpleNamespace(
                    is_last_rank=True, is_first_rank=False
                )
                receiver.future_map = Mock()
                req = _FakeReq(128, output_len=1)
                req.rid = "req-0"
                batch = ScheduleBatch(
                    reqs=[req],
                    forward_mode=ForwardMode.DECODE,
                    spec_algorithm=SpeculativeAlgorithm.EAGLE3
                    if speculative
                    else SpeculativeAlgorithm.NONE,
                    seq_lens=torch.tensor([128]),
                    seq_lens_cpu=None if speculative else torch.tensor([128]),
                    req_pool_indices=torch.tensor([0]),
                    return_logprob=False,
                )
                result = GenerationBatchResult(next_token_ids=torch.tensor([7]))
                if speculative:
                    result.next_token_ids = torch.tensor([7, 8, 0])
                    result.accept_lens = torch.tensor([2])
                    result.new_seq_lens = torch.tensor([130])
                    result.next_draft_input = SimpleNamespace(
                        bonus_tokens=torch.tensor([8]), topk_p=None
                    )

                def forward(batch, pp_proxy_tensors):
                    with timer.wrap({}):
                        pass
                    return result

                receiver.run_batch = wrap_forward_with_fpm(forward, timer)
                metadata, wire_queue = [None], deque()
                with (
                    patch.object(_TimingInterval, "create", return_value=interval),
                    patch("sglang.srt.managers.scheduler_pp_mixin.set_time_batch"),
                    patch(
                        "sglang.srt.managers.scheduler_pp_mixin.get_spec",
                        return_value=SimpleNamespace(speculative_num_draft_tokens=3),
                    ),
                ):
                    receiver._pp_launch_batch(0, batch, None, metadata, wire_queue)
                    wire = wire_queue[0][1]
                    self.assertTrue(
                        all(torch.is_tensor(t) for t in wire.tensors.values())
                    )
                    rebuilt = receiver._pp_prep_batch_result(batch, metadata[0], wire)
                self.assertIs(rebuilt.fpm_timing, result.fpm_timing)
                snapshot = metadata[0].fwd_batch if speculative else batch
                self.reporter.snapshot_spec_decode_metrics(snapshot, rebuilt)
                self.reporter._emit_forward_pass_metrics(snapshot, rebuilt)
                self.assertEqual(self.scheduler._fpm_publisher.metrics, [])
                interval.ready = True
                timer._report()
                metrics = self.scheduler._fpm_publisher.metrics
                self.assertEqual(len(metrics), 1)
                self.assertAlmostEqual(metrics[0].wall_time, 0.009)
                self.assertEqual(
                    metrics[0].scheduled_requests.sum_decode_kv_tokens, 128
                )

    def test_pp_skipped_output_comm_keeps_timing(self):
        timing = object()
        metadata = PPBatchMetadata(can_run_cuda_graph=True, fpm_timing=timing)
        scheduler = types.SimpleNamespace(
            device="cpu",
            device_module=types.SimpleNamespace(
                Event=Mock, current_stream=lambda: object()
            ),
        )
        wire, result, _ = SchedulerPPMixin._pp_make_skip_output_result(
            scheduler, types.SimpleNamespace(reqs=[object()]), metadata
        )
        self.assertIsNone(wire)
        self.assertTrue(result.skipped_output_comm)
        self.assertIs(result.fpm_timing, timing)


class TestFpmPublisherIdleState(unittest.TestCase):
    def test_queue_timeout_does_not_turn_a_long_or_dummy_forward_into_heartbeat(self):
        publisher = object.__new__(_FpmPublisherThread)
        publisher._running = True
        publisher._idle = False
        publisher._seq = count()
        publisher._worker_id = "worker"
        publisher._dp_rank = 1
        publisher._pub = Mock()
        publisher._zmq = types.SimpleNamespace(Again=RuntimeError, NOBLOCK=1)
        dummy = ForwardPassMetrics(worker_id="worker", dp_rank=1, wall_time=0.025)
        actions = iter(["busy", dummy, "busy", "idle", "busy", "stop"])

        def next_item(**kwargs):
            item = next(actions)
            if item == "stop":
                return None
            if isinstance(item, str):
                publisher.set_idle(item == "idle")
                raise queue.Empty
            return item

        publisher._queue = types.SimpleNamespace(get=next_item)
        with patch(
            "sglang.srt.observability.forward_pass_metrics.time.monotonic",
            side_effect=count(0, 2),
        ):
            publisher._run()
        messages = [
            decode(call.args[0][2])
            for call in publisher._pub.send_multipart.call_args_list
        ]
        self.assertEqual(len(messages), 2)
        self.assertEqual([m.wall_time for m in messages], [0.025, 0.0])
        self.assertEqual([m.counter_id for m in messages], [0, 1])
        self.assertTrue(
            all(m.scheduled_requests.num_decode_requests == 0 for m in messages)
        )


def _dp_balance_stats(
    *,
    local_tokens: int,
    global_num_tokens: list,
    sync_wait_seconds: float,
    local_attention_pairs: int | None = None,
    global_attention_pairs: list | None = None,
):
    """One-token rows unless the pairs are given explicitly."""
    return DPBalanceStats.create(
        local_tokens=local_tokens,
        global_num_tokens=global_num_tokens,
        local_attention_pairs=(
            local_tokens if local_attention_pairs is None else local_attention_pairs
        ),
        global_attention_pairs=(
            global_num_tokens
            if global_attention_pairs is None
            else global_attention_pairs
        ),
        sync_wait_seconds=sync_wait_seconds,
    )


class TestDPBalanceMetrics(CustomTestCase):
    def _reporter_with_collector(self, attn_dp_rank: int):
        scheduler = types.SimpleNamespace(
            running_batch=types.SimpleNamespace(reqs=[]),
            waiting_queue=[],
            grammar_manager=[],
            enable_priority_scheduling=False,
            disaggregation_mode=DisaggregationMode.NULL,
        )
        scheduler.server_args = _publish_server_args(
            self,
            enable_metrics=False,
            enable_metrics_for_all_schedulers=False,
            kv_events_config=None,
            enable_mfu_metrics=False,
            enable_forward_pass_metrics=False,
        )
        enter_scope(
            self,
            get_parallel().override(
                tp_size=2,
                tp_rank=attn_dp_rank,
                attn_dp_size=2,
                attn_dp_rank=attn_dp_rank,
                moe_tp_size=2,
            ),
        )
        reporter = _make_reporter(self, scheduler)

        registry = prometheus_client.CollectorRegistry()
        labels = {"model_name": "test", "moe_ep_rank": 0, "dp_rank": attn_dp_rank}
        with patch.multiple(
            prometheus_client,
            **{
                kind: partial(getattr(prometheus_client, kind), registry=registry)
                for kind in ("Counter", "Gauge", "Histogram", "Summary")
            },
        ):
            reporter.metrics_collector = SchedulerMetricsCollector(
                labels=labels, server_args=scheduler.server_args
            )
        reporter.current_scheduler_metrics_enabled = True
        return reporter, registry, {k: str(v) for k, v in labels.items()}

    def test_step_stats_exported_once_per_step(self):
        reporter, registry, labels = self._reporter_with_collector(attn_dp_rank=0)
        get = registry.get_sample_value
        # Pre-seeded, so ratio charts have both operands before the first idle step.
        self.assertEqual(
            get("sglang:dp_attention_steps_total", {**labels, "rank_state": "idle"}), 0
        )

        batch = types.SimpleNamespace(
            dp_balance_stats=_dp_balance_stats(
                local_tokens=4,
                global_num_tokens=[8, 4],
                sync_wait_seconds=0.002,
                local_attention_pairs=4000,
                global_attention_pairs=[800, 4000],
            )
        )
        reporter.log_batch_result_stats(batch, result=object())
        batch.dp_balance_stats = _dp_balance_stats(
            local_tokens=0, global_num_tokens=[8, 0], sync_wait_seconds=0.004
        )
        reporter.log_batch_result_stats(batch, result=object())
        batch.dp_balance_stats = None
        reporter.log_batch_result_stats(batch, result=object())

        self.assertEqual(
            get("sglang:dp_attention_tokens_total", {**labels, "kind": "scheduled"}), 4
        )
        self.assertEqual(
            get("sglang:dp_attention_tokens_total", {**labels, "kind": "imbalance"}),
            12,
        )
        self.assertEqual(
            get("sglang:dp_attention_steps_total", {**labels, "rank_state": "active"}),
            1,
        )
        self.assertEqual(
            get("sglang:dp_attention_steps_total", {**labels, "rank_state": "idle"}), 1
        )
        self.assertEqual(
            get("sglang:dp_attention_token_imbalance_ratio_count", labels), 2
        )
        self.assertAlmostEqual(
            get("sglang:dp_attention_token_imbalance_ratio_sum", labels),
            8 * 2 / 12 + 2.0,
        )
        # Pairs: the first step's long-context rank is the pair-busiest
        # despite having fewer rows; the idle step adds the peer's 8.
        self.assertEqual(
            get("sglang:dp_attention_pairs_total", {**labels, "kind": "scheduled"}),
            4000,
        )
        self.assertEqual(
            get("sglang:dp_attention_pairs_total", {**labels, "kind": "imbalance"}),
            8,
        )
        self.assertAlmostEqual(
            get("sglang:dp_attention_sync_wait_seconds_sum", labels), 0.006
        )

    def test_engine_ratio_reported_by_dp_rank_zero_only(self):
        reporter, registry, labels = self._reporter_with_collector(attn_dp_rank=1)
        batch = types.SimpleNamespace(
            dp_balance_stats=_dp_balance_stats(
                local_tokens=4, global_num_tokens=[8, 4], sync_wait_seconds=0.002
            )
        )
        reporter.log_batch_result_stats(batch, result=object())

        get = registry.get_sample_value
        self.assertEqual(
            get("sglang:dp_attention_steps_total", {**labels, "rank_state": "active"}),
            1,
        )
        self.assertEqual(
            get("sglang:dp_attention_token_imbalance_ratio_count", labels), 0
        )


class TestIdleMetrics(CustomTestCase):
    def setUp(self):
        self.scheduler = types.SimpleNamespace(
            running_batch=types.SimpleNamespace(reqs=[]),
            waiting_queue=[],
            grammar_manager=[],
            enable_priority_scheduling=False,
            disaggregation_mode=DisaggregationMode.NULL,
            pool_stats_observer=types.SimpleNamespace(
                get_pool_stats=lambda: types.SimpleNamespace(
                    update_scheduler_stats=lambda _: None,
                    full_capacity=1,
                    is_hybrid_swa=False,
                ),
                streaming_session_count=lambda: 0,
                session_held_tokens=lambda: 0,
            ),
        )
        self.reporter = _make_reporter(self, self.scheduler)
        self.published_occupancies = []
        self.reporter.metrics_collector = types.SimpleNamespace(
            last_log_time=100.0,
            log_stats=lambda stats: self.published_occupancies.append(
                stats.fwd_occupancy
            ),
        )

    def test_host_receive_metrics_survive_queue_drain(self):
        registry = prometheus_client.CollectorRegistry()
        labels = {"model_name": "test", "priority": "", "moe_ep_rank": 0}
        sample_labels = {key: str(value) for key, value in labels.items()}
        with patch.multiple(
            prometheus_client,
            **{
                kind: partial(getattr(prometheus_client, kind), registry=registry)
                for kind in ("Counter", "Gauge", "Histogram", "Summary")
            },
        ):
            collector = SchedulerMetricsCollector(
                labels=labels, server_args=self.scheduler.server_args
            )
        self.reporter.metrics_collector = collector
        self.reporter.current_scheduler_metrics_enabled = True
        self.scheduler.disaggregation_mode = DisaggregationMode.DECODE
        self.scheduler.enable_priority_scheduling = True
        self.scheduler.disagg_decode_prealloc_queue = types.SimpleNamespace(
            queue=[], pre_allocated_token_counts=lambda: (0, 0)
        )
        host_reqs = [
            types.SimpleNamespace(host_staged=True, priority=priority)
            for priority in (1, 2)
        ]
        self.scheduler.disagg_decode_transfer_queue = types.SimpleNamespace(
            queue=[*host_reqs, types.SimpleNamespace(host_staged=False, priority=1)],
            num_pending_deferred_releases=lambda: 0,
        )
        for _ in host_reqs:
            collector.increment_decode_host_receive_reqs()

        with get_context().override_server_args(
            disaggregation_decode_host_receive_threshold=0.8
        ):
            for waiting in (True, False):
                for req in host_reqs:
                    req.host_staged = waiting
                with patch(
                    "sglang.srt.managers.scheduler_components.metrics_reporter.time.perf_counter",
                    return_value=collector.last_log_time + 31,
                ):
                    self.reporter._maybe_log_idle_metrics()
                for priority, expected in (("", 2), ("1", 1), ("2", 1)):
                    self.assertEqual(
                        registry.get_sample_value(
                            "sglang:num_decode_host_receive_queue_reqs",
                            {**sample_labels, "priority": priority},
                        ),
                        expected if waiting else 0,
                    )
                self.assertEqual(
                    registry.get_sample_value(
                        "sglang:num_decode_host_receive_reqs_total", sample_labels
                    ),
                    2,
                )

    def test_idle_clears_cached_forward_occupancy_immediately(self):
        self.reporter.current_scheduler_metrics_enabled = True
        self.reporter.fwd_occupancy = 72.0
        self.reporter.stats.fwd_occupancy = 72.0
        self.reporter._device_timer_window_batch_count = 7

        with (
            patch(
                "sglang.srt.managers.scheduler_components.metrics_reporter.ENABLE_METRICS_DEVICE_TIMER",
                True,
            ),
            patch(
                "sglang.srt.managers.scheduler_components.metrics_reporter.time.perf_counter",
                return_value=101.0,
            ),
        ):
            self.reporter._maybe_log_idle_metrics()
            self.reporter._maybe_log_idle_metrics()

        self.assertEqual(len(self.published_occupancies), 1)
        self.assertTrue(math.isnan(self.published_occupancies[0]))
        self.assertTrue(math.isnan(self.reporter.fwd_occupancy))
        self.assertTrue(math.isnan(self.reporter.stats.fwd_occupancy))
        self.assertEqual(self.reporter._device_timer_window_batch_count, 0)

    def test_idle_resets_forward_timing_when_metrics_are_disabled(self):
        self.reporter.fwd_occupancy = 72.0
        self.reporter.stats.fwd_occupancy = 72.0
        self.reporter._device_timer_window_batch_count = 7

        with patch(
            "sglang.srt.managers.scheduler_components.metrics_reporter.ENABLE_METRICS_DEVICE_TIMER",
            True,
        ):
            self.reporter._maybe_log_idle_metrics()

        self.assertTrue(math.isnan(self.reporter.fwd_occupancy))
        self.assertTrue(math.isnan(self.reporter.stats.fwd_occupancy))
        self.assertEqual(self.reporter._device_timer_window_batch_count, 0)
        self.assertEqual(self.published_occupancies, [])


class TestSchedulerTimeAccounting(CustomTestCase):
    def setUp(self):
        enter_scope(
            self,
            patch(
                "sglang.srt.managers.scheduler_components.metrics_reporter.ENABLE_METRICS_DEVICE_TIMER",
                True,
            ),
        )
        self.reporter = _make_reporter(self, types.SimpleNamespace(device="cpu"))
        self.idle_seconds = []
        self.process_cpu_seconds = []
        self.stage_seconds = []
        self.reporter.enable_metrics = True
        self.reporter.scheduler_stage_metrics.enabled = True
        self.reporter.metrics_collector = types.SimpleNamespace(
            increment_scheduler_idle_seconds=self.idle_seconds.append,
            increment_scheduler_process_cpu_seconds=self.process_cpu_seconds.append,
            increment_scheduler_stage_seconds=lambda **kwargs: (
                self.stage_seconds.append(kwargs)
            ),
        )

    def test_counts_idle_wall_time_and_process_cpu_time(self):
        wall_timestamps = [
            0,
            1_200_000_000,
            1_200_000_000,
            1_500_000_000,
            2_700_000_000,
            2_900_000_000,
            3_000_000_000,
            4_000_000_000,
            4_100_000_000,
        ]
        process_cpu_timestamps = [
            0,
            400_000_000,
            1_200_000_000,
            1_900_000_000,
        ]
        with (
            patch(
                "sglang.srt.managers.scheduler_components.metrics_reporter.time.monotonic_ns",
                side_effect=wall_timestamps,
            ),
            patch(
                "sglang.srt.managers.scheduler_components.metrics_reporter.time.process_time_ns",
                side_effect=process_cpu_timestamps,
            ),
        ):
            self.reporter.start_scheduler_time_accounting()
            self.reporter.record_scheduler_idle()
            self.reporter.record_scheduler_active(event_ns=1_300_000_000)
            self.assertEqual(
                self.reporter._forward_occupancy_log_window.idle_ns, 1_300_000_000
            )
            self.reporter.record_scheduler_active(event_ns=0)
            # Time between idle capture and accounting still counts as idle.
            self.reporter.record_scheduler_idle()
            self.reporter.record_scheduler_idle()

        self.assertAlmostEqual(sum(self.idle_seconds), 2.5)
        self.assertEqual(
            self.reporter._forward_occupancy_log_window.idle_ns, 1_200_000_000
        )
        self.assertAlmostEqual(sum(self.process_cpu_seconds), 1.9)
        self.assertAlmostEqual(
            sum(sample["seconds"] for sample in self.stage_seconds), 4.1
        )
        self.assertEqual({sample["stage"] for sample in self.stage_seconds}, {"other"})

    def test_state_transitions_accumulate_until_periodic_update(self):
        with (
            patch(
                "sglang.srt.managers.scheduler_components.metrics_reporter.ENABLE_METRICS_DEVICE_TIMER",
                False,
            ),
            patch(
                "sglang.srt.managers.scheduler_components.metrics_reporter.time.monotonic_ns",
                side_effect=[
                    0,
                    200_000_000,
                    400_000_000,
                    400_000_000,
                    700_000_000,
                    1_100_000_000,
                    1_200_000_000,
                    1_300_000_000,
                ],
            ),
            patch(
                "sglang.srt.managers.scheduler_components.metrics_reporter.time.process_time_ns",
                side_effect=[0, 300_000_000],
            ) as process_time,
        ):
            self.reporter.start_scheduler_time_accounting()
            accounting = self.reporter._scheduler_time_accounting
            self.reporter.record_scheduler_active(event_ns=200_000_000)
            self.reporter.record_scheduler_idle()
            self.assertEqual(self.idle_seconds, [])
            self.assertEqual(self.process_cpu_seconds, [])
            with self.reporter.scheduler_stage_metrics.record(
                SCHEDULER_STAGE_RECV_REQUESTS
            ):
                pass
            # Receive crossed the reporting deadline. Backdate only idle time,
            # not the stage drain (which has already sampled receive completion).
            self.reporter.record_scheduler_active(event_ns=700_000_000)
            self.assertEqual(self.idle_seconds, [0.5])
            self.assertEqual(accounting.last_sample_ns, 1_200_000_000)
            self.assertAlmostEqual(
                sum(sample["seconds"] for sample in self.stage_seconds), 1.2
            )
            self.assertTrue(
                all(sample["seconds"] >= 0 for sample in self.stage_seconds)
            )
            self.assertAlmostEqual(
                sum(
                    sample["seconds"]
                    for sample in self.stage_seconds
                    if sample["stage"] == SCHEDULER_STAGE_RECV_REQUESTS
                ),
                0.4,
            )
            self.reporter.record_scheduler_active(event_ns=1_300_000_000)

        self.assertIs(self.reporter._scheduler_time_accounting, accounting)
        self.assertEqual(process_time.call_count, 2)
        self.assertEqual(self.idle_seconds, [0.5])
        self.assertAlmostEqual(self.process_cpu_seconds[0], 0.3)

    def test_periodic_update_skips_zero_idle_but_records_cpu_sample(self):
        with (
            patch(
                "sglang.srt.managers.scheduler_components.metrics_reporter.time.monotonic_ns",
                side_effect=[0, 0, 1_000_000_000],
            ),
            patch(
                "sglang.srt.managers.scheduler_components.metrics_reporter.time.process_time_ns",
                side_effect=[0, 0],
            ),
        ):
            self.reporter.start_scheduler_time_accounting()
            self.reporter.record_scheduler_active(event_ns=-1)
            self.assertEqual(self.reporter._forward_occupancy_log_window.idle_ns, 0)
            self.reporter.record_scheduler_active(event_ns=1_000_000_000)

        self.assertEqual(self.idle_seconds, [])
        self.assertEqual(self.process_cpu_seconds, [0.0])


class TestEstimatedPrefillPerf(CustomTestCase):
    """Causal pair count behind ``est. prefill TFLOPS/s`` and ``estimated_flops``."""

    def setUp(self):
        self.scheduler = types.SimpleNamespace()
        self.scheduler.waiting_queue = []
        self.scheduler.disaggregation_mode = DisaggregationMode.NULL
        self.reporter = _make_reporter(self, self.scheduler)
        # One unit per query-key pair and nothing else, so the returned FLOPs
        # are exactly the attention pair count.
        self.reporter._linear_flops_per_token = 0.0
        self.reporter._attn_dot_flops_coeff = 1.0
        self.reporter._weight_read_bytes_per_token = 0.0
        self.reporter._qkv_act_bytes_per_token = 0.0
        self.reporter._prefill_attn_act_read_per_token = 0.0
        self.reporter._kv_cache_bytes_per_token = 0.0
        self.reporter._ffn_act_bytes_per_token = 0.0

    def _pair_count(self, extend_lens, prefix_lens):
        batch = types.SimpleNamespace(extend_lens=extend_lens, prefix_lens=prefix_lens)
        flops, _, _ = self.reporter._estimate_prefill_perf(batch)
        return flops

    def test_chunk_is_charged_for_its_cached_prefix(self):
        self.assertEqual(self._pair_count([4], [3]), 4 * 3 + 4 * 5 / 2)

    def test_prefix_kv_is_read_once_per_chunk(self):
        # One pass over the prefix per chunk, not one read per query-key pair:
        # the chunk's queries share the same KV stream.
        self.reporter._kv_cache_bytes_per_token = 1.0
        batch = types.SimpleNamespace(extend_lens=[4], prefix_lens=[3])
        _, read_bytes, _ = self.reporter._estimate_prefill_perf(batch)
        self.assertEqual(read_bytes, 3)

    def test_requests_in_one_batch_do_not_attend_to_each_other(self):
        self.assertEqual(self._pair_count([100, 100], [0, 0]), 2 * (100 * 101 / 2))

    def test_mixed_prefill_and_decode_rows_use_their_own_context(self):
        # mix_with_running appends running requests as extend_len 1 with their
        # full context as prefix_len.
        self.assertEqual(
            self._pair_count([8, 1, 1], [0, 100, 200]),
            8 * 9 / 2 + (100 + 1) + (200 + 1),
        )


if __name__ == "__main__":
    unittest.main()
