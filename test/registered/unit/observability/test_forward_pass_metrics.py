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
from unittest.mock import Mock, call, patch

import prometheus_client
import torch

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.schedule_batch import ReqKvInfo, ScheduleBatch
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    PrefillStats,
    SchedulerMetricsReporter,
    _CacheHitRateWindow,
)
from sglang.srt.managers.scheduler_components.request_list import ListKind
from sglang.srt.managers.scheduler_pp_mixin import PPBatchMetadata, SchedulerPPMixin
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.observability.forward_pass_metrics import (
    ForwardPassMetrics,
    FpmTiming,
    TimingSegment,
    _FpmPublisherThread,
    decode,
    wrap_forward_with_fpm,
)
from sglang.srt.observability.metrics_collector import (
    DPBalanceStats,
    QueueCount,
    SchedulerMetricsCollector,
)
from sglang.srt.observability.scheduler_stage_metrics import (
    SCHEDULER_STAGE_RECV_REQUESTS,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils.device_timer import DeviceTimer, _TimingInterval, device_timer_ctx
from sglang.test.scheduler_test_utils import FakeReq, decode_queue
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

    def test_pending_group_supports_span_and_segment_consumers(self):
        reporter = Mock()
        timer = DeviceTimer(reporter)
        intervals = [FakeInterval(2, ready=True), FakeInterval(5, start=4)]
        timing = FpmTiming()
        with patch.object(_TimingInterval, "create", side_effect=intervals):
            with timer.capture(timing):
                with device_timer_ctx(timer, "extend"):
                    pass
                with device_timer_ctx(timer, "decode"):
                    pass
        timing.seal()
        publisher = _CollectingPublisher()
        metrics = ForwardPassMetrics(worker_id="worker", counter_id=17)
        timing.publish_when_ready(metrics, publisher)
        segments = []

        def collect_segments(_span):
            segments.extend(
                (segment.elapsed_seconds, segment.metadata["category"])
                for segment in timing.segments
            )

        timing.when_ready(collect_segments)
        self.assertEqual(publisher.metrics, [])
        self.assertEqual(segments, [])

        intervals[-1].ready = True
        timer._report()
        self.assertEqual(len(publisher.metrics), 1)
        self.assertEqual(publisher.metrics[0].wall_time, 0.009)
        self.assertEqual(publisher.metrics[0].counter_id, 17)
        self.assertEqual(segments, [(0.002, "extend"), (0.005, "decode")])
        self.assertEqual(sum(t for t, _ in segments), 0.007)
        self.assertEqual(
            reporter.call_args_list,
            [
                call(t=0.002, category="extend"),
                call(t=0.005, category="decode"),
            ],
        )
        timer._report()
        self.assertEqual(len(publisher.metrics), 1)
        self.assertEqual(len(segments), 2)

    def test_segments_are_read_only_snapshots_built_once_ready(self):
        timer = DeviceTimer()
        interval = FakeInterval(7)
        interval.elapsed_time = Mock(wraps=interval.elapsed_time)
        timing = capture_timing(timer, interval)
        with self.assertRaises(RuntimeError):
            _ = timing.segments

        interval.ready = True
        timer._report()
        segments = timing.segments
        self.assertEqual(segments, timing.segments)
        interval.elapsed_time.assert_called_once_with()
        (segment,) = segments
        self.assertEqual(segment.elapsed_seconds, 0.007)
        self.assertEqual(dict(segment.metadata), {"category": "test"})
        interval.metadata["category"] = "changed"
        self.assertEqual(segment.metadata["category"], "test")
        with self.assertRaises(TypeError):
            segment.metadata["category"] = "changed"
        with self.assertRaises(AttributeError):
            segment.elapsed_seconds = 0.0
        self.assertFalse(hasattr(segment, "__dict__"))
        self.assertIsInstance(segment, TimingSegment)

    def test_span_only_consumers_skip_segment_durations(self):
        timer = DeviceTimer()
        intervals = [FakeInterval(2, ready=True), FakeInterval(5, True, start=4)]
        for interval in intervals:
            interval.elapsed_time = Mock(wraps=interval.elapsed_time)
        timing = capture_timing(timer, *intervals)
        timer._report()
        ready = Mock()
        timing.when_ready(ready)
        ready.assert_called_once_with(0.009)
        for interval in intervals:
            interval.elapsed_time.assert_not_called()

    def test_completion_before_seal_keeps_all_listeners(self):
        timer = DeviceTimer()
        timing = FpmTiming()
        callbacks = [Mock(), Mock(), Mock()]
        with patch.object(
            _TimingInterval, "create", return_value=FakeInterval(7, ready=True)
        ):
            with timer.capture(timing):
                timing.when_ready(callbacks[0])
                timing.when_ready(callbacks[1])
                with device_timer_ctx(timer, "decode"):
                    pass
        callbacks[0].assert_not_called()
        callbacks[1].assert_not_called()
        timing.seal()
        timing.when_ready(callbacks[2])
        timing.seal()
        timer._report()
        for callback in callbacks:
            callback.assert_called_once_with(0.007)

    def test_captured_intervals_still_reach_late_aggregate_reporters(self):
        original, late = Mock(), Mock()
        timer = DeviceTimer(original)
        interval = FakeInterval(7)
        timing = capture_timing(timer, interval)
        interval.ready = True
        timer.add_reporter(late)
        timer._report()
        for reporter in (original, late):
            reporter.assert_called_once_with(t=0.007, category="test")
        ready = Mock()
        timing.when_ready(ready)
        ready.assert_called_once_with(0.007)


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
                        grammar=None,
                        spec_verify_ct=0,
                        spec_num_correct_drafts=0,
                        update_spec_correct_drafts_histogram=Mock(),
                        kv=ReqKvInfo(),
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
                    processor, result, batch, [True, False, False]
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
        ),
    )
    return server_args


def _make_reporter(test, scheduler) -> SchedulerMetricsReporter:
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
            emit_kv_metrics=Mock(),
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
        enable_metrics=False,
        is_stats_logging_rank=True,
        current_scheduler_metrics_enabled=False,
        enable_kv_cache_events=False,
        collector=None,
    )
    return SchedulerMetricsReporter(
        scheduler=scheduler,
        metrics_collector_context=context,
        metrics_collector=None,
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
                # No unified pool: nothing keeps freed pages on the forward.
                receiver._on_step_forward_recorded = lambda forward_done: False
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
        scheduler._pp_result_done_event = types.MethodType(
            SchedulerPPMixin._pp_result_done_event, scheduler
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
            request_lists=lambda: {kind: [] for kind in ListKind},
            enable_priority_scheduling=False,
            disaggregation_mode=DisaggregationMode.NULL,
            pool_stats_observer=types.SimpleNamespace(
                get_pool_stats=lambda: types.SimpleNamespace(
                    update_scheduler_stats=lambda _: None,
                    full_capacity=1,
                    windows=(),
                    target_window_stats=lambda: None,
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

    def test_idle_publishes_refreshed_kv_metrics(self):
        self.reporter.current_scheduler_metrics_enabled = True
        self.scheduler.disagg_decode_prealloc_queue = types.SimpleNamespace(
            queue=[], pre_allocated_token_counts=lambda windows=(): (0, {})
        )
        self.scheduler.disagg_decode_transfer_queue = types.SimpleNamespace(
            queue=[], num_pending_deferred_releases=lambda: 0
        )
        published = []
        self.scheduler.kv_events_publisher.emit_kv_metrics.side_effect = lambda: (
            published.append(
                (
                    self.reporter.stats.token_usage,
                    self.reporter.stats.num_running_reqs.total,
                    self.reporter.stats.num_queue_reqs.total,
                )
            )
        )
        for mode in DisaggregationMode:
            for token_usage in (0.0, 0.25):
                with self.subTest(mode=mode, token_usage=token_usage):
                    self.scheduler.disaggregation_mode = mode
                    self.reporter.stats.token_usage = 0.82
                    self.reporter.stats.num_running_reqs = QueueCount(total=1)
                    self.reporter.stats.num_queue_reqs = QueueCount(total=1)
                    self.scheduler.pool_stats_observer.get_pool_stats = lambda: (
                        types.SimpleNamespace(
                            update_scheduler_stats=lambda stats: setattr(
                                stats, "token_usage", token_usage
                            ),
                            full_capacity=1,
                            windows=(),
                            target_window_stats=lambda: None,
                        )
                    )
                    with patch(
                        "sglang.srt.managers.scheduler_components.metrics_reporter.time.perf_counter",
                        return_value=101.0,
                    ):
                        self.reporter._maybe_log_idle_metrics()
                    self.assertEqual(published, [(token_usage, 0, 0)])
                    published.clear()

    def test_idle_kv_metrics_preserve_rate_limit_and_metrics_gate(self):
        self.reporter.current_scheduler_metrics_enabled = True
        with (
            patch(
                "sglang.srt.managers.scheduler_components.metrics_reporter.ENABLE_METRICS_DEVICE_TIMER",
                False,
            ),
            patch(
                "sglang.srt.managers.scheduler_components.metrics_reporter.time.perf_counter",
                return_value=101.0,
            ),
        ):
            self.reporter._maybe_log_idle_metrics()
        self.scheduler.kv_events_publisher.emit_kv_metrics.assert_not_called()

        with patch(
            "sglang.srt.managers.scheduler_components.metrics_reporter.time.perf_counter",
            return_value=131.0,
        ):
            self.reporter._maybe_log_idle_metrics()
        self.scheduler.kv_events_publisher.emit_kv_metrics.assert_called_once_with()

        self.scheduler.kv_events_publisher.emit_kv_metrics.reset_mock()
        self.reporter.current_scheduler_metrics_enabled = False
        self.reporter.stats.num_running_reqs = QueueCount(total=1)
        self.reporter._maybe_log_idle_metrics()
        self.scheduler.kv_events_publisher.emit_kv_metrics.assert_not_called()

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
            queue=[], pre_allocated_token_counts=lambda windows: (0, {})
        )
        host_reqs = [
            types.SimpleNamespace(req=FakeReq(priority=priority), host_staged=True)
            for priority in (1, 2)
        ]
        device_req = types.SimpleNamespace(req=FakeReq(priority=1), host_staged=False)
        self.scheduler.disagg_decode_transfer_queue = types.SimpleNamespace(
            queue=decode_queue(ListKind.PD_RECEIVING, [*host_reqs, device_req]),
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

    def test_cpu_clock_reset_after_restore_rebases_accounting(self):
        """CPU clock resets must count CPU time since restore, never decrement."""
        for is_idle in (False, True):
            with self.subTest(is_idle=is_idle):
                registry = prometheus_client.CollectorRegistry()
                cpu_seconds = prometheus_client.Counter(
                    "test_scheduler_process_cpu_seconds",
                    "Scheduler CPU time",
                    registry=registry,
                )
                collector = self.reporter.metrics_collector
                collector.increment_scheduler_process_cpu_seconds = cpu_seconds.inc
                with (
                    patch(
                        "sglang.srt.managers.scheduler_components.metrics_reporter.time.monotonic_ns",
                        side_effect=[
                            0,
                            1_000_000_000,
                            2_000_000_000,
                            3_000_000_000,
                            4_000_000_000,
                        ],
                    ),
                    patch(
                        "sglang.srt.managers.scheduler_components.metrics_reporter.time.process_time_ns",
                        # Each restore starts a new process CPU clock near zero;
                        # its reading is the CPU time since restore.
                        side_effect=[
                            10_000_000_000,
                            100_000_000,
                            400_000_000,
                            10_000_000,
                            210_000_000,
                        ],
                    ),
                ):
                    self.reporter.start_scheduler_time_accounting()
                    for second, expected in enumerate((0.1, 0.4, 0.41, 0.61), start=1):
                        self.reporter._record_scheduler_time(
                            is_idle=is_idle, event_ns=second * 1_000_000_000
                        )
                        self.assertAlmostEqual(
                            registry.get_sample_value(
                                "test_scheduler_process_cpu_seconds_total"
                            ),
                            expected,
                        )

    def test_monotonic_clock_reset_after_restore_rebases_accounting(self):
        """A backwards monotonic clock must rebase accounting, not stall or go negative."""
        with (
            patch(
                "sglang.srt.managers.scheduler_components.metrics_reporter.time.monotonic_ns",
                # Restored on a host with less uptime than the checkpointed one.
                side_effect=[
                    10_000_000_000,
                    11_000_000_000,
                    1_000_000_000,
                    2_000_000_000,
                ],
            ),
            patch(
                "sglang.srt.managers.scheduler_components.metrics_reporter.time.process_time_ns",
                side_effect=[1_000_000_000, 1_500_000_000, 50_000_000, 250_000_000],
            ),
        ):
            self.reporter.start_scheduler_time_accounting()
            self.reporter._record_scheduler_time(is_idle=True, event_ns=11_000_000_000)
            # An occupancy window open across the reset would keep measuring
            # from a pre-restore perf_counter() start.
            self.reporter._device_timer_window_batch_count = 7
            for event_ns in (1_000_000_000, 2_000_000_000):
                self.reporter._record_scheduler_time(is_idle=True, event_ns=event_ns)

        self.assertEqual(self.reporter._device_timer_window_batch_count, 0)
        self.assertEqual(self.idle_seconds, [1.0, 1.0])
        self.assertEqual(self.process_cpu_seconds, [0.5, 0.05, 0.2])
        self.assertTrue(all(kwargs["seconds"] >= 0 for kwargs in self.stage_seconds))
        self.assertAlmostEqual(
            sum(kwargs["seconds"] for kwargs in self.stage_seconds), 2.0
        )

    def test_monotonic_reset_preserves_buffered_idle_and_stage_totals(self):
        """Restore must publish measured spans even before the reporting deadline."""
        with (
            patch(
                "sglang.srt.managers.scheduler_components.metrics_reporter.time.monotonic_ns",
                side_effect=[
                    10_000_000_000,
                    10_200_000_000,
                    10_300_000_000,
                    10_500_000_000,
                    10_600_000_000,
                    1_000_000_000,
                    1_100_000_000,
                    2_000_000_000,
                ],
            ),
            patch(
                "sglang.srt.managers.scheduler_components.metrics_reporter.time.process_time_ns",
                side_effect=[1_000_000_000, 50_000_000, 250_000_000],
            ),
        ):
            self.reporter.start_scheduler_time_accounting()
            self.reporter.record_scheduler_active(event_ns=10_200_000_000)
            with self.reporter.scheduler_stage_metrics.record(
                SCHEDULER_STAGE_RECV_REQUESTS
            ):
                pass
            with self.reporter.scheduler_stage_metrics.record(
                SCHEDULER_STAGE_RECV_REQUESTS
            ):
                # Restore during a stage must retain its category until it exits.
                self.reporter._record_scheduler_time(
                    is_idle=False, event_ns=1_000_000_000
                )
            self.reporter._record_scheduler_time(is_idle=True, event_ns=2_000_000_000)

        self.assertEqual(self.idle_seconds, [0.2])
        self.assertEqual(self.process_cpu_seconds, [0.05, 0.2])
        self.assertTrue(all(sample["seconds"] >= 0 for sample in self.stage_seconds))
        self.assertAlmostEqual(
            sum(sample["seconds"] for sample in self.stage_seconds), 1.6
        )
        self.assertAlmostEqual(
            sum(
                sample["seconds"]
                for sample in self.stage_seconds
                if sample["stage"] == SCHEDULER_STAGE_RECV_REQUESTS
            ),
            0.3,
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


def _fake_model_config(
    *,
    hidden_size=4096,
    num_attention_layers=8,
    head_dim=128,
    num_attn_heads=32,
    num_kv_heads=8,
    attention_arch=None,
    dtype_itemsize=2,
    hf_text_config_overrides=None,
    **model_config_overrides,
):
    """Minimal model_config stand-in for _init_estimated_perf_constants()."""
    hf_text_config_kwargs = {"intermediate_size": 14336}
    hf_text_config_kwargs.update(hf_text_config_overrides or {})
    hf_text_config = types.SimpleNamespace(**hf_text_config_kwargs)
    config = types.SimpleNamespace(
        hf_text_config=hf_text_config,
        hidden_size=hidden_size,
        num_attention_layers=num_attention_layers,
        head_dim=head_dim,
        attention_arch=attention_arch,
        dtype=types.SimpleNamespace(itemsize=dtype_itemsize),
        get_num_attention_heads=lambda tp_size: num_attn_heads,
        get_num_kv_heads=lambda tp_size: num_kv_heads,
    )
    for key, value in model_config_overrides.items():
        setattr(config, key, value)
    return config


class TestEstimatedPerfConstants(CustomTestCase):
    """Covers _init_estimated_perf_constants()'s MLA/MoE-aware FLOPs and bytes."""

    def setUp(self):
        self.scheduler = types.SimpleNamespace()
        self.scheduler.waiting_queue = []
        self.scheduler.disaggregation_mode = DisaggregationMode.NULL
        self.reporter = _make_reporter(self, self.scheduler)

    def _init(self, model_config):
        self.scheduler.model_config = model_config
        self.reporter._init_estimated_perf_constants()

    def test_dense_gqa_matches_generic_formula(self):
        # Regression guard for the non-MLA/non-MoE path, which must still
        # reduce to the original dense/GQA formula after the MLA/MoE refactor.
        hidden, head_dim, heads, kv_heads, layers = 4096, 128, 32, 8, 8
        self._init(
            _fake_model_config(
                hidden_size=hidden,
                head_dim=head_dim,
                num_attn_heads=heads,
                num_kv_heads=kv_heads,
                num_attention_layers=layers,
            )
        )
        attn_linear_flops = (
            2.0 * hidden * head_dim * (heads + 2.0 * kv_heads)
            + 2.0 * hidden * head_dim * heads
        )
        mlp_flops = 6.0 * hidden * 14336
        self.assertEqual(
            self.reporter._linear_flops_per_token,
            attn_linear_flops * layers + mlp_flops * layers,
        )
        self.assertEqual(
            self.reporter._attn_dot_flops_coeff, 4.0 * heads * head_dim * layers
        )
        self.assertEqual(
            self.reporter._kv_cache_bytes_per_token,
            2.0 * kv_heads * head_dim * layers * 2.0,
        )

    def test_mla_uses_low_rank_shapes_not_generic_gqa(self):
        hidden, heads, layers = 2048, 16, 4
        kv_lora_rank, qk_rope, qk_nope, v_head_dim = 64.0, 32.0, 96.0, 96.0
        self._init(
            _fake_model_config(
                hidden_size=hidden,
                num_attn_heads=heads,
                num_attention_layers=layers,
                attention_arch=AttentionArch.MLA,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope,
                qk_nope_head_dim=qk_nope,
                v_head_dim=v_head_dim,
                hf_text_config_overrides={"q_lora_rank": 0, "intermediate_size": 10944},
            )
        )
        qk_head_dim = qk_nope + qk_rope
        q_proj_flops = 2.0 * hidden * heads * qk_head_dim
        kv_a_dim = kv_lora_rank + qk_rope
        kv_a_proj_flops = 2.0 * hidden * kv_a_dim
        kv_b_out_dim = heads * (qk_nope + v_head_dim)
        kv_b_proj_flops = 2.0 * kv_lora_rank * kv_b_out_dim
        o_proj_flops = 2.0 * heads * v_head_dim * hidden
        attn_linear_flops = (
            q_proj_flops + kv_a_proj_flops + kv_b_proj_flops + o_proj_flops
        )
        attn_dot_flops_per_layer = 2.0 * heads * (qk_head_dim + v_head_dim)
        mlp_flops = 6.0 * hidden * 10944

        self.assertEqual(
            self.reporter._linear_flops_per_token,
            attn_linear_flops * layers + mlp_flops * layers,
        )
        self.assertEqual(
            self.reporter._attn_dot_flops_coeff, attn_dot_flops_per_layer * layers
        )
        # The defining MLA property: KV cache holds one shared latent per
        # token per layer (kv_a_dim), not num_kv_heads * head_dim.
        self.assertEqual(
            self.reporter._kv_cache_bytes_per_token, kv_a_dim * layers * 2.0
        )

    def test_mla_with_q_lora_rank_compresses_query_projection(self):
        hidden, heads, layers = 2048, 16, 4
        q_lora_rank = 384.0
        self._init(
            _fake_model_config(
                hidden_size=hidden,
                num_attn_heads=heads,
                num_attention_layers=layers,
                attention_arch=AttentionArch.MLA,
                kv_lora_rank=64.0,
                qk_rope_head_dim=32.0,
                qk_nope_head_dim=96.0,
                v_head_dim=96.0,
                hf_text_config_overrides={
                    "q_lora_rank": q_lora_rank,
                    "intermediate_size": 10944,
                },
            )
        )
        qk_head_dim = 96.0 + 32.0
        compressed_q_proj_flops = 2.0 * hidden * q_lora_rank + 2.0 * q_lora_rank * (
            heads * qk_head_dim
        )
        uncompressed_q_proj_flops = 2.0 * hidden * heads * qk_head_dim
        # Low-rank Q compression must actually reduce projection FLOPs for a
        # realistic rank choice, not just change which branch runs.
        self.assertLess(compressed_q_proj_flops, uncompressed_q_proj_flops)

    def test_moe_with_explicit_moe_intermediate_size_scales_by_active_experts(self):
        hidden, layers = 4096, 4
        self._init(
            _fake_model_config(
                hidden_size=hidden,
                num_attention_layers=layers,
                hf_text_config_overrides={
                    "intermediate_size": 18432,
                    "moe_intermediate_size": 1408,
                    "num_experts_per_tok": 6,
                    "n_shared_experts": 2,
                },
            )
        )
        head_dim, heads, kv_heads = 128, 32, 8
        attn_linear_flops = (
            2.0 * hidden * head_dim * (heads + 2.0 * kv_heads)
            + 2.0 * hidden * head_dim * heads
        )
        active_expert_units = 6 + 2
        mlp_flops = 6.0 * hidden * 1408 * active_expert_units
        single_expert_mlp_flops = 6.0 * hidden * 1408
        # The MLP cost must scale by all active expert units (routed + shared),
        # not just treat moe_intermediate_size as a single expert's cost.
        self.assertGreater(mlp_flops, single_expert_mlp_flops)
        self.assertEqual(
            self.reporter._linear_flops_per_token,
            attn_linear_flops * layers + mlp_flops * layers,
        )

    def test_moe_without_moe_intermediate_size_falls_back_to_per_expert_size(self):
        # OLMoE/Mixtral-style configs: no dense layers at all, so there is no
        # separate moe_intermediate_size -- intermediate_size IS the per-expert
        # FFN size. Detected via num_experts (also covers num_local_experts /
        # n_routed_experts).
        hidden, layers = 2048, 16
        self._init(
            _fake_model_config(
                hidden_size=hidden,
                num_attention_layers=layers,
                hf_text_config_overrides={
                    "intermediate_size": 1024,
                    "num_experts_per_tok": 8,
                    "num_experts": 64,
                },
            )
        )
        head_dim, heads, kv_heads = 128, 32, 8
        attn_linear_flops = (
            2.0 * hidden * head_dim * (heads + 2.0 * kv_heads)
            + 2.0 * hidden * head_dim * heads
        )
        active_expert_units = 8
        mlp_flops = 6.0 * hidden * 1024 * active_expert_units
        single_expert_mlp_flops = 6.0 * hidden * 1024
        # Must scale by all 8 active experts, not silently charge for just one
        # (the bug: falling through to the dense branch with no multiplier).
        wrong_total = attn_linear_flops * layers + single_expert_mlp_flops * layers
        self.assertNotEqual(self.reporter._linear_flops_per_token, wrong_total)
        self.assertEqual(
            self.reporter._linear_flops_per_token,
            attn_linear_flops * layers + mlp_flops * layers,
        )

    def test_num_experts_per_tok_alone_does_not_trigger_moe_without_expert_count(self):
        # num_experts_per_tok present but no num_experts/num_local_experts/
        # n_routed_experts field at all: must NOT guess intermediate_size is
        # per-expert -- falls through to the plain dense formula untouched.
        hidden, layers = 2048, 4
        self._init(
            _fake_model_config(
                hidden_size=hidden,
                num_attention_layers=layers,
                hf_text_config_overrides={
                    "intermediate_size": 4096,
                    "num_experts_per_tok": 8,
                },
            )
        )
        head_dim, heads, kv_heads = 128, 32, 8
        attn_linear_flops = (
            2.0 * hidden * head_dim * (heads + 2.0 * kv_heads)
            + 2.0 * hidden * head_dim * heads
        )
        dense_mlp_flops = 6.0 * hidden * 4096
        self.assertEqual(
            self.reporter._linear_flops_per_token,
            attn_linear_flops * layers + dense_mlp_flops * layers,
        )

    def test_hybrid_layers_block_type_scales_kv_cache_by_attention_layers_only(self):
        # Mamba/SSM layers carry a fixed-size recurrent state, not a growing
        # per-token KV cache -- kv_cache_bytes_per_token must scale by the
        # attention-layer count parsed from layers_block_type, never the
        # unrelated num_attention_layers field (999, deliberately wrong here).
        #
        # layers_block_type is a real Python list on every HF config that
        # defines it (NemotronH, Jamba, Zamba, Bamba, FalconH1, ...), never a
        # comma-joined string -- use a list here so this test exercises the
        # same type the production code actually sees.
        head_dim, kv_heads = 128.0, 8.0
        self._init(
            _fake_model_config(
                head_dim=head_dim,
                num_kv_heads=kv_heads,
                num_attention_layers=999,
                hf_text_config_overrides={
                    "layers_block_type": [
                        "attention",
                        "mamba",
                        "mlp",
                        "attention",
                        "mamba",
                        "mlp",
                        "mamba",
                    ],
                    "intermediate_size": 4096,
                },
            )
        )
        num_attn_layers = 2.0  # "attention" appears twice in layers_block_type
        expected_kv_cache_bytes = 2.0 * kv_heads * head_dim * num_attn_layers * 2.0
        self.assertEqual(
            self.reporter._kv_cache_bytes_per_token, expected_kv_cache_bytes
        )

    def test_falcon_h1_style_uniform_hybrid_label_counts_every_layer(self):
        # FalconH1Config.layers_block_type is a property that returns
        # ["hybrid"] * num_hidden_layers -- every layer, never "attention",
        # "mlp", or "moe". Each FalconH1DecoderLayer actually contains its own
        # self.self_attn, self.mamba, AND self.feed_forward all at once, so
        # every layer genuinely pays both the attention and the MLP cost.
        # The regression this guards: naively counting literal "attention"/
        # "mlp"/"moe" labels gives num_attn_layers=0 AND num_mlp_layers=0 for
        # this convention, zeroing every FLOPs/bandwidth estimate.
        hidden, head_dim, heads, kv_heads, layers = 4096, 128, 32, 8, 6
        self._init(
            _fake_model_config(
                hidden_size=hidden,
                head_dim=head_dim,
                num_attn_heads=heads,
                num_kv_heads=kv_heads,
                num_attention_layers=layers,
                hf_text_config_overrides={
                    "layers_block_type": ["hybrid"] * layers,
                    "intermediate_size": 14336,
                },
            )
        )
        attn_linear_flops = (
            2.0 * hidden * head_dim * (heads + 2.0 * kv_heads)
            + 2.0 * hidden * head_dim * heads
        )
        mlp_flops = 6.0 * hidden * 14336
        # Must match the plain dense/GQA formula over every layer -- not zero.
        self.assertEqual(
            self.reporter._linear_flops_per_token,
            attn_linear_flops * layers + mlp_flops * layers,
        )
        self.assertGreater(self.reporter._linear_flops_per_token, 0.0)
        self.assertEqual(
            self.reporter._attn_dot_flops_coeff, 4.0 * heads * head_dim * layers
        )
        self.assertGreater(self.reporter._kv_cache_bytes_per_token, 0.0)

    def test_jamba_style_attention_mamba_only_labels_still_charge_mlp_everywhere(self):
        # Jamba/Bamba's layers_block_type only ever contains "attention" or
        # "mamba" -- there is no separate "mlp"/"moe" label, because every
        # layer (whichever mixer it uses) has its own attached MLP. Only the
        # attention *mixer* count should shrink to the real "attention"
        # entries; the MLP count must still cover every layer.
        hidden, head_dim, heads, kv_heads = 4096, 128, 32, 8
        layers_block_type = [
            "attention",
            "mamba",
            "mamba",
            "attention",
            "mamba",
            "mamba",
            "mamba",
            "attention",
        ]
        total_layers = float(len(layers_block_type))
        num_attn_layers = 3.0  # "attention" appears 3 times
        self._init(
            _fake_model_config(
                hidden_size=hidden,
                head_dim=head_dim,
                num_attn_heads=heads,
                num_kv_heads=kv_heads,
                num_attention_layers=int(total_layers),
                hf_text_config_overrides={
                    "layers_block_type": layers_block_type,
                    "intermediate_size": 14336,
                },
            )
        )
        attn_linear_flops = (
            2.0 * hidden * head_dim * (heads + 2.0 * kv_heads)
            + 2.0 * hidden * head_dim * heads
        )
        mlp_flops = 6.0 * hidden * 14336
        # The regression this guards against: num_mlp_layers silently forced
        # to 0 because the vocabulary has no literal "mlp"/"moe" label.
        wrong_zeroed_mlp = attn_linear_flops * num_attn_layers + mlp_flops * 0.0
        expected = attn_linear_flops * num_attn_layers + mlp_flops * total_layers
        self.assertNotEqual(self.reporter._linear_flops_per_token, wrong_zeroed_mlp)
        self.assertEqual(self.reporter._linear_flops_per_token, expected)
        # KV cache bandwidth only applies to the real attention-mixer layers.
        self.assertEqual(
            self.reporter._kv_cache_bytes_per_token,
            2.0 * kv_heads * head_dim * num_attn_layers * 2.0,
        )

    def test_full_attention_layer_ids_read_from_model_config_when_present(self):
        # sglang computes swa/full_attention_layer_ids in ModelConfig.__init__
        # (via get_hybrid_layer_ids()) and stores them on model_config itself
        # -- this is the real path for Gemma2/Gemma3/Llama4-style SWA hybrid
        # models. No layers_block_type is involved for these.
        head_dim, kv_heads = 128.0, 8.0
        full_attention_layer_ids = [2, 5, 9, 12]
        self._init(
            _fake_model_config(
                head_dim=head_dim,
                num_kv_heads=kv_heads,
                num_attention_layers=16,
                full_attention_layer_ids=full_attention_layer_ids,
            )
        )
        num_attn_layers = float(len(full_attention_layer_ids))
        self.assertEqual(
            self.reporter._kv_cache_bytes_per_token,
            2.0 * kv_heads * head_dim * num_attn_layers * 2.0,
        )

    def test_full_attention_layer_ids_falls_back_to_hf_text_config(self):
        # KimiLinear/KimiK3's own HF config class defines full_attention_layer_ids
        # as a native field (read directly as self.config.full_attention_layer_ids
        # in kimi_linear.py/kimi_k3.py) -- sglang's ModelConfig never computes or
        # mirrors it for these architectures, so it only ever lives on
        # hf_text_config. The regression this guards against: looking ONLY at
        # hf_text_config (pre-fix) silently breaks SWA models that have it on
        # model_config instead, or looking ONLY at model_config (an earlier
        # draft of this fix) would silently break KimiLinear/KimiK3.
        head_dim, kv_heads = 128.0, 8.0
        full_attention_layer_ids = [0, 3, 6]
        self._init(
            _fake_model_config(
                head_dim=head_dim,
                num_kv_heads=kv_heads,
                num_attention_layers=16,
                hf_text_config_overrides={
                    "full_attention_layer_ids": full_attention_layer_ids,
                },
            )
        )
        num_attn_layers = float(len(full_attention_layer_ids))
        self.assertEqual(
            self.reporter._kv_cache_bytes_per_token,
            2.0 * kv_heads * head_dim * num_attn_layers * 2.0,
        )


if __name__ == "__main__":
    unittest.main()
