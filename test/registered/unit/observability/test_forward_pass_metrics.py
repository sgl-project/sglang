from sglang.srt.runtime_context import (
    get_context,
    get_disagg,
    get_observability,
    get_parallel,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.disagg_owner_shells import decode_owner_shell, prefill_owner_shell
from sglang.test.scheduler_request_lists import make_request_registry

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

import math
import queue
import types
import unittest
from functools import partial
from itertools import count
from types import SimpleNamespace
from unittest.mock import Mock, patch

import prometheus_client
import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.request_lists import CoreLists
from sglang.srt.managers.schedule_batch import ReqKvInfo, ScheduleBatch
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    MetricsSink,
    PrefillStats,
    SchedulerMetricsReporter,
    _CacheHitRateWindow,
)
from sglang.srt.managers.scheduler_components.mode_owners import ModeOwners
from sglang.srt.managers.scheduler_components.pp_ring import PPRing
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
                reporter.metrics_sink.enable_fpm = True
                reqs = [
                    types.SimpleNamespace(
                        seqlen=length + 1,
                        is_retracted=length == 200,
                        finished=lambda length=length: length == 300,
                        grammar=None,
                        spec_stats=Mock(verify_ct=0, num_correct_drafts=0),
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
                    services=types.SimpleNamespace(metrics_sink=reporter.metrics_sink),
                    workers=types.SimpleNamespace(
                        model_worker=types.SimpleNamespace(
                            on_verify_complete_cpu=Mock()
                        )
                    ),
                )
                with patch(
                    "sglang.srt.managers.scheduler_components.batch_result_processor.advance_grammar_fsm",
                    side_effect=settle_grammar,
                ):
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
        sink = _make_reporter(self, types.SimpleNamespace()).metrics_sink
        result = GenerationBatchResult()
        sink.snapshot_spec_decode_metrics(None, result)
        self.assertIsNone(result.fpm_scheduled_requests)
        sink.enable_fpm = True
        for algorithm in (SpeculativeAlgorithm.UNO, SpeculativeAlgorithm.FROZEN_KV_MTP):
            sink.snapshot_spec_decode_metrics(
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
    metrics_sink = MetricsSink(
        metrics_collector_context=context,
        metrics_collector=None,
        device=getattr(scheduler, "device", ""),
    )
    if hasattr(scheduler, "disaggregation_mode"):
        _set_disaggregation_mode(test, scheduler.disaggregation_mode)
    mode_owners = ModeOwners()
    if not hasattr(scheduler, "request_registry"):
        scheduler.request_registry = make_request_registry(
            core_lists=getattr(scheduler, "lists", None) or CoreLists(),
            mode_owners=mode_owners,
        )
    return SchedulerMetricsReporter(
        metrics_sink=metrics_sink,
        request_registry=scheduler.request_registry,
        counters=None,
        pool_stats_observer=getattr(scheduler, "pool_stats_observer", None),
        grammar_manager=getattr(scheduler, "grammar_manager", None),
        mm_receiver=None,
        kv_events_publisher=scheduler.kv_events_publisher,
        mode_owners=mode_owners,
        kv=None,
        capacity=None,
        workers=types.SimpleNamespace(
            tp_worker=scheduler.tp_worker, draft_worker=scheduler.draft_worker
        ),
        model_config=None,
        device=getattr(scheduler, "device", ""),
    )


def _set_disaggregation_mode(test, mode: DisaggregationMode) -> None:
    """The reporter reads the live mode from config."""
    enter_scope(test, get_disagg().override(disaggregation_mode=mode.value))


class TestForwardPassMetrics(unittest.TestCase):
    def setUp(self):
        self.scheduler = types.SimpleNamespace(lists=CoreLists())
        self.scheduler.disaggregation_mode = DisaggregationMode.NULL
        self.reporter = _make_reporter(self, self.scheduler)
        self.publisher = _CollectingPublisher()
        self.reporter.metrics_sink._fpm_worker_id = "worker-7"
        self.reporter.metrics_sink._fpm_dp_rank = 0
        self.reporter.metrics_sink._fpm_publisher = self.publisher
        self.reporter.metrics_sink.enable_fpm = True

    def test_cache_hit_rate_window_keeps_last_15s_of_tokens(self):
        window = _CacheHitRateWindow()
        self.assertEqual(window.add(hit_tokens=20, total_tokens=100, now=0.0), 0.2)
        self.assertEqual(window.add(hit_tokens=80, total_tokens=100, now=10.0), 0.5)
        self.assertEqual(window.add(hit_tokens=90, total_tokens=100, now=15.0), 0.85)

    def _make_batch(self, **overrides):
        defaults = dict(
            forward_mode=_FakeForwardMode(),
            reqs=[],
            num_decode_rows=0,
            prefix_lens=[],
            prefill_stats=None,
            seq_lens_cpu=[],
        )
        defaults.update(overrides)
        batch = types.SimpleNamespace(**defaults)
        batch.prefill_reqs = partial(ScheduleBatch.prefill_reqs, batch)
        return batch

    def _emit_ready(self, batch, milliseconds=1):
        timing = capture_timing(DeviceTimer(), FakeInterval(milliseconds, ready=True))
        self.reporter._emit_forward_pass_metrics(
            batch, GenerationBatchResult(fpm_timing=timing)
        )

    def test_emit_mixed_batch_separates_prefill_and_decode(self):
        self.reporter.metrics_sink._fpm_dp_rank = 3
        self.scheduler.lists.waiting_queue = [_FakeReq(6), _FakeReq(4, output_len=2)]

        prefill_a = _FakeReq(10)
        prefill_b = _FakeReq(14)
        decode_req = _FakeReq(8, output_len=3)
        batch = self._make_batch(
            forward_mode=_FakeForwardMode(is_mixed=True, is_extend=True),
            reqs=[prefill_a, prefill_b, decode_req],
            num_decode_rows=1,
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

        self.assertEqual(len(self.publisher.metrics), 1)
        metrics = self.publisher.metrics[0]
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
        self.scheduler.lists.waiting_queue = [_FakeReq(128), _FakeReq(256)]
        batch = self._make_batch(forward_mode=ForwardMode.IDLE)
        self._emit_ready(batch, milliseconds=23)
        metric = self.publisher.metrics[-1]
        self.assertEqual(metric.scheduled_requests.num_prefill_requests, 0)
        self.assertEqual(metric.scheduled_requests.num_decode_requests, 0)
        self.assertAlmostEqual(metric.wall_time, 0.023)
        self.assertEqual(metric.queued_requests.num_prefill_requests, 2)
        self.assertEqual(metric.queued_requests.sum_prefill_tokens, 384)

    def test_scheduler_state_controls_heartbeat_with_metrics_disabled(self):
        sink = self.reporter.metrics_sink
        sink.record_scheduler_active(event_ns=0)
        self.assertFalse(self.publisher.idle)
        sink.record_scheduler_idle()
        self.assertTrue(self.publisher.idle)
        sink.record_scheduler_active(event_ns=1)
        self.assertFalse(self.publisher.idle)

    def test_emit_skips_uninstrumented_iteration(self):
        for timing in (None, capture_timing(DeviceTimer())):
            with self.subTest(timing=timing):
                self.reporter._emit_forward_pass_metrics(
                    self._make_batch(), GenerationBatchResult(fpm_timing=timing)
                )
                self.assertEqual(self.publisher.metrics, [])

    def test_delayed_timing_uses_frozen_batch_and_queue_stats(self):
        timer = self.reporter.metrics_sink.forward_pass_device_timer = DeviceTimer()
        interval = FakeInterval(7)
        timing = capture_timing(timer, interval)
        batch = self._make_batch(seq_lens_cpu=[100, 200])
        self.scheduler.lists.waiting_queue = [_FakeReq(300)]
        result = types.SimpleNamespace(fpm_timing=timing)
        self.reporter._emit_forward_pass_metrics(batch, result)
        self.assertEqual(self.publisher.metrics, [])

        # Later iterations may mutate both sources before timing becomes ready.
        batch.seq_lens_cpu = [900]
        self.scheduler.lists.waiting_queue.clear()
        interval.ready = True
        with patch(
            "sglang.srt.managers.scheduler_components.metrics_reporter.ENABLE_METRICS_DEVICE_TIMER",
            False,
        ):
            self.reporter.metrics_sink.update_device_timer()
        metrics = self.publisher.metrics[0]
        self.assertAlmostEqual(metrics.wall_time, 0.007)
        self.assertEqual(metrics.scheduled_requests.num_decode_requests, 2)
        self.assertEqual(metrics.scheduled_requests.sum_decode_kv_tokens, 300)
        self.assertEqual(metrics.queued_requests.sum_prefill_tokens, 300)
        timer._report()
        self.assertEqual(len(self.publisher.metrics), 1)

    def test_disagg_prefill_queued_metrics_include_compute_waiting_queue(self):
        _set_disaggregation_mode(self, DisaggregationMode.PREFILL)
        self.reporter.mode_owners.disagg_prefill = prefill_owner_shell()
        self.reporter.mode_owners.disagg_prefill.bootstrap_queue.queue = [_FakeReq(100)]
        self.scheduler.lists.waiting_queue = [_FakeReq(200), _FakeReq(50)]
        batch = self._make_batch()

        self._emit_ready(batch)

        metrics = self.publisher.metrics[0]
        self.assertEqual(metrics.queued_requests.num_prefill_requests, 3)
        self.assertEqual(metrics.queued_requests.sum_prefill_tokens, 350)
        self.assertEqual(metrics.queued_requests.num_decode_requests, 0)

    def test_disagg_decode_queued_metrics(self):
        _set_disaggregation_mode(self, DisaggregationMode.DECODE)
        owner = self.reporter.mode_owners.disagg_decode = decode_owner_shell()
        owner.prealloc_queue.queue = [
            types.SimpleNamespace(req=_FakeReq(10, output_len=5)),
            types.SimpleNamespace(req=_FakeReq(20, output_len=10)),
        ]
        owner.transfer_queue.queue = [
            types.SimpleNamespace(req=_FakeReq(30, output_len=15))
        ]
        batch = self._make_batch()

        self._emit_ready(batch)

        metrics = self.publisher.metrics[0]
        self.assertEqual(metrics.queued_requests.num_prefill_requests, 0)
        self.assertEqual(metrics.queued_requests.num_decode_requests, 3)
        self.assertEqual(metrics.queued_requests.sum_decode_kv_tokens, 15 + 30 + 45)

    def test_disagg_decode_queued_metrics_count_every_queued_list(self):
        """Ready (transferred, waiting for a row) and retracted requests are
        queued decode work too."""
        _set_disaggregation_mode(self, DisaggregationMode.DECODE)
        owner = self.reporter.mode_owners.disagg_decode = decode_owner_shell()
        owner.prealloc_queue.queue = [
            types.SimpleNamespace(req=_FakeReq(10, output_len=5))
        ]
        owner.prealloc_queue.retracted_queue = [_FakeReq(20, output_len=10)]
        owner.ready = [_FakeReq(30, output_len=15)]
        batch = self._make_batch()

        self._emit_ready(batch)

        metrics = self.publisher.metrics[0]
        self.assertEqual(metrics.queued_requests.num_decode_requests, 3)
        self.assertEqual(metrics.queued_requests.sum_decode_kv_tokens, 15 + 30 + 45)

    def test_init_metrics_uses_server_worker_id(self):
        scheduler = types.SimpleNamespace()
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

        sink = reporter.metrics_sink
        self.assertTrue(sink.enable_fpm)
        # The step launcher wraps its forward with this timer.
        self.assertIsInstance(sink.forward_pass_device_timer, DeviceTimer)
        self.assertEqual(sink._fpm_worker_id, "endpoint-42")
        self.assertEqual(sink._fpm_dp_rank, 2)
        self.assertEqual(sink._fpm_publisher.worker_id, "endpoint-42")
        self.assertEqual(sink._fpm_publisher.dp_rank, 2)
        self.assertTrue(sink._fpm_publisher.endpoint.startswith("ipc://"))
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

        self.assertFalse(reporter.metrics_sink.enable_fpm)

    def test_pp_skipped_output_comm_keeps_timing(self):
        timing = object()
        launch_result = GenerationBatchResult(
            can_run_cuda_graph=True, fpm_timing=timing
        )
        ring = types.SimpleNamespace(
            device="cpu",
            device_module=types.SimpleNamespace(
                Event=Mock, current_stream=lambda: object()
            ),
        )
        wire, result, _ = PPRing._make_skip_output_result(
            ring, types.SimpleNamespace(reqs=[object()]), launch_result
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
            lists=CoreLists(),
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
            reporter.metrics_sink.metrics_collector = SchedulerMetricsCollector(
                labels=labels, server_args=scheduler.server_args
            )
        reporter.metrics_sink.current_scheduler_metrics_enabled = True
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
        reporter.metrics_sink.log_batch_result_stats(batch, result=object())
        batch.dp_balance_stats = _dp_balance_stats(
            local_tokens=0, global_num_tokens=[8, 0], sync_wait_seconds=0.004
        )
        reporter.metrics_sink.log_batch_result_stats(batch, result=object())
        batch.dp_balance_stats = None
        reporter.metrics_sink.log_batch_result_stats(batch, result=object())

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
        reporter.metrics_sink.log_batch_result_stats(batch, result=object())

        get = registry.get_sample_value
        self.assertEqual(
            get("sglang:dp_attention_steps_total", {**labels, "rank_state": "active"}),
            1,
        )
        self.assertEqual(
            get("sglang:dp_attention_token_imbalance_ratio_count", labels), 0
        )


def _transfer_req(*, host_staged, priority):
    """A transfer queue entry: a DecodeRequest's host_staged and priority."""
    return types.SimpleNamespace(
        host_staged=host_staged,
        priority=priority,
        req=types.SimpleNamespace(priority=priority),
    )


class TestIdleMetrics(CustomTestCase):
    def setUp(self):
        self.scheduler = types.SimpleNamespace(
            lists=CoreLists(),
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
        self.reporter.metrics_sink.metrics_collector = types.SimpleNamespace(
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
        self.reporter.metrics_sink.metrics_collector = collector
        self.reporter.metrics_sink.current_scheduler_metrics_enabled = True
        host_reqs = [
            _transfer_req(host_staged=True, priority=priority) for priority in (1, 2)
        ]
        transfer_queue = types.SimpleNamespace(
            queue=[*host_reqs, _transfer_req(host_staged=False, priority=1)],
            num_pending_deferred_releases=lambda: 0,
        )
        self.reporter.mode_owners.disagg_decode = decode_owner_shell(
            transfer_queue=transfer_queue
        )
        for _ in host_reqs:
            collector.increment_decode_host_receive_reqs()

        with get_context().override_server_args(
            disaggregation_mode=DisaggregationMode.DECODE.value,
            disaggregation_decode_host_receive_threshold=0.8,
            enable_priority_scheduling=True,
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
        self.reporter.metrics_sink.current_scheduler_metrics_enabled = True
        self.reporter.metrics_sink.fwd_occupancy = 72.0
        self.reporter.metrics_sink.stats.fwd_occupancy = 72.0
        self.reporter.metrics_sink._device_timer_window_batch_count = 7

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
        self.assertTrue(math.isnan(self.reporter.metrics_sink.fwd_occupancy))
        self.assertTrue(math.isnan(self.reporter.metrics_sink.stats.fwd_occupancy))
        self.assertEqual(self.reporter.metrics_sink._device_timer_window_batch_count, 0)

    def test_idle_resets_forward_timing_when_metrics_are_disabled(self):
        self.reporter.metrics_sink.fwd_occupancy = 72.0
        self.reporter.metrics_sink.stats.fwd_occupancy = 72.0
        self.reporter.metrics_sink._device_timer_window_batch_count = 7

        with patch(
            "sglang.srt.managers.scheduler_components.metrics_reporter.ENABLE_METRICS_DEVICE_TIMER",
            True,
        ):
            self.reporter._maybe_log_idle_metrics()

        self.assertTrue(math.isnan(self.reporter.metrics_sink.fwd_occupancy))
        self.assertTrue(math.isnan(self.reporter.metrics_sink.stats.fwd_occupancy))
        self.assertEqual(self.reporter.metrics_sink._device_timer_window_batch_count, 0)
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
        self.reporter.metrics_sink.enable_metrics = True
        self.reporter.metrics_sink.scheduler_stage_metrics.enabled = True
        self.reporter.metrics_sink.metrics_collector = types.SimpleNamespace(
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
            self.reporter.metrics_sink.start_scheduler_time_accounting()
            self.reporter.metrics_sink.record_scheduler_idle()
            self.reporter.metrics_sink.record_scheduler_active(event_ns=1_300_000_000)
            self.assertEqual(
                self.reporter.metrics_sink._forward_occupancy_log_window.idle_ns,
                1_300_000_000,
            )
            self.reporter.metrics_sink.record_scheduler_active(event_ns=0)
            # Time between idle capture and accounting still counts as idle.
            self.reporter.metrics_sink.record_scheduler_idle()
            self.reporter.metrics_sink.record_scheduler_idle()

        self.assertAlmostEqual(sum(self.idle_seconds), 2.5)
        self.assertEqual(
            self.reporter.metrics_sink._forward_occupancy_log_window.idle_ns,
            1_200_000_000,
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
            self.reporter.metrics_sink.start_scheduler_time_accounting()
            accounting = self.reporter.metrics_sink._scheduler_time_accounting
            self.reporter.metrics_sink.record_scheduler_active(event_ns=200_000_000)
            self.reporter.metrics_sink.record_scheduler_idle()
            self.assertEqual(self.idle_seconds, [])
            self.assertEqual(self.process_cpu_seconds, [])
            with self.reporter.metrics_sink.scheduler_stage_metrics.record(
                SCHEDULER_STAGE_RECV_REQUESTS
            ):
                pass
            # Receive crossed the reporting deadline. Backdate only idle time,
            # not the stage drain (which has already sampled receive completion).
            self.reporter.metrics_sink.record_scheduler_active(event_ns=700_000_000)
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
            self.reporter.metrics_sink.record_scheduler_active(event_ns=1_300_000_000)

        self.assertIs(self.reporter.metrics_sink._scheduler_time_accounting, accounting)
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
            self.reporter.metrics_sink.start_scheduler_time_accounting()
            self.reporter.metrics_sink.record_scheduler_active(event_ns=-1)
            self.assertEqual(
                self.reporter.metrics_sink._forward_occupancy_log_window.idle_ns, 0
            )
            self.reporter.metrics_sink.record_scheduler_active(event_ns=1_000_000_000)

        self.assertEqual(self.idle_seconds, [])
        self.assertEqual(self.process_cpu_seconds, [0.0])


class TestEstimatedPrefillPerf(CustomTestCase):
    """Causal pair count behind ``est. prefill TFLOPS/s`` and ``estimated_flops``."""

    def setUp(self):
        self.scheduler = types.SimpleNamespace(lists=CoreLists())
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
