from sglang.srt.runtime_context import get_context, get_observability
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import math
import types
import unittest
from unittest.mock import patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    PrefillStats,
    SchedulerMetricsReporter,
    _CacheHitRateWindow,
)
from sglang.test.test_utils import CustomTestCase


def _make_ps(**overrides) -> ParallelState:
    """Build a ParallelState with reasonable defaults for tests; override fields via kwargs."""
    defaults = dict(
        dp_rank=None,
        moe_dp_rank=None,
    )
    defaults.update(overrides)
    return ParallelState.trivial(**defaults)


class _FakeReq:
    def __init__(
        self,
        prompt_len: int,
        output_len: int = 0,
        prefix_len: int = 0,
    ):
        self.origin_input_ids = list(range(prompt_len))
        self.output_ids = list(range(output_len))
        self.prefix_indices = list(range(prefix_len))
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


class _DummyPublisherThread:
    def __init__(self, endpoint: str, worker_id: str, dp_rank: int, **_: object):
        self.endpoint = endpoint
        self.worker_id = worker_id
        self.dp_rank = dp_rank

    def shutdown(self):
        pass


def _publish_server_args(test, **fields):
    """Publish a config for the reporter under test and return the instance."""
    fields.setdefault("decode_log_interval", 40)
    override = get_context().override_server_args(**fields)
    server_args = override.install()
    test.addCleanup(override.restore)
    return server_args


def _make_reporter(test, scheduler) -> SchedulerMetricsReporter:
    if not hasattr(scheduler, "server_args"):
        scheduler.server_args = _publish_server_args(
            test,
            enable_metrics=False,
            enable_metrics_for_all_schedulers=False,
            kv_events_config=None,
            enable_mfu_metrics=False,
            enable_forward_pass_metrics=False,
        )
    if not hasattr(scheduler, "ps"):
        scheduler.ps = ParallelState.trivial()
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
    context = types.SimpleNamespace(
        enable_metrics=False,
        is_stats_logging_rank=True,
        current_scheduler_metrics_enabled=False,
        enable_kv_cache_events=False,
        collector=None,
    )
    return SchedulerMetricsReporter(
        scheduler=scheduler,
        tp_rank=0,
        pp_rank=0,
        dp_rank=0,
        metrics_collector_context=context,
        metrics_collector=None,
    )


class TestForwardPassMetrics(unittest.TestCase):
    def setUp(self):
        self.scheduler = types.SimpleNamespace()
        self.scheduler._fpm_worker_id = "worker-7"
        self.scheduler._fpm_dp_rank = 0
        self.scheduler._fpm_publisher = _CollectingPublisher()
        self.scheduler._fpm_uses_device_timer = False
        self.scheduler._fpm_gpu_time_acc = 0.0
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
            prefill_stats=None,
            seq_lens_cpu=[],
            fpm_start_time=100.0,
        )
        defaults.update(overrides)
        return types.SimpleNamespace(**defaults)

    def test_emit_mixed_batch_separates_prefill_and_decode(self):
        self.scheduler._fpm_dp_rank = 3
        self.scheduler.waiting_queue = [_FakeReq(6), _FakeReq(4, output_len=2)]

        prefill_a = _FakeReq(10, prefix_len=2)
        prefill_b = _FakeReq(14, prefix_len=3)
        decode_req = _FakeReq(8, output_len=3)
        batch = self._make_batch(
            forward_mode=_FakeForwardMode(is_mixed=True, is_extend=True),
            reqs=[prefill_a, prefill_b, decode_req],
            decoding_reqs=[decode_req],
            prefill_stats=PrefillStats(
                log_input_tokens=12,
                log_hit_tokens=5,
                new_token_ratio=1.0,
                num_running_reqs=types.SimpleNamespace(),
                num_new_seqs=2,
            ),
            seq_lens_cpu=[decode_req.seqlen],
        )

        with patch(
            "sglang.srt.managers.scheduler_components.metrics_reporter.time.monotonic",
            return_value=104.5,
        ):
            self.reporter._emit_forward_pass_metrics(batch)

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

    def test_emit_uses_device_timer_gpu_time(self):
        self.scheduler._fpm_uses_device_timer = True
        self.scheduler._fpm_gpu_time_acc = 0.042
        self.reporter.forward_pass_device_timer = types.SimpleNamespace(
            _report=lambda: None,
        )
        batch = self._make_batch()

        self.reporter._emit_forward_pass_metrics(batch)

        self.assertEqual(len(self.scheduler._fpm_publisher.metrics), 1)
        self.assertAlmostEqual(
            self.scheduler._fpm_publisher.metrics[0].wall_time, 0.042, places=4
        )
        self.assertAlmostEqual(self.scheduler._fpm_gpu_time_acc, 0.0)

    def test_emit_skips_when_device_timer_zero(self):
        self.scheduler._fpm_uses_device_timer = True
        self.scheduler._fpm_gpu_time_acc = 0.0
        self.reporter.forward_pass_device_timer = types.SimpleNamespace(
            _report=lambda: None,
        )
        batch = self._make_batch()

        self.reporter._emit_forward_pass_metrics(batch)

        self.assertEqual(len(self.scheduler._fpm_publisher.metrics), 0)

    def test_emit_uses_monotonic_without_device_timer(self):
        batch = self._make_batch()

        with patch(
            "sglang.srt.managers.scheduler_components.metrics_reporter.time.monotonic",
            return_value=100.035,
        ):
            self.reporter._emit_forward_pass_metrics(batch, result=None)

        self.assertEqual(len(self.scheduler._fpm_publisher.metrics), 1)
        self.assertAlmostEqual(
            self.scheduler._fpm_publisher.metrics[0].wall_time, 0.035, places=4
        )

    def test_disagg_prefill_queued_metrics_include_compute_waiting_queue(self):
        self.scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        self.scheduler.disagg_prefill_bootstrap_queue = types.SimpleNamespace(
            queue=[_FakeReq(100)],
        )
        self.scheduler.waiting_queue = [_FakeReq(200), _FakeReq(50)]
        batch = self._make_batch()

        with patch(
            "sglang.srt.managers.scheduler_components.metrics_reporter.time.monotonic",
            return_value=101.0,
        ):
            self.reporter._emit_forward_pass_metrics(batch)

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

        with patch(
            "sglang.srt.managers.scheduler_components.metrics_reporter.time.monotonic",
            return_value=101.0,
        ):
            self.reporter._emit_forward_pass_metrics(batch)

        metrics = self.scheduler._fpm_publisher.metrics[0]
        self.assertEqual(metrics.queued_requests.num_prefill_requests, 0)
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
        scheduler.ps = _make_ps(attn_tp_rank=0, dp_rank=2, pp_rank=0, pp_size=1)
        scheduler.enable_kv_cache_events = False

        with patch(
            "sglang.srt.observability.forward_pass_metrics._FpmPublisherThread",
            _DummyPublisherThread,
        ):
            reporter = _make_reporter(self, scheduler)

        self.assertTrue(scheduler.enable_fpm)
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
        scheduler.ps = _make_ps(attn_tp_rank=0, dp_rank=0, pp_rank=0, pp_size=2)
        scheduler.enable_kv_cache_events = False

        with patch(
            "sglang.srt.observability.forward_pass_metrics._FpmPublisherThread",
            _DummyPublisherThread,
        ):
            reporter = _make_reporter(self, scheduler)

        self.assertFalse(scheduler.enable_fpm)


class TestIdleMetrics(unittest.TestCase):
    def setUp(self):
        self.scheduler = types.SimpleNamespace(
            running_batch=types.SimpleNamespace(reqs=[]),
            waiting_queue=[],
            grammar_manager=[],
            enable_priority_scheduling=False,
            disaggregation_mode=DisaggregationMode.NULL,
            pool_stats_observer=types.SimpleNamespace(
                get_pool_stats=lambda: types.SimpleNamespace(
                    update_scheduler_stats=lambda _: None
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
        self.reporter = _make_reporter(self, types.SimpleNamespace())
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
            1_500_000_000,
            2_700_000_000,
            3_000_000_000,
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
            self.reporter.record_scheduler_active()
            self.reporter.record_scheduler_active()
            self.reporter.record_scheduler_idle()
            self.reporter.record_scheduler_idle()

        self.assertAlmostEqual(sum(self.idle_seconds), 2.6)
        self.assertAlmostEqual(sum(self.process_cpu_seconds), 1.9)
        self.assertAlmostEqual(
            sum(sample["seconds"] for sample in self.stage_seconds), 4.1
        )
        self.assertEqual({sample["stage"] for sample in self.stage_seconds}, {"other"})

    def test_state_transitions_accumulate_until_periodic_update(self):
        with (
            patch(
                "sglang.srt.managers.scheduler_components.metrics_reporter.time.monotonic_ns",
                side_effect=[0, 200_000_000, 400_000_000, 700_000_000, 1_100_000_000],
            ),
            patch(
                "sglang.srt.managers.scheduler_components.metrics_reporter.time.process_time_ns",
                side_effect=[0, 300_000_000],
            ) as process_time,
        ):
            self.reporter.start_scheduler_time_accounting()
            accounting = self.reporter._scheduler_time_accounting
            self.reporter.record_scheduler_active()
            self.reporter.record_scheduler_idle()
            self.reporter.record_scheduler_active()
            self.assertEqual(self.idle_seconds, [])
            self.assertEqual(self.process_cpu_seconds, [])
            self.assertEqual(
                self.reporter._scheduler_time_accounting.accumulate_idle_ns,
                500_000_000,
            )
            self.reporter.record_scheduler_active()

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
            self.reporter.record_scheduler_active()
            self.reporter.record_scheduler_active()

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
