from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import types
import unittest
from unittest.mock import patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    PrefillStats,
    SchedulerMetricsReporter,
)


def _make_ps(**overrides) -> ParallelState:
    """Build a ParallelState with reasonable defaults for tests; override fields via kwargs."""
    defaults = dict(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        dp_rank=None,
        dp_size=1,
        attn_tp_rank=0,
        attn_tp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        attn_dp_rank=0,
        attn_dp_size=1,
        moe_ep_rank=0,
        moe_ep_size=1,
        moe_dp_rank=None,
        moe_dp_size=1,
        gpu_id=0,
    )
    defaults.update(overrides)
    return ParallelState(**defaults)


def _make_server_args(**overrides):
    defaults = dict(
        benchmark_mode=None,
        enable_forward_pass_metrics=False,
        forward_pass_metrics_worker_id="",
        forward_pass_metrics_ipc_name=None,
        enable_metrics=False,
        enable_metrics_for_all_schedulers=False,
        kv_events_config=None,
        enable_mfu_metrics=False,
        extra_metric_labels=None,
    )
    defaults.update(overrides)
    return _fake_server_args(**defaults)


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


def _fake_server_args(**fields):
    """server_args stand-in: carries fields and the override() entry point."""
    fields.setdefault("decode_log_interval", 40)
    ns = types.SimpleNamespace(**fields)

    def _override(source, **updates):
        for key, value in updates.items():
            setattr(ns, key, value)

    ns.override = _override
    return ns


def _make_reporter(scheduler) -> SchedulerMetricsReporter:
    if not hasattr(scheduler, "server_args"):
        scheduler.server_args = _make_server_args()
    if not hasattr(scheduler, "ps"):
        scheduler.ps = types.SimpleNamespace(attn_tp_rank=0, attn_cp_rank=0)
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
        self.reporter = _make_reporter(self.scheduler)
        self.scheduler.enable_fpm = True

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
            emitted = self.reporter._emit_forward_pass_metrics(batch)

        self.assertEqual(len(self.scheduler._fpm_publisher.metrics), 1)
        metrics = self.scheduler._fpm_publisher.metrics[0]
        self.assertIs(emitted, metrics)
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

    def test_disagg_prefill_queued_metrics(self):
        self.scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        self.scheduler.disagg_prefill_bootstrap_queue = types.SimpleNamespace(
            queue=[_FakeReq(100), _FakeReq(200), _FakeReq(50)],
        )
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
        scheduler.server_args = _make_server_args(
            enable_forward_pass_metrics=True,
            forward_pass_metrics_worker_id="endpoint-42",
        )
        scheduler.ps = _make_ps(attn_tp_rank=0, dp_rank=2, pp_rank=0, pp_size=1)
        scheduler.enable_kv_cache_events = False

        with patch(
            "sglang.srt.observability.forward_pass_metrics._FpmPublisherThread",
            _DummyPublisherThread,
        ):
            _make_reporter(scheduler)

        self.assertTrue(scheduler.enable_fpm)
        self.assertEqual(scheduler._fpm_worker_id, "endpoint-42")
        self.assertEqual(scheduler._fpm_dp_rank, 2)
        self.assertEqual(scheduler._fpm_publisher.worker_id, "endpoint-42")
        self.assertEqual(scheduler._fpm_publisher.dp_rank, 2)
        self.assertTrue(scheduler._fpm_publisher.endpoint.startswith("ipc://"))
        self.assertIsNotNone(scheduler.server_args.forward_pass_metrics_ipc_name)

    def test_init_fpm_disabled_on_non_last_pp_rank(self):
        scheduler = types.SimpleNamespace()
        scheduler.server_args = _make_server_args(
            enable_forward_pass_metrics=True,
            forward_pass_metrics_worker_id="endpoint-42",
        )
        scheduler.ps = _make_ps(attn_tp_rank=0, dp_rank=0, pp_rank=0, pp_size=2)
        scheduler.enable_kv_cache_events = False

        with patch(
            "sglang.srt.observability.forward_pass_metrics._FpmPublisherThread",
            _DummyPublisherThread,
        ):
            _make_reporter(scheduler)

        self.assertFalse(scheduler.enable_fpm)

    def test_init_fpm_forced_on_for_benchmark_rank(self):
        rank_cases = (
            {"attn_tp_rank": 1, "attn_tp_size": 2},
            {"attn_cp_rank": 1, "attn_cp_size": 2},
        )
        for rank_overrides in rank_cases:
            with self.subTest(rank_overrides=rank_overrides):
                scheduler = types.SimpleNamespace(
                    server_args=_make_server_args(
                        benchmark_mode="agg", enable_forward_pass_metrics=True
                    ),
                    ps=_make_ps(**rank_overrides),
                    enable_kv_cache_events=False,
                )
                with patch(
                    "sglang.srt.observability.forward_pass_metrics._FpmPublisherThread",
                    _DummyPublisherThread,
                ):
                    reporter = _make_reporter(scheduler)

                self.assertTrue(scheduler.enable_fpm)
                self.assertFalse(scheduler._fpm_is_real_rank)
                self.assertTrue(scheduler._fpm_benchmark_forced)
                reporter.shutdown_benchmark_forced_fpm()
                self.assertFalse(scheduler._fpm_benchmark_forced)


if __name__ == "__main__":
    unittest.main()
