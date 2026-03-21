"""Unit tests for req_time_stats.py — no server, no model loading."""

# ── Lightweight stubs for heavy transitive deps (torch, distributed, etc.) ──
# req_time_stats.py imports from disaggregation, model_executor, and other
# modules that transitively pull in torch/CUDA.  We pre-populate sys.modules
# with minimal stubs so the module loads in a CPU-only environment.
import os
import sys
import types
from dataclasses import dataclass
from enum import Enum, IntEnum, auto


def _ensure_module(name):
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
    return sys.modules[name]


# Mirrors sglang.srt.disaggregation.utils.DisaggregationMode (stubbed to avoid torch dep).
class _DisaggregationMode(Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"


def _kv_to_page_num(n, page_size):
    return (n + page_size - 1) // page_size


_ensure_module("sglang.srt.disaggregation")
_du = _ensure_module("sglang.srt.disaggregation.utils")
_du.DisaggregationMode = _DisaggregationMode
_du.kv_to_page_num = _kv_to_page_num


# Mirrors sglang.srt.model_executor.forward_batch_info.ForwardMode (stubbed to avoid torch dep).
class _ForwardMode(IntEnum):
    EXTEND = auto()
    DECODE = auto()
    MIXED = auto()
    IDLE = auto()
    TARGET_VERIFY = auto()
    DRAFT_EXTEND = auto()
    DRAFT_EXTEND_V2 = auto()
    PREBUILT = auto()
    SPLIT_PREFILL = auto()
    DLLM_EXTEND = auto()

    def is_decode(self):
        return self == _ForwardMode.DECODE

    def is_prefill(self):
        return self in (
            _ForwardMode.EXTEND,
            _ForwardMode.MIXED,
            _ForwardMode.DRAFT_EXTEND,
            _ForwardMode.TARGET_VERIFY,
            _ForwardMode.SPLIT_PREFILL,
        )

    def is_prebuilt(self):
        return self == _ForwardMode.PREBUILT


_ensure_module("sglang.srt.model_executor")
_fbi = _ensure_module("sglang.srt.model_executor.forward_batch_info")
_fbi.ForwardMode = _ForwardMode


# -- sglang.srt.observability.metrics_collector --
_mc = _ensure_module("sglang.srt.observability.metrics_collector")
_mc.SchedulerMetricsCollector = type("SchedulerMetricsCollector", (), {})
_mc.TokenizerMetricsCollector = type("TokenizerMetricsCollector", (), {})


# -- sglang.srt.observability.trace --
@dataclass
class _TraceNullContext:
    tracing_enable: bool = False

    def __getattr__(self, name):
        return self

    def __call__(self, *args, **kwargs):
        return self


# -- sglang.srt.utils --
# Stub both get_bool_env_var (for req_time_stats) and get_int_env_var (for trace.py)
# so the real trace module can load without torch.
def _get_bool_env_var(name, default="false"):
    return os.getenv(name, default).lower() in ("true", "1")


def _get_int_env_var(name, default=0):
    return int(os.getenv(name, str(default)))


_su = _ensure_module("sglang.srt.utils")
_su.get_bool_env_var = _get_bool_env_var
if not hasattr(_su, "get_int_env_var"):
    _su.get_int_env_var = _get_int_env_var
_ensure_module("sglang.srt.utils.common")

# ── End stubs ────────────────────────────────────────────────────────

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu-only")

import unittest
from unittest.mock import MagicMock

import sglang.srt.observability.req_time_stats as rts_module
import sglang.srt.observability.trace as trace_module
from sglang.srt.observability.req_time_stats import (
    APIServerReqTimeStats,
    DPControllerReqTimeStats,
    ReqTimeStatsBase,
    RequestStage,
    RequestStageConfig,
    SchedulerReqTimeStats,
    calibrate_time_diff,
    convert_time_cross_thread,
    convert_time_to_realtime,
    convert_time_to_realtime_ns,
    monotonic_time,
    real_time,
    set_schedule_time_batch,
    set_time_batch,
)
from sglang.srt.observability.trace import SpanAttributes

DisaggregationMode = _DisaggregationMode
ForwardMode = _ForwardMode


class TestUtilityFunctions(unittest.TestCase):
    def test_real_time_and_monotonic_time(self):
        self.assertGreater(real_time(), 0)
        self.assertGreater(monotonic_time(), 0)

    def test_convert_time_to_realtime(self):
        result = convert_time_to_realtime(100.0)
        self.assertIsInstance(result, float)

    def test_convert_time_to_realtime_ns(self):
        result = convert_time_to_realtime_ns(100.0)
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)

    def test_convert_time_cross_thread(self):
        self.assertAlmostEqual(
            convert_time_cross_thread(10.0, old_diff=5.0, new_diff=3.0), 12.0
        )

    def test_calibrate_time_diff(self):
        calibrate_time_diff()  # should not raise


class TestRequestStage(unittest.TestCase):
    def test_stage_config_defaults(self):
        cfg = RequestStageConfig("test_stage")
        self.assertEqual(cfg.stage_name, "test_stage")
        self.assertEqual(cfg.level, 0)
        self.assertFalse(cfg.metrics_is_observed)

    def test_predefined_stages(self):
        self.assertEqual(RequestStage.TOKENIZE.stage_name, "tokenize")
        self.assertTrue(RequestStage.PREFILL_FORWARD.metrics_is_observed)
        self.assertFalse(RequestStage.PREFILL_WAITING.metrics_is_observed)
        self.assertEqual(RequestStage.ANONYMOUS.stage_name, "")


class TestReqTimeStatsBase(unittest.TestCase):
    def test_disagg_mode_str(self):
        base = ReqTimeStatsBase()
        base.disagg_mode = DisaggregationMode.NULL
        self.assertEqual(base.disagg_mode_str(), "unified")
        base.disagg_mode = DisaggregationMode.DECODE
        self.assertEqual(base.disagg_mode_str(), "decode")
        base.disagg_mode = DisaggregationMode.PREFILL
        self.assertEqual(base.disagg_mode_str(), "prefill")

    def test_set_metrics_collector(self):
        base = ReqTimeStatsBase()
        collector = MagicMock()
        base.set_metrics_collector(collector)
        self.assertTrue(base.enable_metrics)
        self.assertIs(base.metrics_collector, collector)

    def test_set_metrics_collector_falsy(self):
        base = ReqTimeStatsBase()
        base.set_metrics_collector(None)
        self.assertFalse(base.enable_metrics)

    def test_observe_per_stage_req_latency(self):
        base = ReqTimeStatsBase()
        collector = MagicMock()
        base.set_metrics_collector(collector)

        stage = RequestStageConfig("test", metrics_is_observed=True)
        base.observe_per_stage_req_latency(stage, 1.5)
        collector.observe_per_stage_req_latency.assert_called_once_with("test", 1.5)

    def test_observe_per_stage_not_observed(self):
        base = ReqTimeStatsBase()
        collector = MagicMock()
        base.set_metrics_collector(collector)

        stage = RequestStageConfig("test", metrics_is_observed=False)
        base.observe_per_stage_req_latency(stage, 1.0)
        collector.observe_per_stage_req_latency.assert_not_called()

    def test_init_trace_ctx(self):
        base = ReqTimeStatsBase()
        base.init_trace_ctx("rid-1", bootstrap_room=None)
        # TraceReqContext stub has tracing_enable=False → replaced by TraceNullContext
        self.assertFalse(base.trace_ctx.tracing_enable)

    def test_trace_slice_noop_when_tracing_disabled(self):
        base = ReqTimeStatsBase()
        base.trace_slice(RequestStage.TOKENIZE, 0.0, 1.0)

    def test_trace_slice_when_tracing_enabled(self):
        base = ReqTimeStatsBase()
        base.trace_ctx = MagicMock()
        base.trace_ctx.tracing_enable = True
        base.trace_slice(RequestStage.TOKENIZE, 0.0, 1.0, {"key": "val"})
        base.trace_ctx.trace_slice.assert_called_once()

    def test_new_from_obj_none(self):
        obj = ReqTimeStatsBase.new_from_obj(None)
        self.assertIsInstance(obj, ReqTimeStatsBase)

    def test_new_from_obj_copy(self):
        src = ReqTimeStatsBase()
        src.disagg_mode = DisaggregationMode.PREFILL
        src.enable_metrics = True
        dst = ReqTimeStatsBase.new_from_obj(src)
        self.assertEqual(dst.disagg_mode, DisaggregationMode.PREFILL)
        self.assertTrue(dst.enable_metrics)

    def test_getstate(self):
        base = ReqTimeStatsBase()
        base.disagg_mode = DisaggregationMode.DECODE
        state = base.__getstate__()
        self.assertEqual(state["disagg_mode"], DisaggregationMode.DECODE)
        self.assertFalse(state["enable_metrics"])

    def test_setstate_converts_time_fields(self):
        base = ReqTimeStatsBase()
        state = {
            "created_time": 100.0,
            "diff_realtime_monotonic": 50.0,
            "disagg_mode": DisaggregationMode.NULL,
        }
        base.__setstate__(state)
        self.assertEqual(base.disagg_mode, DisaggregationMode.NULL)
        # created_time ends with "time" → converted via convert_time_cross_thread
        expected = convert_time_cross_thread(
            100.0, 50.0, rts_module.global_diff_realtime_monotonic
        )
        self.assertAlmostEqual(base.created_time, expected)


class TestAPIServerReqTimeStats(unittest.TestCase):
    def test_setters_auto_timestamp(self):
        """All setters work with default ts=None (auto perf_counter)."""
        s = APIServerReqTimeStats()
        s.set_created_time()
        s.set_tokenize_finish_time()
        s.set_api_server_dispatch_time()
        s.set_api_server_dispatch_finish_time()
        s.set_first_token_time()
        s.set_last_time()
        s.set_response_sent_to_client_time()
        s.set_finished_time()
        self.assertGreater(s.created_time, 0)
        self.assertGreater(s.finished_time, 0)

    def test_getters(self):
        s = APIServerReqTimeStats()
        s.set_created_time(1.0)
        s.set_first_token_time(3.0)
        s.set_finished_time(5.0)
        self.assertAlmostEqual(s.get_first_token_latency(), 2.0)
        self.assertAlmostEqual(s.get_e2e_latency(), 4.0)
        self.assertAlmostEqual(s.get_decode_latency(), 2.0)

    def test_get_interval(self):
        s = APIServerReqTimeStats()
        s.set_first_token_time(monotonic_time())
        self.assertGreaterEqual(s.get_interval(), 0.0)

    def test_getstate(self):
        s = APIServerReqTimeStats(disagg_mode=DisaggregationMode.NULL)
        state = s.__getstate__()
        self.assertIn("disagg_mode", state)
        self.assertFalse(state["enable_metrics"])

    def test_get_response_sent_to_client_realtime(self):
        s = APIServerReqTimeStats()
        s.set_response_sent_to_client_time(100.0)
        result = s.get_response_sent_to_client_realtime()
        self.assertIsInstance(result, float)

    def test_convert_to_output_meta_info(self):
        s = APIServerReqTimeStats()
        s.set_created_time(1.0)
        s.set_api_server_dispatch_finish_time(2.0)
        s.set_first_token_time(3.0)
        s.set_response_sent_to_client_time(4.0)
        s.set_finished_time(5.0)

        meta = s.convert_to_output_meta_info(completion_tokens=10)
        self.assertIn("request_received_ts", meta)
        self.assertIn("api_server_dispatch_finish_ts", meta)
        self.assertIn("response_sent_to_client_ts", meta)
        self.assertIn("request_finished_ts", meta)
        self.assertIn("decode_throughput", meta)

    def test_convert_to_output_meta_info_with_scheduler_stats(self):
        s = APIServerReqTimeStats()
        s.set_created_time(1.0)
        s.set_first_token_time(3.0)
        s.set_finished_time(5.0)

        sched = MagicMock()
        sched.forward_entry_time = 2.0
        meta = s.convert_to_output_meta_info(
            scheduler_time_stats=sched, completion_tokens=10
        )
        self.assertIn("inference_time", meta)
        self.assertAlmostEqual(meta["inference_time"], 3.0)

    def test_convert_to_output_meta_info_empty(self):
        """No timestamps set → minimal meta_info."""
        s = APIServerReqTimeStats()
        meta = s.convert_to_output_meta_info()
        self.assertNotIn("request_received_ts", meta)
        self.assertNotIn("decode_throughput", meta)

    def test_convert_to_gen_ai_span_attrs(self):
        s = APIServerReqTimeStats()
        s.set_created_time(1.0)
        s.set_first_token_time(3.0)
        s.set_api_server_dispatch_finish_time(2.0)
        s.set_finished_time(5.0)

        attrs = s.convert_to_gen_ai_span_attrs()
        self.assertAlmostEqual(
            attrs[SpanAttributes.GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN], 2.0
        )
        self.assertAlmostEqual(attrs[SpanAttributes.GEN_AI_LATENCY_E2E], 4.0)
        self.assertAlmostEqual(
            attrs[SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_DECODE], 2.0
        )
        self.assertAlmostEqual(
            attrs[SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE], 3.0
        )
        self.assertAlmostEqual(
            attrs[SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL], 1.0
        )

    def test_convert_to_gen_ai_span_attrs_empty(self):
        s = APIServerReqTimeStats()
        attrs = s.convert_to_gen_ai_span_attrs()
        self.assertEqual(len(attrs), 0)


class TestDPControllerReqTimeStats(unittest.TestCase):
    def test_setters(self):
        s = DPControllerReqTimeStats()
        s.set_dp_dispatch_time()
        s.set_dp_dispatch_finish_time()
        self.assertGreater(s.dc_dispatch_time, 0)
        self.assertGreater(s.dc_dispatch_finish_time, 0)

    def test_setters_explicit(self):
        s = DPControllerReqTimeStats()
        s.set_dp_dispatch_time(10.0)
        s.set_dp_dispatch_finish_time(20.0)
        self.assertEqual(s.dc_dispatch_time, 10.0)
        self.assertEqual(s.dc_dispatch_finish_time, 20.0)

    def test_getstate(self):
        s = DPControllerReqTimeStats(disagg_mode=DisaggregationMode.NULL)
        state = s.__getstate__()
        self.assertIn("disagg_mode", state)


class TestSchedulerReqTimeStats(unittest.TestCase):
    def _make_stats(self, **kwargs):
        defaults = dict(disagg_mode=DisaggregationMode.NULL)
        defaults.update(kwargs)
        return SchedulerReqTimeStats(**defaults)

    def _make_enabled_stats(self, **kwargs):
        s = self._make_stats(**kwargs)
        s.set_metrics_collector(MagicMock())
        return s

    def test_getstate_metrics_disabled(self):
        s = self._make_stats()
        self.assertEqual(s.__getstate__(), {})

    def test_getstate_metrics_enabled(self):
        s = self._make_enabled_stats()
        s.wait_queue_entry_time = 1.0
        s.forward_entry_time = 2.0
        state = s.__getstate__()
        self.assertIn("wait_queue_entry_time", state)
        self.assertIn("forward_entry_time", state)

    def test_set_scheduler_recv_time(self):
        s = self._make_stats()
        s.set_scheduler_recv_time()
        self.assertGreater(s.scheduler_recv_time, 0)

    def test_set_prefill_run_batch_times(self):
        s = self._make_stats()
        s.set_prefill_run_batch_start_time(1.0)
        s.set_prefill_run_batch_end_time(2.0)
        self.assertEqual(s.prefill_run_batch_start_time, 1.0)
        self.assertEqual(s.prefill_run_batch_end_time, 2.0)

    def test_set_quick_finish_time(self):
        s = self._make_stats()
        s.set_quick_finish_time(5.0)
        self.assertEqual(s.completion_time, 5.0)
        self.assertEqual(s.forward_entry_time, 5.0)

    def test_set_bootstrap_done_time(self):
        s = self._make_stats()
        s.set_bootstrap_done_time(1.0)
        self.assertEqual(s.bootstrap_done_time, 1.0)
        # Second call does not overwrite
        s.set_bootstrap_done_time(2.0)
        self.assertEqual(s.bootstrap_done_time, 1.0)

    def test_set_completion_time(self):
        s = self._make_stats()
        s.set_completion_time(10.0)
        self.assertEqual(s.completion_time, 10.0)

    def test_set_prefill_transfer_queue_entry_time(self):
        s = self._make_stats()
        s.set_prefill_transfer_queue_entry_time(1.0)
        self.assertEqual(s.prefill_transfer_queue_entry_time, 1.0)

    def test_set_prefill_kv_transfer_finish_time(self):
        s = self._make_stats()
        s.prefill_transfer_queue_entry_time = 1.0
        s.set_prefill_kv_transfer_finish_time(3.0)
        self.assertEqual(s.prefill_kv_transfer_finish_time, 3.0)

    def test_set_decode_prealloc_queue_entry_time(self):
        s = self._make_stats()
        s.scheduler_recv_time = 1.0
        s.set_decode_prealloc_queue_entry_time(2.0)
        self.assertEqual(s.decode_prealloc_queue_entry_time, 2.0)

    def test_set_decode_transfer_queue_entry_time(self):
        s = self._make_stats()
        s.decode_prealloc_queue_entry_time = 1.0
        s.set_decode_transfer_queue_entry_time(2.0)
        self.assertEqual(s.decode_transfer_queue_entry_time, 2.0)

    def test_set_decode_prebuilt_finish_time(self):
        s = self._make_stats()
        s.last_forward_entry_time = 1.0
        s.set_decode_prebuilt_finish_time(3.0)
        self.assertEqual(s.decode_prebuilt_finish_time, 3.0)

    def test_set_prefill_bootstrap_queue_entry_time(self):
        s = self._make_stats()
        s.scheduler_recv_time = 1.0
        s.set_prefill_bootstrap_queue_entry_time(2.0)
        self.assertEqual(s.prefill_bootstrap_queue_entry_time, 2.0)

    def test_auto_timestamp_setters(self):
        """All setters default to perf_counter() when called without args."""
        s = self._make_stats()
        s.set_scheduler_recv_time()
        s.set_prefill_run_batch_start_time()
        s.set_prefill_run_batch_end_time()
        s.set_prefill_transfer_queue_entry_time()
        s.set_prefill_kv_transfer_finish_time()
        s.set_decode_prealloc_queue_entry_time()
        s.set_decode_transfer_queue_entry_time()
        s.set_bootstrap_done_time()
        s.set_decode_prebuilt_finish_time()
        s.set_completion_time()
        s.set_quick_finish_time()
        s.set_retract_time()
        s.set_wait_queue_entry_time()
        s.set_forward_entry_time()
        s.set_last_chunked_prefill_finish_time()
        s.set_prefill_finished_time()
        s.set_last_decode_finish_time()
        s.set_prefill_bootstrap_queue_entry_time()
        s.set_last_scheduled_time(ForwardMode.DECODE)
        self.assertGreater(s.scheduler_recv_time, 0)
        self.assertGreater(s.completion_time, 0)

    def test_set_retract_time(self):
        s = self._make_stats()
        s.last_forward_entry_time = 1.0
        s.last_prefill_finished_time = 2.0
        s.set_retract_time(5.0)
        self.assertEqual(s.last_forward_entry_time, 0.0)
        self.assertEqual(s.last_prefill_finished_time, 0.0)

    def test_set_wait_queue_entry_time_first_call_null(self):
        s = self._make_enabled_stats()
        s.scheduler_recv_time = 1.0
        s.set_wait_queue_entry_time(3.0)
        self.assertEqual(s.wait_queue_entry_time, 3.0)
        s.metrics_collector.observe_per_stage_req_latency.assert_called()

    def test_set_wait_queue_entry_time_first_call_prefill(self):
        s = self._make_enabled_stats(disagg_mode=DisaggregationMode.PREFILL)
        s.prefill_bootstrap_queue_entry_time = 1.0
        s.set_wait_queue_entry_time(3.0)
        self.assertEqual(s.wait_queue_entry_time, 3.0)

    def test_set_wait_queue_entry_time_first_call_decode(self):
        s = self._make_enabled_stats(disagg_mode=DisaggregationMode.DECODE)
        s.decode_transfer_queue_entry_time = 1.0
        s.set_wait_queue_entry_time(3.0)
        self.assertEqual(s.wait_queue_entry_time, 3.0)

    def test_set_wait_queue_entry_time_retract(self):
        s = self._make_stats()
        s.wait_queue_entry_time = 1.0  # already set
        s.set_wait_queue_entry_time(5.0)
        self.assertEqual(s.wait_queue_entry_time, 5.0)
        # retract resets these
        self.assertEqual(s.last_forward_entry_time, 0.0)

    def test_set_forward_entry_time_first_call(self):
        s = self._make_enabled_stats()
        s.wait_queue_entry_time = 1.0
        s.set_forward_entry_time(3.0)
        self.assertEqual(s.forward_entry_time, 3.0)
        self.assertEqual(s.last_forward_entry_time, 3.0)
        s.metrics_collector.observe_queue_time.assert_called_once()

    def test_set_forward_entry_time_first_call_decode(self):
        s = self._make_enabled_stats(disagg_mode=DisaggregationMode.DECODE)
        s.wait_queue_entry_time = 1.0
        s.set_forward_entry_time(3.0)
        self.assertEqual(s.forward_entry_time, 3.0)

    def test_set_forward_entry_time_retract(self):
        s = self._make_stats()
        s.forward_entry_time = 1.0  # already set
        s.last_forward_entry_time = 0.0  # reset by retract
        s.set_forward_entry_time(5.0)
        self.assertEqual(s.last_forward_entry_time, 5.0)

    def test_set_last_chunked_prefill_finish_time_first(self):
        s = self._make_stats()
        s.last_forward_entry_time = 1.0
        s.set_last_chunked_prefill_finish_time(3.0)
        self.assertEqual(s.last_chunked_prefill_finish_time, 3.0)

    def test_set_last_chunked_prefill_finish_time_subsequent(self):
        s = self._make_stats()
        s.last_forward_entry_time = 1.0
        s.last_chunked_prefill_finish_time = 2.0
        s.set_last_chunked_prefill_finish_time(4.0)
        self.assertEqual(s.last_chunked_prefill_finish_time, 4.0)

    def test_set_prefill_finished_time_first_call(self):
        s = self._make_enabled_stats()
        s.last_forward_entry_time = 1.0
        s.set_prefill_finished_time(3.0)
        self.assertEqual(s.prefill_finished_time, 3.0)
        self.assertEqual(s.last_prefill_finished_time, 3.0)

    def test_set_prefill_finished_time_retract(self):
        s = self._make_stats()
        s.prefill_finished_time = 1.0  # already set
        s.last_prefill_finished_time = 0.0  # reset by retract
        s.last_forward_entry_time = 0.5
        s.set_prefill_finished_time(5.0)
        self.assertEqual(s.last_prefill_finished_time, 5.0)

    def test_set_prefill_finished_time_retract_with_chunked(self):
        s = self._make_stats()
        s.prefill_finished_time = 1.0
        s.last_prefill_finished_time = 0.0
        s.last_chunked_prefill_finish_time = 2.0
        s.set_prefill_finished_time(5.0)
        self.assertEqual(s.last_prefill_finished_time, 5.0)

    def test_set_prefill_finished_time_tracing_enabled(self):
        s = self._make_enabled_stats()
        s.trace_ctx = MagicMock()
        s.trace_ctx.tracing_enable = True
        s.last_forward_entry_time = 1.0
        s.last_chunked_prefill_finish_time = 2.0
        s.last_decode_scheduled_time = 1.5
        s.set_prefill_finished_time(3.0)
        self.assertEqual(s.prefill_finished_time, 3.0)
        s.trace_ctx.trace_slice_end.assert_called_once()
        # NULL mode + last_decode_scheduled_time > 0 → trace_slice_start for decode
        s.trace_ctx.trace_slice_start.assert_called_once()

    def test_set_last_decode_finish_time_first(self):
        s = self._make_enabled_stats()
        s.last_prefill_finished_time = 1.0
        s.last_decode_scheduled_time = 0.5
        s.set_last_decode_finish_time(3.0)
        self.assertEqual(s.last_decode_finish_time, 3.0)
        self.assertEqual(s.decode_ct, 1)

    def test_set_last_decode_finish_time_first_decode_mode(self):
        s = self._make_enabled_stats(disagg_mode=DisaggregationMode.DECODE)
        s.decode_prebuilt_finish_time = 1.0
        s.set_last_decode_finish_time(3.0)
        self.assertEqual(s.decode_ct, 1)

    def test_set_last_decode_finish_time_first_with_scheduled(self):
        s = self._make_enabled_stats()
        s.last_prefill_finished_time = 1.0
        s.last_decode_scheduled_time = 2.0
        s.set_last_decode_finish_time(3.0)
        self.assertEqual(s.decode_ct, 1)

    def test_set_last_decode_finish_time_subsequent(self):
        s = self._make_enabled_stats()
        s.last_decode_finish_time = 1.0
        s.set_last_decode_finish_time(3.0)
        self.assertEqual(s.decode_ct, 1)

    def test_set_last_scheduled_time_decode(self):
        s = self._make_stats()
        s.set_last_scheduled_time(ForwardMode.DECODE, 5.0)
        self.assertEqual(s.last_decode_scheduled_time, 5.0)

    def test_set_last_scheduled_time_extend(self):
        s = self._make_stats()
        s.set_last_scheduled_time(ForwardMode.EXTEND, 5.0)
        self.assertEqual(s.last_decode_scheduled_time, 0.0)  # not decode

    def test_set_last_scheduled_time_tracing_enabled(self):
        s = self._make_stats()
        s.trace_ctx = MagicMock()
        s.trace_ctx.tracing_enable = True
        s.last_prefill_finished_time = 1.0
        s.set_last_scheduled_time(ForwardMode.DECODE, 5.0)
        s.trace_ctx.trace_event.assert_called_once()
        # NULL mode + first decode + last_prefill_finished_time > 0 → trace decode waiting
        self.assertEqual(s.last_decode_scheduled_time, 5.0)

    def test_get_queueing_time(self):
        s = self._make_stats()
        s.forward_entry_time = 5.0
        s.wait_queue_entry_time = 2.0
        self.assertAlmostEqual(s.get_queueing_time(), 3.0)

    def test_get_prefill_waiting_latency(self):
        s = self._make_stats()
        self.assertIsNone(s.get_prefill_waiting_latency())
        s.prefill_run_batch_start_time = 3.0
        s.forward_entry_time = 1.0
        self.assertAlmostEqual(s.get_prefill_waiting_latency(), 2.0)

    def test_get_prefill_launch_latency(self):
        s = self._make_stats()
        self.assertIsNone(s.get_prefill_launch_latency())
        s.prefill_run_batch_start_time = 1.0
        s.prefill_run_batch_end_time = 3.0
        self.assertAlmostEqual(s.get_prefill_launch_latency(), 2.0)

    def test_format_duration(self):
        s = self._make_stats()
        self.assertEqual(s.format_duration(0.001), "1.00ms")

    def test_convert_to_duration_null(self):
        s = self._make_stats(
            wait_queue_entry_time=1.0,
            forward_entry_time=2.0,
            completion_time=5.0,
        )
        result = s.convert_to_duration()
        self.assertIn("queue_duration=", result)
        self.assertIn("forward_duration=", result)

    def test_convert_to_duration_prefill_no_bootstrap(self):
        s = self._make_stats(
            disagg_mode=DisaggregationMode.PREFILL,
            prefill_bootstrap_queue_entry_time=1.0,
            wait_queue_entry_time=2.0,
            forward_entry_time=3.0,
            completion_time=5.0,
        )
        result = s.convert_to_duration()
        self.assertIn("bootstrap_queue_duration", result)
        self.assertNotIn("alloc_wait", result)

    def test_convert_to_duration_prefill_with_bootstrap(self):
        s = self._make_stats(
            disagg_mode=DisaggregationMode.PREFILL,
            prefill_bootstrap_queue_entry_time=1.0,
            bootstrap_done_time=1.5,
            wait_queue_entry_time=2.0,
            forward_entry_time=3.0,
            completion_time=5.0,
        )
        result = s.convert_to_duration()
        self.assertIn("bootstrap(", result)
        self.assertIn("alloc_wait(", result)

    def test_convert_to_duration_decode_no_bootstrap(self):
        s = self._make_stats(
            disagg_mode=DisaggregationMode.DECODE,
            decode_prealloc_queue_entry_time=1.0,
            decode_transfer_queue_entry_time=2.0,
            wait_queue_entry_time=3.0,
            forward_entry_time=4.0,
            completion_time=6.0,
        )
        result = s.convert_to_duration()
        self.assertIn("prealloc_queue_duration", result)
        self.assertIn("transfer_duration=", result)
        self.assertNotIn("alloc_wait", result)

    def test_convert_to_duration_decode_with_bootstrap(self):
        s = self._make_stats(
            disagg_mode=DisaggregationMode.DECODE,
            decode_prealloc_queue_entry_time=1.0,
            bootstrap_done_time=1.5,
            decode_transfer_queue_entry_time=2.0,
            wait_queue_entry_time=3.0,
            forward_entry_time=4.0,
            completion_time=6.0,
        )
        result = s.convert_to_duration()
        self.assertIn("bootstrap(", result)
        self.assertIn("alloc_wait(", result)

    def test_convert_to_duration_unknown_mode(self):
        s = self._make_stats()
        # Force an invalid disagg_mode to hit the else branch
        s.disagg_mode = "invalid"
        self.assertEqual(s.convert_to_duration(), "Unknown Time Stats")

    def test_convert_to_output_meta_info(self):
        s = self._make_stats(
            forward_entry_time=2.0,
            prefill_finished_time=3.0,
            wait_queue_entry_time=1.0,
            prefill_run_batch_start_time=2.5,
            prefill_run_batch_end_time=2.8,
        )
        meta = s.convert_to_output_meta_info()
        self.assertIn("forward_entry_time", meta)
        self.assertIn("prefill_finished_time", meta)
        self.assertIn("queue_time", meta)
        self.assertIn("prefill_waiting_latency", meta)
        self.assertIn("prefill_launch_latency", meta)

    def test_compute_kv_transfer_metrics(self):
        s = self._make_enabled_stats()
        s.prefill_transfer_queue_entry_time = 1.0
        s.completion_time = 2.0
        result = s.compute_and_observe_kv_transfer_metrics(
            num_tokens=10, page_size=4, bytes_per_page_all_layers=1024
        )
        self.assertIn("latency_ms", result)
        self.assertIn("total_mb", result)
        self.assertIn("speed_gb_s", result)
        s.metrics_collector.observe_kv_transfer_metrics.assert_called_once()

    def test_compute_kv_transfer_with_bootstrap(self):
        s = self._make_enabled_stats()
        s.prefill_transfer_queue_entry_time = 1.0
        s.completion_time = 2.0
        s.prefill_bootstrap_queue_entry_time = 0.5
        s.bootstrap_done_time = 0.8
        s.wait_queue_entry_time = 1.0
        result = s.compute_and_observe_kv_transfer_metrics(
            num_tokens=10, page_size=4, bytes_per_page_all_layers=1024
        )
        self.assertIn("bootstrap_ms", result)
        self.assertIn("alloc_ms", result)
        s.metrics_collector.observe_kv_transfer_bootstrap.assert_called_once()

    def test_compute_kv_transfer_metrics_none(self):
        s = self._make_stats()
        result = s.compute_and_observe_kv_transfer_metrics(
            num_tokens=10, page_size=4, bytes_per_page_all_layers=1024
        )
        self.assertIsNone(result)


class TestBatchFunctions(unittest.TestCase):
    def test_set_time_batch_empty(self):
        set_time_batch(None, "set_completion_time")
        set_time_batch([], "set_completion_time")

    def test_set_time_batch(self):
        req = MagicMock()
        set_time_batch([req], "set_completion_time")
        req.time_stats.set_completion_time.assert_called_once()

    def test_set_schedule_time_batch_tracing_disabled(self):
        batch = MagicMock()
        set_schedule_time_batch(batch)
        # Tracing is disabled in our stub → returns early

    def test_set_schedule_time_batch_tracing_enabled(self):
        orig = trace_module.get_global_tracing_enabled
        trace_module.get_global_tracing_enabled = lambda: True
        rts_module.get_global_tracing_enabled = lambda: True
        try:
            req = MagicMock()
            batch = MagicMock()
            batch.reqs = [req]
            batch.forward_mode = ForwardMode.DECODE
            batch.forward_mode.is_decode = lambda: True
            batch.forward_mode.is_prefill = lambda: False
            batch.forward_mode.is_prebuilt = lambda: False
            set_schedule_time_batch(batch)
            req.time_stats.set_last_scheduled_time.assert_called_once()
        finally:
            trace_module.get_global_tracing_enabled = orig
            rts_module.get_global_tracing_enabled = orig


if __name__ == "__main__":
    unittest.main()
