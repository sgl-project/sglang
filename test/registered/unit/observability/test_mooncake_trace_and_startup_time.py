"""Unit tests for mooncake_trace.py and startup_time.py."""

import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.model_executor.graph_memory_usage import empty_graph_time_usage
from sglang.srt.observability.mooncake_trace import (
    MooncakeRequestStage,
    mooncake_trace_func,
)
from sglang.srt.observability.startup_time import (
    aggregate_scheduler_startup_times,
    build_engine_startup_time,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _DummyMooncakeWorker:
    def __init__(self, trace_ctx):
        self.trace_ctx = trace_ctx

    @mooncake_trace_func(MooncakeRequestStage.MOONCAKE_SEND)
    def send_ok(self, x: int, y: int = 1) -> int:
        """Docstring for send_ok."""
        return x + y

    @mooncake_trace_func(MooncakeRequestStage.MOONCAKE_RECV)
    def recv_fail(self) -> None:
        raise RuntimeError("transfer failed")


class TestMooncakeTraceFunc(unittest.TestCase):
    def test_preserves_function_metadata(self):
        self.assertEqual(_DummyMooncakeWorker.send_ok.__name__, "send_ok")
        self.assertEqual(_DummyMooncakeWorker.send_ok.__doc__, "Docstring for send_ok.")

    def test_skips_tracing_when_trace_ctx_is_none(self):
        worker = _DummyMooncakeWorker(None)
        with patch(
            "sglang.srt.observability.mooncake_trace.convert_time_to_realtime_ns"
        ) as mock_conv:
            self.assertEqual(worker.send_ok(3, y=4), 7)
            mock_conv.assert_not_called()

    def test_skips_tracing_when_tracing_disabled(self):
        trace_ctx = MagicMock()
        trace_ctx.tracing_enable = False
        worker = _DummyMooncakeWorker(trace_ctx)
        with patch(
            "sglang.srt.observability.mooncake_trace.convert_time_to_realtime_ns"
        ) as mock_conv:
            self.assertEqual(worker.send_ok(5, y=2), 7)
            mock_conv.assert_not_called()
            trace_ctx.trace_slice_start.assert_not_called()
            trace_ctx.trace_slice_end.assert_not_called()

    def test_closes_slice_when_wrapped_function_raises(self):
        trace_ctx = MagicMock()
        trace_ctx.tracing_enable = True
        worker = _DummyMooncakeWorker(trace_ctx)
        with self.assertRaisesRegex(RuntimeError, "transfer failed"):
            worker.recv_fail()
        trace_ctx.trace_slice_start.assert_called_once()
        trace_ctx.trace_slice_end.assert_called_once_with(
            MooncakeRequestStage.MOONCAKE_RECV.stage_name,
            MooncakeRequestStage.MOONCAKE_RECV.level,
        )


class TestStartupTimeAggregation(unittest.TestCase):
    def test_aggregate_handles_none_fields_and_none_cuda_graph(self):
        ranks = [
            {
                "load_weight": None,
                "kv_cache_allocation": 1.5,
                "scheduler_e2e": None,
                "cuda_graph": None,
            },
            {
                "load_weight": 3.25,
                "kv_cache_allocation": None,
                "scheduler_e2e": 6.0,
                "cuda_graph": {"decode": 2.0, "prefill": None},
            },
        ]
        res = build_engine_startup_time(ranks, tokenizer_e2e=0.75)
        self.assertEqual(res["load_weight"], 3.25)
        self.assertEqual(res["kv_cache_allocation"], 1.5)
        self.assertEqual(res["scheduler_e2e"], 6.0)
        self.assertEqual(res["cuda_graph"]["decode"], 2.0)
        self.assertEqual(res["cuda_graph"]["prefill"], 0.0)
        self.assertEqual(res["tokenizer_e2e"], 0.75)

    def test_aggregate_empty_and_none_entries(self):
        res = aggregate_scheduler_startup_times([None, {}])
        self.assertEqual(res["load_weight"], 0.0)
        self.assertEqual(res["kv_cache_allocation"], 0.0)
        self.assertEqual(res["scheduler_e2e"], 0.0)
        self.assertEqual(res["cuda_graph"], empty_graph_time_usage())


if __name__ == "__main__":
    unittest.main()
