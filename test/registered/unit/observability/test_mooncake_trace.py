"""Unit tests for observability/mooncake_trace.py - no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.observability.mooncake_trace import (
    mooncake_trace_func,
    mooncake_trace_slice,
)
from sglang.test.test_utils import CustomTestCase

_PATCH_CONVERT = "sglang.srt.observability.mooncake_trace.convert_time_to_realtime_ns"


class TestMooncakeTraceSlice(CustomTestCase):
    def setUp(self):
        self.stage = MagicMock()
        self.stage.stage_name = "test_stage"
        self.stage.level = 2

    def test_none_ctx_returns_early(self):
        """None trace_ctx returns immediately without calling convert or any trace method."""
        with patch(_PATCH_CONVERT) as mock_convert:
            mooncake_trace_slice(None, self.stage, 0.5)
        mock_convert.assert_not_called()

    def test_disabled_tracing_skips_all_calls(self):
        """tracing_enable=False returns before convert, trace_slice_start, and trace_slice_end."""
        ctx = MagicMock()
        ctx.tracing_enable = False
        with patch(_PATCH_CONVERT) as mock_convert:
            mooncake_trace_slice(ctx, self.stage, 0.5)
        mock_convert.assert_not_called()
        ctx.trace_slice_start.assert_not_called()
        ctx.trace_slice_end.assert_not_called()

    def test_enabled_tracing_calls_start_and_end(self):
        """Enabled tracing converts start_ts and calls trace_slice_start then trace_slice_end."""
        ctx = MagicMock()
        ctx.tracing_enable = True
        with patch(_PATCH_CONVERT, return_value=9999) as mock_convert:
            mooncake_trace_slice(ctx, self.stage, 0.5)
        mock_convert.assert_called_once_with(0.5)
        ctx.trace_slice_start.assert_called_once_with("test_stage", 2, 9999)
        ctx.trace_slice_end.assert_called_once_with(
            "test_stage", 2, thread_finish_flag=False
        )

    def test_thread_finish_flag_forwarded(self):
        """thread_finish_flag=True propagates to the trace_slice_end call."""
        ctx = MagicMock()
        ctx.tracing_enable = True
        with patch(_PATCH_CONVERT, return_value=1):
            mooncake_trace_slice(ctx, self.stage, 0.5, thread_finish_flag=True)
        ctx.trace_slice_end.assert_called_once_with(
            "test_stage", 2, thread_finish_flag=True
        )


class TestMooncakeTraceFunc(CustomTestCase):
    def setUp(self):
        self.stage = MagicMock()
        self.stage.stage_name = "op_name"
        self.stage.level = 1

    def _make_doubler(self):
        """Return a function decorated with mooncake_trace_func that doubles its argument."""

        @mooncake_trace_func(self.stage)
        def fn(self_inner, x):
            return x * 2

        return fn

    def test_none_ctx_calls_original_function(self):
        """self.trace_ctx=None bypasses tracing: convert not called, result correct."""
        fn = self._make_doubler()
        caller = MagicMock()
        caller.trace_ctx = None
        with patch(_PATCH_CONVERT) as mock_convert:
            result = fn(caller, 3)
        self.assertEqual(result, 6)
        mock_convert.assert_not_called()

    def test_enabled_ctx_calls_start_and_end(self):
        """Non-None trace_ctx causes trace_slice_start and trace_slice_end to be called."""
        fn = self._make_doubler()
        caller = MagicMock()
        with patch(_PATCH_CONVERT, return_value=42):
            result = fn(caller, 5)
        self.assertEqual(result, 10)
        caller.trace_ctx.trace_slice_start.assert_called_once_with("op_name", 1, 42)
        # mooncake_trace_func does not pass thread_finish_flag to trace_slice_end
        caller.trace_ctx.trace_slice_end.assert_called_once_with("op_name", 1)

    def test_return_value_preserved(self):
        """Wrapped function return value passes through the decorator unchanged."""

        @mooncake_trace_func(self.stage)
        def fn(self_inner, a, b):
            return a + b

        caller = MagicMock()
        with patch(_PATCH_CONVERT, return_value=1):
            result = fn(caller, 7, 8)
        self.assertEqual(result, 15)

    def test_kwargs_forwarded(self):
        """Keyword arguments are forwarded correctly to the wrapped function."""

        @mooncake_trace_func(self.stage)
        def fn(self_inner, x, multiplier=1):
            return x * multiplier

        caller = MagicMock()
        with patch(_PATCH_CONVERT, return_value=1):
            result = fn(caller, 3, multiplier=4)
        self.assertEqual(result, 12)


if __name__ == "__main__":
    unittest.main()
