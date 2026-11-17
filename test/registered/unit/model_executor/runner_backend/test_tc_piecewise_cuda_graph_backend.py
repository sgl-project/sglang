"""Unit tests for ``TcPiecewiseCudaGraphBackend.capture_one`` profiling hooks.

CPU-only. This backend runs two forwards per shape (FX warmup, then capture).
The per-shape profiler schedule waits two steps, so an extra step before the
loop is required. Without it the profiler never records and
``finish_cuda_capture_profile`` crashes on ``key_averages()``.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_backend.tc_piecewise_cuda_graph_backend import (
    TcPiecewiseCudaGraphBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_UNSET = object()


def _make_backend(runner):
    backend = TcPiecewiseCudaGraphBackend.__new__(TcPiecewiseCudaGraphBackend)
    backend._cuda_graph_runner = runner
    backend._device_module = runner.device_module
    backend._tp_group = SimpleNamespace(barrier=mock.Mock(name="barrier"))
    return backend


def _make_runner(*, enable_profile, profiler):
    runner = SimpleNamespace(
        device_module=SimpleNamespace(synchronize=mock.Mock(name="synchronize")),
        enable_profile_cuda_graph=enable_profile,
    )
    if profiler is not _UNSET:
        runner._profiler = profiler
    return runner


class TestTcPiecewiseCaptureOneNoProfiling(CustomTestCase):
    def test_runs_two_forwards_without_stepping(self):
        runner = _make_runner(enable_profile=False, profiler=None)
        backend = _make_backend(runner)
        forward_fn = mock.Mock()
        post_warmup_hook = mock.Mock()

        backend.capture_one(
            ShapeKey(size=4), forward_fn, post_warmup_hook=post_warmup_hook
        )

        self.assertEqual(forward_fn.call_count, 2)
        self.assertEqual(post_warmup_hook.call_count, 2)

    def test_enable_flag_set_but_no_profiler_attr_does_not_step(self):
        runner = _make_runner(enable_profile=True, profiler=_UNSET)
        backend = _make_backend(runner)
        self.assertFalse(hasattr(runner, "_profiler"))
        forward_fn = mock.Mock()

        backend.capture_one(ShapeKey(size=2), forward_fn)

        self.assertEqual(forward_fn.call_count, 2)


class TestTcPiecewiseCaptureOneWithProfiling(CustomTestCase):
    def test_steps_once_before_the_loop_and_once_after_each_forward(self):
        order = []

        def _step():
            order.append("step")

        profiler = SimpleNamespace(step=mock.Mock(side_effect=_step))
        runner = _make_runner(enable_profile=True, profiler=profiler)
        backend = _make_backend(runner)
        forward_fn = mock.Mock(side_effect=lambda: order.append("forward"))

        backend.capture_one(ShapeKey(size=4), forward_fn)

        self.assertEqual(profiler.step.call_count, 3)
        self.assertEqual(forward_fn.call_count, 2)
        self.assertEqual(order, ["step", "forward", "step", "forward", "step"])


if __name__ == "__main__":
    unittest.main()
