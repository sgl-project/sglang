"""Unit tests for ``BreakableCudaGraphBackend.capture_one`` profiling hooks.

CPU-only. The real capture path needs a device graph, so the graph objects are
mocked. The logic under test is the profiler stepping:

  * With profiling disabled, capture runs two warmups plus one capture and
    never calls ``profiler.step()``.
  * With an active ``_profiler`` (``--enable-profile-cuda-graph`` plus
    ``SGLANG_GRAPH_BATCH_CAPTURE``), ``capture_one`` steps past the two warmups
    and once after the capture (schedule ``wait=2, warmup=0, active=1``).
  * A runner that sets the flag but has no ``_profiler`` attribute does not step.
"""

import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_backend import (
    breakable_cuda_graph_backend as bcg_mod,
)
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_UNSET = object()


def _make_backend(runner):
    backend = BreakableCudaGraphBackend.__new__(BreakableCudaGraphBackend)
    backend._graphs = {}
    backend._outputs = {}
    backend._capture_inputs = {}
    backend._pool = None
    backend._capture_stream = None
    backend._shared_output_buffer = None
    backend._debug_eager = False
    backend._hicache_graph_events = None
    backend._precarve = SimpleNamespace(
        measure=contextlib.nullcontext, mint=mock.Mock()
    )
    backend._cuda_graph_runner = runner
    backend._device_module = runner.device_module
    backend._tp_group = SimpleNamespace(barrier=mock.Mock(name="barrier"))
    backend.deduped_cuda_graph = None
    return backend


def _make_runner(*, enable_profile, profiler):
    runner = SimpleNamespace(
        device_module=SimpleNamespace(synchronize=mock.Mock(name="synchronize")),
        enable_profile_cuda_graph=enable_profile,
        cuda_graph_output_capacity_rows=lambda output: None,
        cuda_graph_output_rows=lambda output: None,
    )
    if profiler is not _UNSET:
        runner._profiler = profiler
    return runner


@contextlib.contextmanager
def _mocked_graph():
    with (
        mock.patch.object(bcg_mod, "BreakableCUDAGraph", return_value="GRAPH"),
        mock.patch.object(
            bcg_mod,
            "BreakableCUDAGraphCapture",
            side_effect=lambda **kwargs: contextlib.nullcontext(),
        ),
        mock.patch.object(bcg_mod, "graph_pool_capture_scope", contextlib.nullcontext),
    ):
        yield


class TestBreakableCaptureOneNoProfiling(CustomTestCase):
    def test_runs_two_warmups_and_capture_without_stepping(self):
        runner = _make_runner(enable_profile=False, profiler=None)
        backend = _make_backend(runner)
        sentinel = torch.ones((4, 2))
        forward_fn = mock.Mock(return_value=sentinel)
        post_warmup_hook = mock.Mock()
        shape_key = ShapeKey(size=4)

        with _mocked_graph():
            backend.capture_one(
                shape_key, forward_fn, post_warmup_hook=post_warmup_hook
            )

        self.assertEqual(forward_fn.call_count, 3)
        self.assertEqual(post_warmup_hook.call_count, 2)
        self.assertEqual(backend._graphs[shape_key], "GRAPH")
        self.assertEqual(tuple(backend._outputs[shape_key].shape), (4, 2))

    def test_enable_flag_set_but_no_profiler_attr_does_not_step(self):
        runner = _make_runner(enable_profile=True, profiler=_UNSET)
        backend = _make_backend(runner)
        self.assertFalse(hasattr(runner, "_profiler"))
        forward_fn = mock.Mock(return_value=torch.ones((2, 2)))
        with _mocked_graph():
            backend.capture_one(ShapeKey(size=2), forward_fn)
        self.assertEqual(forward_fn.call_count, 3)


class TestBreakableCaptureOneWithProfiling(CustomTestCase):
    def test_steps_twice_in_warmup_and_once_after_capture(self):
        profiler = SimpleNamespace(step=mock.Mock(name="step"))
        runner = _make_runner(enable_profile=True, profiler=profiler)
        backend = _make_backend(runner)
        forward_fn = mock.Mock(return_value=torch.ones((4, 2)))

        with _mocked_graph():
            backend.capture_one(ShapeKey(size=4), forward_fn)

        self.assertEqual(profiler.step.call_count, 3)
        self.assertEqual(forward_fn.call_count, 3)


if __name__ == "__main__":
    unittest.main()
