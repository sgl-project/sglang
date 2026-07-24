"""Unit tests for ``FullCudaGraphBackend.capture_one`` profiling hooks — CPU-only.

These cover the changes from the "cuda graph profile traces" PR that wire the
runner's torch profiler into the capture loop:

  * When profiling is disabled, ``capture_one`` runs exactly two warmups + one
    capture and never touches a profiler (behavior-identical to before the PR).
  * When the runner exposes an active ``_profiler`` (``--enable-profile-cuda-graph``),
    ``capture_one`` calls ``profiler.step()`` past the two warmups and once after
    the capture (schedule ``wait=2, warmup=0, active=1``). The captured forward is
    NOT wrapped in a ``record_function``; per-bs trace naming is handled by the
    profiler's ``on_trace_ready`` callback instead.
  * The ``getattr`` guards mean a runner that sets the flag but has no
    ``_profiler`` attribute degrades gracefully (no stepping, no crash).

The real capture path needs CUDA (``torch.cuda.CUDAGraph`` + device graph
context), so those are mocked; the logic under test (call counts, ordering,
profiler stepping) is pure-Python and runs on CPU.
"""

import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
    FullCudaGraphBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# Sentinel: distinguishes "runner has no _profiler attribute" from
# "_profiler is None" in the test fixtures.
_UNSET = object()


class _FakeGraphCtx:
    """Stand-in for ``device_module.graph(...)`` — a no-op context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _make_backend(runner):
    """Build a ``FullCudaGraphBackend`` without running ``__init__`` (which would
    touch CUDA), wiring just the attributes ``capture_one`` reads."""
    backend = FullCudaGraphBackend.__new__(FullCudaGraphBackend)
    backend._graphs = {}
    backend._outputs = {}
    backend._pool = None
    backend._capture_stream = None
    backend._memory_saver_adapter = None
    backend._cuda_graph_runner = runner
    backend._device_module = runner.device_module
    backend._tp_group = runner.model_runner.tp_group
    return backend


def _make_runner(*, enable_profile, profiler, num_tokens_per_bs=1, mode_name="DECODE"):
    device_module = SimpleNamespace(
        synchronize=mock.Mock(name="synchronize"),
        graph=mock.Mock(name="graph", side_effect=lambda **kw: _FakeGraphCtx()),
    )
    tp_group = SimpleNamespace(barrier=mock.Mock(name="barrier"))
    runner = SimpleNamespace(
        device_module=device_module,
        model_runner=SimpleNamespace(tp_group=tp_group),
        num_tokens_per_bs=num_tokens_per_bs,
        capture_forward_mode=SimpleNamespace(name=mode_name),
        enable_profile_cuda_graph=enable_profile,
    )
    if profiler is not _UNSET:
        runner._profiler = profiler
    return runner


class TestCaptureOneNoProfiling(CustomTestCase):
    def test_runs_two_warmups_and_capture_without_stepping(self):
        runner = _make_runner(enable_profile=False, profiler=None)
        backend = _make_backend(runner)

        sentinel_out = object()
        forward_fn = mock.Mock(return_value=sentinel_out)
        post_warmup_hook = mock.Mock()
        shape_key = ShapeKey(size=4)

        with mock.patch("torch.cuda.CUDAGraph", return_value="GRAPH"):
            backend.capture_one(
                shape_key, forward_fn, post_warmup_hook=post_warmup_hook
            )

        # 2 warmups + 1 capture.
        self.assertEqual(forward_fn.call_count, 3)
        # post_warmup_hook only runs in the two warmup iterations.
        self.assertEqual(post_warmup_hook.call_count, 2)
        # Graph + output are recorded against the shape key.
        self.assertEqual(backend._graphs[shape_key], "GRAPH")
        self.assertIs(backend._outputs[shape_key], sentinel_out)

    def test_enable_flag_set_but_no_profiler_attr_does_not_step(self):
        # The runner advertises the flag but never created a profiler; the
        # getattr guard must keep capture_one on the non-profiling path.
        runner = _make_runner(enable_profile=True, profiler=_UNSET)
        backend = _make_backend(runner)
        self.assertFalse(hasattr(runner, "_profiler"))

        forward_fn = mock.Mock(return_value=object())
        with mock.patch("torch.cuda.CUDAGraph", return_value="GRAPH"):
            backend.capture_one(ShapeKey(size=2), forward_fn)

        self.assertEqual(forward_fn.call_count, 3)


class TestCaptureOneWithProfiling(CustomTestCase):
    def _run(self, *, size, num_tokens_per_bs, mode_name):
        profiler = SimpleNamespace(step=mock.Mock(name="step"))
        runner = _make_runner(
            enable_profile=True,
            profiler=profiler,
            num_tokens_per_bs=num_tokens_per_bs,
            mode_name=mode_name,
        )
        backend = _make_backend(runner)

        forward_fn = mock.Mock(return_value=object())
        rf_names = []

        def _fake_record_function(name):
            rf_names.append(name)
            return contextlib.nullcontext()

        with mock.patch("torch.cuda.CUDAGraph", return_value="GRAPH"), mock.patch(
            "torch.profiler.record_function", side_effect=_fake_record_function
        ):
            backend.capture_one(ShapeKey(size=size), forward_fn)

        return profiler, forward_fn, rf_names

    def test_steps_twice_in_warmup_and_once_after_capture(self):
        profiler, forward_fn, _ = self._run(
            size=4, num_tokens_per_bs=1, mode_name="DECODE"
        )
        # Schedule wait=2 + active=1 => one step per warmup (x2) + one post-capture.
        self.assertEqual(profiler.step.call_count, 3)
        self.assertEqual(forward_fn.call_count, 3)

    def test_capture_not_wrapped_in_record_function(self):
        # The capture forward is no longer wrapped in a record_function; per-bs
        # trace naming is handled by the profiler's on_trace_ready callback.
        _, _, rf_names = self._run(size=4, num_tokens_per_bs=1, mode_name="DECODE")
        self.assertEqual(rf_names, [])


if __name__ == "__main__":
    unittest.main()
