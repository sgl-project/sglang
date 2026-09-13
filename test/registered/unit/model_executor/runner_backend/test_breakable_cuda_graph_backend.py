"""Ordering regression for ``BreakableCudaGraphBackend.capture_one`` — CPU-only.

Bug mechanism (black-box): ``capture_one`` runs two eager warmup iterations and,
on each, calls ``post_warmup_hook`` — the hook the attention backend uses to reset
state that warmup mutated. It then constructs the ``BreakableCUDAGraph`` and enters
capture. Between the *final* warmup/hook and that construction there was no
device synchronize and no TP barrier, so asynchronous work still in flight from the
last warmup (or from the hook) could straddle the capture boundary, and the TP ranks
were not aligned before one of them started capturing.

``FullCudaGraphBackend.capture_one`` has the same warmup-to-capture completion gap;
closing it there is what #33795 proposes. This case pins the invariant for the
breakable backend, which is the default prefill graph backend on CUDA.

Guarded invariant — after the last ``post_warmup_hook`` and before the
``BreakableCUDAGraph`` is constructed there must be at least one
``device_module.synchronize()`` followed by at least one ``tp_group.barrier()``.
A future diff that drops either call, or reorders them, turns this case red.

The real capture path needs CUDA (breakable graph capture + device graph pool), so
the graph type and its capture context are mocked; the logic under test (call
counts and call ordering) is pure Python and runs on CPU.
"""

import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_backend import (
    breakable_cuda_graph_backend as bcg_module,
)
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _make_backend(call_log):
    """Build a ``BreakableCudaGraphBackend`` without running ``__init__`` (which
    would touch CUDA), wiring just the attributes ``capture_one`` reads."""
    backend = BreakableCudaGraphBackend.__new__(BreakableCudaGraphBackend)
    backend._graphs = {}
    backend._outputs = {}
    backend._capture_inputs = {}
    backend._pool = None
    backend._capture_stream = None
    backend._debug_eager = False
    backend._shared_output_buffer = None
    backend._memory_saver_adapter = None
    backend.deduped_cuda_graph = None
    backend._device_module = SimpleNamespace(
        synchronize=mock.Mock(side_effect=lambda: call_log.append("synchronize"))
    )
    backend._tp_group = SimpleNamespace(
        barrier=mock.Mock(side_effect=lambda: call_log.append("barrier"))
    )
    return backend


class TestBreakableCaptureOrdering(CustomTestCase):
    def test_syncs_after_final_warmup_before_graph_construction(self):
        call_log = []

        def forward_fn():
            call_log.append("forward")
            return None

        def post_warmup_hook():
            call_log.append("post_warmup_hook")

        class FakeBreakableCUDAGraph:
            def __init__(self, *args, **kwargs):
                call_log.append("breakable_cudagraph_create")

        with mock.patch.object(
            bcg_module, "BreakableCUDAGraph", FakeBreakableCUDAGraph
        ), mock.patch.object(
            bcg_module,
            "BreakableCUDAGraphCapture",
            side_effect=lambda **kwargs: contextlib.nullcontext(),
        ):
            _make_backend(call_log).capture_one(
                ShapeKey(size=4),
                forward_fn,
                post_warmup_hook=post_warmup_hook,
            )

        # Two warmups plus the captured forward; the hook only runs in the warmups.
        self.assertEqual(call_log.count("forward"), 3)
        self.assertEqual(call_log.count("post_warmup_hook"), 2)
        self.assertEqual(call_log.count("breakable_cudagraph_create"), 1)

        graph_idx = call_log.index("breakable_cudagraph_create")
        last_hook_idx = max(
            i
            for i, value in enumerate(call_log)
            if value == "post_warmup_hook" and i < graph_idx
        )

        sync_between = [
            i
            for i, value in enumerate(call_log)
            if value == "synchronize" and last_hook_idx < i < graph_idx
        ]
        barrier_between = [
            i
            for i, value in enumerate(call_log)
            if value == "barrier" and last_hook_idx < i < graph_idx
        ]

        self.assertTrue(
            sync_between,
            "No synchronize() after final warmup hook before BCG construction: "
            f"{call_log}",
        )
        self.assertTrue(
            barrier_between,
            "No barrier() after final warmup hook before BCG construction: "
            f"{call_log}",
        )
        # The device must be drained before the ranks rendezvous.
        self.assertLess(sync_between[0], barrier_between[0])
        self.assertLess(barrier_between[0], graph_idx)


if __name__ == "__main__":
    unittest.main()
