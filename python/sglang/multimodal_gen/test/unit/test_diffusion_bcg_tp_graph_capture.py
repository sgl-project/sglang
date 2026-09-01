"""BCG capture must run inside the TP group's graph-capture context.

Custom all-reduce records every capture-time input into a rank-data slot whose
peer pointers are written only by the ``register_graph_buffers()`` that runs when
``CustomAllreduce.capture()`` exits. That registration is a host-side IPC
exchange, so it cannot happen inside the captured region -- it has to be driven
by a context that wraps the whole capture. Two things have to hold:

1. ``GroupCoordinator.graph_capture()`` enters ``srt_custom_allreduce.capture()``
   and leaves it only after the captured body is done.
2. ``BaseBreakableCudaGraphRunner._capture()`` wraps its capture in the TP
   group's ``graph_capture()``.

Break either and BCG under ``--tp-size > 1`` faults on replay with
``cudaErrorIllegalAddress``. The end-to-end guard for that is
``single_test_file/test_diffusion_bcg_tp2_zimage_turbo.py``, which needs 2 GPUs;
these tests guard the same wiring on every commit without one.
"""

from __future__ import annotations

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.multimodal_gen.runtime.breakable_cuda_graph import runner as runner_mod
from sglang.multimodal_gen.runtime.breakable_cuda_graph.runner import (
    BaseBreakableCudaGraphRunner,
)
from sglang.multimodal_gen.runtime.distributed.group_coordinator import (
    GraphCaptureContext,
    GroupCoordinator,
)
from sglang.test.test_utils import CustomTestCase


def _recording_context(events: list, name: str):
    """A context manager that appends ``name`` enter/exit to ``events``."""

    @contextlib.contextmanager
    def _ctx(*args, **kwargs):
        events.append(f"enter:{name}")
        try:
            yield MagicMock()
        finally:
            events.append(f"exit:{name}")

    return _ctx


class TestBCGTPGraphCapture(CustomTestCase):
    # --- GroupCoordinator.graph_capture -> CustomAllreduce.capture ---------- #

    def _run_graph_capture(self, custom_ar, events):
        """Drive the CUDA branch of graph_capture() with a fake custom AR."""
        group = SimpleNamespace(srt_custom_allreduce=custom_ar)
        ctx = GraphCaptureContext(MagicMock())
        with patch(
            "sglang.multimodal_gen.runtime.distributed.group_coordinator.current_platform.is_cuda_alike",
            return_value=True,
        ), patch("torch.cuda.stream"), patch("torch.cuda.current_stream"):
            with GroupCoordinator.graph_capture(group, ctx) as yielded:
                events.append("body")
        return yielded, ctx

    def test_graph_capture_enters_custom_allreduce_capture(self):
        events = []
        custom_ar = SimpleNamespace(capture=_recording_context(events, "ca"))

        yielded, ctx = self._run_graph_capture(custom_ar, events)

        # The body must run *inside* capture(), so registration (which happens on
        # its exit) lands after the captured region is closed.
        self.assertEqual(events, ["enter:ca", "body", "exit:ca"])
        self.assertIs(yielded, ctx)

    def test_graph_capture_without_custom_allreduce_is_a_noop(self):
        events = []

        yielded, ctx = self._run_graph_capture(None, events)

        self.assertEqual(events, ["body"])
        self.assertIs(yielded, ctx)

    # --- runner._tp_graph_capture -> GroupCoordinator.graph_capture -------- #

    def _tp_graph_capture_events(self, *, world_size: int, initialized: bool = True):
        events = []
        capture_stream = MagicMock()
        tp_group = MagicMock()
        tp_group.world_size = world_size
        tp_group.graph_capture = _recording_context(events, "tp")
        runner = SimpleNamespace(_capture_stream=capture_stream)

        with patch(
            "sglang.multimodal_gen.runtime.distributed.parallel_state.model_parallel_is_initialized",
            return_value=initialized,
        ), patch(
            "sglang.multimodal_gen.runtime.distributed.parallel_state.get_tp_group",
            return_value=tp_group,
        ):
            with BaseBreakableCudaGraphRunner._tp_graph_capture(runner):
                events.append("body")
        return events, capture_stream

    def test_tp_graph_capture_enters_tp_group_context(self):
        events, _ = self._tp_graph_capture_events(world_size=2)

        self.assertEqual(events, ["enter:tp", "body", "exit:tp"])

    def test_tp_graph_capture_reuses_the_runners_capture_stream(self):
        # Handing our own stream in keeps graph_capture() from creating a second
        # one that nothing captures on.
        events = []
        capture_stream = MagicMock()
        tp_group = MagicMock()
        tp_group.world_size = 2
        recorded = {}

        @contextlib.contextmanager
        def _graph_capture(graph_capture_context=None):
            recorded["ctx"] = graph_capture_context
            yield graph_capture_context

        tp_group.graph_capture = _graph_capture
        runner = SimpleNamespace(_capture_stream=capture_stream)

        with patch(
            "sglang.multimodal_gen.runtime.distributed.parallel_state.model_parallel_is_initialized",
            return_value=True,
        ), patch(
            "sglang.multimodal_gen.runtime.distributed.parallel_state.get_tp_group",
            return_value=tp_group,
        ):
            with BaseBreakableCudaGraphRunner._tp_graph_capture(runner):
                pass

        self.assertIsNotNone(recorded["ctx"])
        self.assertIs(recorded["ctx"].stream, capture_stream)

    def test_tp_graph_capture_is_noop_without_tensor_parallelism(self):
        single_gpu, _ = self._tp_graph_capture_events(world_size=1)
        self.assertEqual(single_gpu, ["body"])

        uninitialized, _ = self._tp_graph_capture_events(
            world_size=2, initialized=False
        )
        self.assertEqual(uninitialized, ["body"])

    # --- runner._capture wraps the graph capture in the TP context --------- #

    def test_capture_wraps_graph_capture_in_the_tp_context(self):
        events = []
        runner = object.__new__(BaseBreakableCudaGraphRunner)
        runner.transformer = lambda **kwargs: torch.zeros(1)
        runner.device = "cpu"
        runner.device_module = MagicMock()
        runner._pool = (0, 0)
        runner._capture_stream = MagicMock()
        runner.entries = {}
        runner._blocked = set()
        runner._disabled_reason = None
        runner.max_entries = 0
        runner.max_segments = 0

        graph = MagicMock()
        graph._segments = []
        kwargs = {"hidden_states": torch.zeros(1)}

        with patch.object(
            BaseBreakableCudaGraphRunner,
            "_tp_graph_capture",
            _recording_context(events, "tp"),
        ), patch.object(
            runner_mod, "BreakableCUDAGraph", return_value=graph
        ), patch.object(
            runner_mod,
            "enable_breakable_cuda_graph",
            _recording_context(events, "bcg_enable"),
        ), patch.object(
            runner_mod,
            "BreakableCUDAGraphCapture",
            _recording_context(events, "bcg_capture"),
        ):
            runner._capture(kwargs, key=runner_mod._signature_kwargs(kwargs))

        # The TP context must be the outermost one: registration on its exit has
        # to happen after the captured region is closed, and the eager warmup
        # forwards above it must not run with _IS_CAPTURING set.
        self.assertEqual(
            events,
            [
                "enter:tp",
                "enter:bcg_enable",
                "enter:bcg_capture",
                "exit:bcg_capture",
                "exit:bcg_enable",
                "exit:tp",
            ],
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
