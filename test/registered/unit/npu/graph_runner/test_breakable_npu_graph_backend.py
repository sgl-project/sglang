"""CPU unit tests for the breakable NPU graph backend's Python wiring.

Guards the NPU-specific branch points without a device: the backend
class itself, the per-segment graph class dispatch, and that the
capture path stays on portable (non-cuda-python) APIs when a breakable
capture runs on NPU.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.hardware_backend.npu.graph_runner import (
    breakable_npu_graph_backend as mod,
)
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    breakable_cuda_graph as bcg,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_runner():
    device_module = SimpleNamespace(
        current_device=mock.Mock(return_value=0),
        synchronize=mock.Mock(),
        set_device=mock.Mock(),
        Event=mock.Mock(),
    )
    return SimpleNamespace(
        device_module=device_module,
        model_runner=SimpleNamespace(device="npu"),
    )


def _flags(*, enable_torch_compile=False):
    return SimpleNamespace(
        capture=SimpleNamespace(enable_torch_compile=enable_torch_compile)
    )


class TestBreakableNPUGraphBackend(CustomTestCase):
    def test_construction_reuses_parent_backend(self):
        runner = _make_runner()
        with (
            get_parallel().override(tp_group=SimpleNamespace(barrier=mock.Mock())),
            mock.patch.object(mod, "get_flags", return_value=_flags()),
        ):
            backend = mod.BreakableNPUGraphBackend(runner)

        # The prefill runner's isinstance protocol keys off the parent class.
        self.assertIsInstance(backend, BreakableCudaGraphBackend)
        self.assertIs(backend._cuda_graph_runner, runner)

    def test_torch_compile_is_rejected(self):
        runner = _make_runner()
        with (
            get_parallel().override(tp_group=SimpleNamespace(barrier=mock.Mock())),
            mock.patch.object(
                mod, "get_flags", return_value=_flags(enable_torch_compile=True)
            ),
        ):
            with self.assertRaisesRegex(AssertionError, "torch.compile"):
                mod.BreakableNPUGraphBackend(runner)


class TestDeviceGraphCls(CustomTestCase):
    def test_dispatch_follows_device(self):
        npu_graph_cls = object()
        with (
            mock.patch.object(bcg, "is_npu", return_value=True),
            mock.patch(
                "torch.npu", SimpleNamespace(NPUGraph=npu_graph_cls), create=True
            ),
        ):
            self.assertIs(bcg._device_graph_cls(), npu_graph_cls)
        with (
            mock.patch.object(bcg, "is_npu", return_value=False),
            mock.patch.object(bcg, "_is_xpu", False),
        ):
            self.assertIs(bcg._device_graph_cls(), torch.cuda.CUDAGraph)


def _make_fake_device_module():
    return SimpleNamespace(
        Stream=SimpleNamespace(wait_stream=mock.Mock()),
        current_stream=mock.Mock(return_value=mock.Mock()),
    )


def _make_fake_graph_cls(registry):
    class _FakeGraph:
        def __init__(self):
            self.capture_begin = mock.Mock()
            self.capture_end = mock.Mock()
            self.replay = mock.Mock()
            registry.append(self)

    return _FakeGraph


class TestNpuSegmentCapture(CustomTestCase):
    def test_capture_splits_at_breaks_and_replays_in_order(self):
        fake_dm = _make_fake_device_module()
        made = []
        order = []

        @bcg.eager_on_graph
        def break_one():
            order.append("break1")
            return "out1"

        @bcg.eager_on_graph
        def break_two():
            order.append("break2")
            return "out2"

        graph = bcg.BreakableCUDAGraph()
        with (
            mock.patch.object(bcg, "get_device_module", return_value=fake_dm),
            mock.patch.object(bcg, "_is_xpu", False),
            mock.patch.object(
                bcg, "_device_graph_cls", return_value=_make_fake_graph_cls(made)
            ),
        ):
            with bcg.BreakableCUDAGraphCapture(
                cuda_graph=graph, pool=(0, 0), barrier_fn=None
            ):
                order.append("seg1")
                break_one()
                order.append("seg2")
                break_two()
                order.append("seg3")

            # Capture: one graph per segment, begin/end strictly paired, and
            # NPU graphs go through the CUDA-style capture_begin signature.
            self.assertEqual([id(g) for g in made], [id(g) for g in graph._segments])
            for seg in made:
                seg.capture_begin.assert_called_once_with(
                    pool=(0, 0), capture_error_mode="global"
                )
                seg.capture_end.assert_called_once()
            self.assertEqual(order, ["seg1", "break1", "seg2", "break2", "seg3"])
            self.assertEqual(len(graph._break_fns), 2)

            # Replay: segment replay and eager breaks strictly alternate.
            order.clear()
            for i, seg in enumerate(made):
                seg.replay.side_effect = (
                    lambda i=i: order.append(f"seg{i+1}")  # noqa: B023
                )
            graph.replay()
            self.assertEqual(order, ["seg1", "break1", "seg2", "break2", "seg3"])


class TestIsStreamCapturingNpu(CustomTestCase):
    def test_uses_portable_query_on_npu(self):
        # On NPU the query must stay on the portable torch API; falling back
        # to the cuda-python path would break segment teardown on device.
        entered = []

        class _StreamCtx:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                entered.append(self.stream)
                return self

            def __exit__(self, *args):
                return False

        fake_dm = SimpleNamespace(
            stream=lambda s: _StreamCtx(s),
            is_current_stream_capturing=mock.Mock(return_value=True),
        )
        target = mock.Mock()
        with (
            mock.patch.object(bcg, "is_npu", return_value=True),
            mock.patch.object(bcg, "get_device_module", return_value=fake_dm),
        ):
            self.assertTrue(bcg._is_stream_capturing(target))

        fake_dm.is_current_stream_capturing.assert_called_once_with()
        self.assertEqual(entered, [target])


if __name__ == "__main__":
    unittest.main()
