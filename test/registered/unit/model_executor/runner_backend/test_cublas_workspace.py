"""CPU-only ordering/guard tests; GPU ownership/replay needs separate tests."""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
)
from sglang.srt.model_executor.runner_backend_utils.cublas_workspace import (
    prepare_cublas_for_graph_capture,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

BCG = (
    "sglang.srt.model_executor.runner_backend_utils."
    "breakable_cuda_graph.breakable_cuda_graph"
)


class TestCublasWorkspace(unittest.TestCase):
    def test_sync_precedes_clear(self):
        events = []
        device = SimpleNamespace(synchronize=lambda: events.append("sync"))
        with (
            mock.patch("torch.version.cuda", "13.0"),
            mock.patch(
                "torch._C._cuda_clearCublasWorkspaces",
                create=True,
                side_effect=lambda: events.append("clear"),
            ),
        ):
            prepare_cublas_for_graph_capture(device)
        self.assertEqual(events, ["sync", "clear"])

    def test_non_cuda_noop(self):
        device = SimpleNamespace(synchronize=mock.Mock())
        with (
            mock.patch("torch.version.cuda", None),
            mock.patch(
                "torch._C._cuda_clearCublasWorkspaces",
                create=True,
            ) as clear,
        ):
            prepare_cublas_for_graph_capture(device)
        device.synchronize.assert_not_called()
        clear.assert_not_called()

    def test_each_bcg_segment_prepares_before_capture(self):
        events = []
        device = SimpleNamespace(synchronize=lambda: events.append("sync"))
        graph = SimpleNamespace(capture_begin=lambda **kw: events.append("begin"))
        capture = BreakableCUDAGraphCapture(BreakableCUDAGraph())
        with (
            mock.patch(BCG + "._is_xpu", False),
            mock.patch(BCG + ".get_device_module", return_value=device),
            mock.patch("torch.version.cuda", "13.0"),
            mock.patch("torch.cuda.CUDAGraph", return_value=graph),
            mock.patch(
                "torch._C._cuda_clearCublasWorkspaces",
                create=True,
                side_effect=lambda: events.append("clear"),
            ),
        ):
            capture._begin_new_segment()
            events.append("eager-break")
            capture._begin_new_segment()
        self.assertEqual(
            events, ["sync", "clear", "begin", "eager-break", "sync", "clear", "begin"]
        )

    def test_xpu_segment_does_not_prepare_cuda_workspace(self):
        graph = mock.Mock()
        capture = BreakableCUDAGraphCapture(BreakableCUDAGraph())
        with (
            mock.patch(BCG + "._is_xpu", True),
            mock.patch(BCG + ".prepare_cublas_for_graph_capture") as prepare,
            mock.patch("torch.xpu.XPUGraph", return_value=graph, create=True),
        ):
            capture._begin_new_segment()
        prepare.assert_not_called()
        graph.capture_begin.assert_called_once()


if __name__ == "__main__":
    unittest.main()
