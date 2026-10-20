"""Capture status dispatch and stream dependencies at BCG segment boundaries."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    breakable_cuda_graph as bcg,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCaptureStatus(unittest.TestCase):
    def test_device_api_checks_requested_stream_and_restores_current_stream(self):
        # The target intentionally has no cuda_stream, matching an NPU stream.
        main, side = object(), object()
        current = [main]

        @contextlib.contextmanager
        def stream_context(stream):
            previous = current[0]
            current[0] = stream
            try:
                yield
            finally:
                current[0] = previous

        for npu, xpu, hip in (
            (True, False, False),
            (False, True, False),
            (False, False, True),
        ):
            for capturing in (True, False):
                with self.subTest(npu=npu, xpu=xpu, hip=hip, capturing=capturing):
                    device = SimpleNamespace(
                        stream=stream_context,
                        is_current_stream_capturing=Mock(
                            side_effect=lambda: (
                                capturing if current[0] is side else False
                            )
                        ),
                    )
                    with (
                        patch.object(bcg, "_is_npu", npu),
                        patch.object(bcg, "_is_xpu", xpu),
                        patch.object(bcg, "is_hip", return_value=hip),
                        patch.object(bcg, "get_device_module", return_value=device),
                        patch.object(bcg, "_capture_status") as cuda_status,
                    ):
                        self.assertEqual(bcg._is_stream_capturing(side), capturing)
                        self.assertIs(current[0], main)
                        cuda_status.assert_not_called()
                        device.is_current_stream_capturing.assert_called_once_with()

    def test_cuda_uses_runtime_status(self):
        active, inactive = object(), object()
        runtime = SimpleNamespace(
            cudaStreamCaptureStatus=SimpleNamespace(
                cudaStreamCaptureStatusActive=active
            )
        )
        stream = SimpleNamespace(cuda_stream=42)
        with (
            patch.object(bcg, "_is_npu", False),
            patch.object(bcg, "_is_xpu", False),
            patch.object(bcg, "is_hip", return_value=False),
            patch.object(bcg, "rt", runtime),
            patch.object(bcg, "get_device_module") as device,
            patch.object(bcg, "_capture_status") as status,
        ):
            for actual, expected in ((active, True), (inactive, False)):
                status.return_value = actual
                self.assertEqual(bcg._is_stream_capturing(stream), expected)
                status.assert_called_with(42)
            device.assert_not_called()


class TestStreamDependencies(unittest.TestCase):
    def setUp(self):
        self.main = SimpleNamespace(stream_id=1)
        self.side = type("SideStream", (), {"stream_id": 2})()
        self.forked = set()
        self.main_token = bcg._current_stream_var.set(self.main)
        self.forked_token = bcg._forked_streams_var.set(self.forked)
        self.addCleanup(bcg._current_stream_var.reset, self.main_token)
        self.addCleanup(bcg._forked_streams_var.reset, self.forked_token)

    def test_explicit_join_removes_side_stream_from_current_segment(self):
        with (
            patch.object(bcg, "_original_wait_stream", Mock()) as wait,
            patch.object(bcg, "_is_stream_capturing", return_value=True),
        ):
            bcg._hooked_wait_stream(self.side, self.main)
            self.assertEqual(self.forked, {self.side})
            bcg._hooked_wait_stream(self.main, self.side)
            self.assertEqual(self.forked, set())
            self.assertEqual(wait.call_args_list[0].args, (self.side, self.main))
            self.assertEqual(wait.call_args_list[1].args, (self.main, self.side))

    def test_join_from_previous_segment_is_skipped(self):
        # Auto-join has already ended the side's capture before an eager break.
        with (
            patch.object(bcg, "_original_wait_stream", Mock()) as wait,
            patch.object(bcg, "_is_stream_capturing", return_value=False),
        ):
            bcg._hooked_wait_stream(self.main, self.side)
            wait.assert_not_called()

    def test_segment_end_joins_active_forks_before_capture_end(self):
        self.forked.add(self.side)
        events = []
        graph = SimpleNamespace(capture_end=lambda: events.append("capture_end"))
        graphs = bcg.BreakableCUDAGraph()
        capture = bcg.BreakableCUDAGraphCapture(graphs)
        capture._current_graph = graph
        with (
            patch.object(
                bcg,
                "_original_wait_stream",
                lambda main, side: events.append((main, side)),
            ),
            patch.object(bcg, "_is_stream_capturing", return_value=True),
        ):
            capture._end_current_segment()
        self.assertEqual(events, [(self.main, self.side), "capture_end"])
        self.assertEqual(self.forked, set())
        self.assertEqual(graphs._segments, [graph])


if __name__ == "__main__":
    unittest.main()
