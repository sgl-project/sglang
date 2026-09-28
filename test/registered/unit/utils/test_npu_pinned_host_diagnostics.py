import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.utils import npu_pinned_host_diagnostics as diagnostics


class TestNpuPinnedHostDiagnostics(unittest.TestCase):
    def setUp(self):
        diagnostics._last_log_time.clear()
        diagnostics._call_counts.clear()

    def test_disabled_does_not_read_memory_state(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(diagnostics, "_memory_snapshot") as snapshot,
        ):
            with diagnostics.trace_npu_pinned_host_allocation("test"):
                pass
            snapshot.assert_not_called()

    def test_failure_logs_allocator_state_and_preserves_error(self):
        fake_torch_npu = SimpleNamespace(
            npu=SimpleNamespace(
                host_memory_stats=lambda: {
                    "active_bytes.current": 10,
                    "allocated_bytes.current": 20,
                }
            )
        )
        with (
            mock.patch.dict(os.environ, {"SGLANG_NPU_PINNED_HOST_DEBUG": "1"}),
            mock.patch.dict(sys.modules, {"torch_npu": fake_torch_npu}),
            mock.patch.object(diagnostics.logger, "exception") as log_failure,
        ):
            with self.assertRaisesRegex(RuntimeError, "injected allocation failure"):
                with diagnostics.trace_npu_pinned_host_allocation(
                    "test", requested_bytes=32, details={"shape": (4, 8)}
                ):
                    raise RuntimeError("injected allocation failure")

        context = log_failure.call_args.args[1]
        self.assertEqual(context["site"], "test")
        self.assertEqual(context["requested_bytes"], 32)
        self.assertEqual(context["shape"], (4, 8))
        self.assertEqual(context["pinned_active_bytes"], 10)
        self.assertEqual(context["pinned_reserved_bytes"], 20)

    def test_small_allocations_are_sampled(self):
        with (
            mock.patch.dict(os.environ, {"SGLANG_NPU_PINNED_HOST_DEBUG": "1"}),
            mock.patch.object(diagnostics, "_memory_snapshot", return_value={}),
            mock.patch.object(
                diagnostics.time, "monotonic", side_effect=[0, 0.1, 1, 31, 31.1]
            ),
            mock.patch.object(diagnostics.logger, "warning") as log_sample,
        ):
            for _ in range(3):
                with diagnostics.trace_npu_pinned_host_allocation(
                    "test", requested_bytes=8
                ):
                    pass

        self.assertEqual(log_sample.call_count, 4)
        self.assertEqual(diagnostics._call_counts["test"], 3)

    def test_large_allocations_are_always_logged(self):
        with (
            mock.patch.dict(os.environ, {"SGLANG_NPU_PINNED_HOST_DEBUG": "1"}),
            mock.patch.object(diagnostics, "_memory_snapshot", return_value={}),
            mock.patch.object(diagnostics.time, "monotonic", return_value=1),
            mock.patch.object(diagnostics.logger, "warning") as log_sample,
        ):
            for _ in range(2):
                with diagnostics.trace_npu_pinned_host_allocation(
                    "test", requested_bytes=1 << 30
                ):
                    pass

        self.assertEqual(log_sample.call_count, 4)


if __name__ == "__main__":
    unittest.main()
