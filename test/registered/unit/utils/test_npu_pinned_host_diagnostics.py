import os
import sys
import tempfile
import unittest
from pathlib import Path
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

    def test_cgroup_parent_limit_can_be_the_effective_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / "worker"
            child.mkdir()
            (root / "memory.current").write_text("800\n")
            (root / "memory.max").write_text("1000\n")
            (child / "memory.current").write_text("100\n")
            (child / "memory.max").write_text("500\n")

            with mock.patch.object(
                diagnostics, "_cgroup_v2_directory", return_value=child
            ):
                current, limit, limiting_group = diagnostics._cgroup_v2_memory(root)

        self.assertEqual((current, limit, limiting_group), (800, 1000, root))

    def test_monitor_compares_loading_and_runtime_allocator_peaks(self):
        loading = {
            "pinned_active_peak_bytes": 100,
            "pinned_reserved_peak_bytes": 200,
        }
        running = {
            "pinned_active_peak_bytes": 150,
            "pinned_reserved_peak_bytes": 200,
        }
        with (
            mock.patch.dict(os.environ, {"SGLANG_NPU_PINNED_HOST_DEBUG": "1"}),
            mock.patch.object(
                diagnostics, "_memory_snapshot", side_effect=[loading, running]
            ),
            mock.patch.object(diagnostics.logger, "warning") as log_monitor,
        ):
            monitor = diagnostics.PinnedHostMemoryMonitor(enabled=True)
            monitor.mark_runtime()

        record = log_monitor.call_args.args[1]
        self.assertEqual(record["event"], "runtime_start")
        self.assertEqual(record["loading_pinned_active_peak_bytes"], 100)
        self.assertEqual(record["active_peak_phase"], "runtime")
        self.assertEqual(record["reserved_peak_phase"], "loading_or_equal")

    def test_host_and_cgroup_usage_percentages(self):
        with (
            mock.patch.object(
                diagnostics,
                "_meminfo_bytes",
                side_effect=lambda key: {"MemTotal": 1000, "MemAvailable": 250}[key],
            ),
            mock.patch.object(
                diagnostics,
                "_cgroup_v2_memory",
                return_value=(400, 500, Path("/sys/fs/cgroup/test")),
            ),
            mock.patch.object(diagnostics, "_read_int", return_value=None),
        ):
            snapshot = diagnostics._host_and_cgroup_snapshot()

        self.assertEqual(snapshot["host_memory_used_pct"], 75.0)
        self.assertEqual(snapshot["cgroup_memory_used_pct"], 80.0)
        self.assertEqual(snapshot["cgroup_memory_headroom_bytes"], 100)


if __name__ == "__main__":
    unittest.main()
