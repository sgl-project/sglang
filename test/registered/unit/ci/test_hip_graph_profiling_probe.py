"""Unit tests for scripts/ci/amd/check_hip_graph_profiling.py — no GPU, no server.

The probe decides whether a ROCm image's torch.profiler records kernels launched
by HIP graph replay. Its verdict logic is what makes a red run actionable, so it
is pinned here: a trace holding no device events must never read as a pass, and a
wedged graph launch must be reported as the HIP deadlock it is rather than as a
missing-kernel gap.
"""

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PROBE_PATH = (
    Path(__file__).resolve().parents[4] / "scripts/ci/amd/check_hip_graph_profiling.py"
)

ROCM_LIBS = [
    "/opt/rocm-7.2.4/lib/libamdhip64.so.7.2.70204",
    "/opt/rocm-7.2.4/lib/libroctracer64.so.4.1.70204",
]
# What a rocm724 image maps by default: the wheel's own 7.2.0 copies.
VENDORED_LIBS = [
    "/opt/venv/lib/python3.12/site-packages/torch/lib/libamdhip64.so",
    "/opt/venv/lib/python3.12/site-packages/torch/lib/libroctracer64.so",
]
ENV_OK = {
    "status": "ok",
    "device_count": 8,
    "torch": "2.11.0+rocm7.2",
    "torch_hip": "7.2.41134",
    "rocm_version": "7.2.4",
    "device_name": "AMD Instinct MI355X",
    "gcn_arch": "gfx950",
    "libs": ROCM_LIBS,
}
TRACED = {
    "status": "ok",
    "kernels": 512,
    "expected_kernels": 512,
    "self_device_time_us": 1234.5,
    "trace_events": 9000,
    "libs": ROCM_LIBS,
}
# The measured ROCm 7.2.0 result: most kernels traced, some silently dropped.
PARTIAL = dict(TRACED, kernels=448, self_device_time_us=1000.0)
UNTRACED = dict(TRACED, kernels=0, self_device_time_us=0.0)
TIMED_OUT = {"status": "timeout", "output": "(no output before the kill)"}
CRASHED = {"status": "error", "output": "Segmentation fault"}


def _load_probe():
    """Fresh module per test: the verdict tests monkeypatch its run_child."""
    spec = importlib.util.spec_from_file_location("hip_graph_probe", PROBE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDeviceEventCounting(CustomTestCase):
    def test_counts_device_categories_case_insensitively(self):
        probe = _load_probe()
        events = [
            {"ph": "X", "cat": "kernel", "name": "gemm"},
            {"ph": "X", "cat": "Kernel", "name": "gemm"},
            {"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy DtoD"},
            {"ph": "X", "cat": "gpu_memset", "name": "Memset"},
        ]
        self.assertEqual(
            probe.count_device_events(events),
            {"kernel": 2, "gpu_memcpy": 1, "gpu_memset": 1},
        )

    def test_host_side_events_are_not_device_events(self):
        # A trace of a broken graph launch is full of these and nothing else,
        # which is exactly the failure the probe has to catch.
        probe = _load_probe()
        events = [
            {"ph": "X", "cat": "cpu_op", "name": "aten::mm"},
            {"ph": "X", "cat": "cuda_runtime", "name": "hipGraphLaunch"},
            {"ph": "X", "cat": "ac2g", "name": "flow"},
            {"ph": "M", "name": "process_name"},
            {"name": "no category at all"},
        ]
        self.assertEqual(probe.count_device_events(events), {})


class TestVerdict(CustomTestCase):
    def _verdict(self, env=ENV_OK, eager=TRACED, graph=TRACED):
        probe = _load_probe()
        phases = {"env": env, "eager": eager, "graph": graph}
        probe.run_child = lambda phase, args, p=phases: p[phase]
        out = io.StringIO()
        with mock.patch.object(sys, "argv", ["check_hip_graph_profiling"]):
            with contextlib.redirect_stdout(out):
                code = probe.main()
        return code, out.getvalue()

    def test_passes_when_graph_replay_kernels_reach_the_trace(self):
        code, out = self._verdict()
        self.assertEqual(code, 0)
        self.assertIn("VERDICT: PASS", out)

    def test_untraced_graph_replay_cites_the_roctracer_bug(self):
        code, out = self._verdict(graph=UNTRACED)
        self.assertEqual(code, 1)
        self.assertIn("VERDICT: FAIL", out)
        self.assertIn("6102", out)

    def test_partial_graph_loss_fails_and_reports_the_shortfall(self):
        # The whole reason the check counts dispatches: 448 of 512 kernels is the
        # broken runtime, and "some kernels arrived" must not read as healthy.
        code, out = self._verdict(graph=PARTIAL)
        self.assertEqual(code, 1)
        self.assertIn("64 of 512", out)
        self.assertIn("6102", out)

    def test_untraced_eager_launches_invalidate_the_graph_result(self):
        code, out = self._verdict(eager=UNTRACED, graph=UNTRACED)
        self.assertEqual(code, 1)
        self.assertIn("cannot trace GPU work reliably at all", out)

    def test_wedged_graph_replay_is_reported_as_a_deadlock(self):
        code, out = self._verdict(graph=TIMED_OUT)
        self.assertEqual(code, 1)
        self.assertIn("hipGraphLaunch", out)

    def test_crashed_graph_phase_is_not_reported_as_missing_kernels(self):
        code, out = self._verdict(graph=CRASHED)
        self.assertEqual(code, 1)
        self.assertIn("did not complete", out)
        self.assertNotIn("6102", out)

    def test_unusable_environment_stops_before_probing(self):
        code, out = self._verdict(env={"status": "error", "error": "no GPU visible"})
        self.assertEqual(code, 1)
        self.assertIn("environment: unusable", out)

    def test_missing_marketing_name_is_flagged(self):
        # An image without libdrm-amdgpu resolves no device name, which silently
        # disables every device-name-keyed tuned config.
        code, out = self._verdict(env=dict(ENV_OK, device_name=""))
        self.assertEqual(code, 0)
        self.assertIn("libdrm-amdgpu-common", out)

    def test_vendored_runtime_is_named_when_graph_replay_is_lossy(self):
        # A rocm724 image failing this way has the fix installed but is not
        # loading it, so the remedy is the preload, not a different image.
        code, out = self._verdict(
            env=dict(ENV_OK, libs=VENDORED_LIBS),
            eager=dict(TRACED, libs=VENDORED_LIBS),
            graph=dict(PARTIAL, libs=VENDORED_LIBS),
        )
        self.assertEqual(code, 1)
        self.assertIn("torch is using its own HIP/roctracer", out)
        self.assertIn("torch/lib/libamdhip64.so", out)
        self.assertNotIn("--disable-cuda-graph", out)

    def test_rocm_install_libs_are_not_reported_as_vendored(self):
        code, out = self._verdict()
        self.assertEqual(code, 0)
        self.assertNotIn("torch is using its own", out)


class TestVendoredRuntimeDetection(CustomTestCase):
    def test_wheel_copies_are_flagged_when_no_rocm_copy_is_mapped(self):
        probe = _load_probe()
        libs = VENDORED_LIBS + ["/opt/venv/.../torch/lib/libtorch_hip.so"]
        self.assertEqual(probe.vendored_runtime_libs(libs), VENDORED_LIBS)

    def test_nothing_flagged_when_the_rocm_install_is_in_use(self):
        probe = _load_probe()
        self.assertEqual(probe.vendored_runtime_libs(ROCM_LIBS), [])

    def test_a_preload_maps_both_copies_and_is_not_a_finding(self):
        # LD_PRELOAD leaves the wheel's copies mapped too; the preloaded ones
        # interpose, so reporting them would contradict the PASS they produced.
        probe = _load_probe()
        self.assertEqual(probe.vendored_runtime_libs(ROCM_LIBS + VENDORED_LIBS), [])

    def test_a_partial_preload_still_flags_the_library_left_behind(self):
        probe = _load_probe()
        libs = [ROCM_LIBS[0]] + VENDORED_LIBS
        self.assertEqual(probe.vendored_runtime_libs(libs), [VENDORED_LIBS[1]])

    def test_the_rocm_library_copied_over_the_wheel_path_is_not_a_finding(self):
        # What the image build leaves behind: the ROCm build sitting at the
        # wheel's path. Reporting it would tell the reader to redo a fix that is
        # already in place, so the size decides rather than the location.
        probe = _load_probe()
        with tempfile.TemporaryDirectory() as tmp:
            rocm_dir = Path(tmp) / "rocm"
            torch_dir = Path(tmp) / "torch-lib"
            rocm_dir.mkdir()
            torch_dir.mkdir()
            mapped = []
            for name, payload in (
                ("libamdhip64", b"rocm-hip-build"),
                ("libroctracer64", b"rocm-tracer"),
            ):
                (rocm_dir / f"{name}.so.7.2.70204").write_bytes(payload)
                vendored = torch_dir / f"{name}.so"
                vendored.write_bytes(payload)
                mapped.append(str(vendored))
            self.assertEqual(
                probe.vendored_runtime_libs(mapped, str(rocm_dir)),
                [],
            )

    def test_the_wheel_build_at_the_wheel_path_is_still_a_finding(self):
        probe = _load_probe()
        with tempfile.TemporaryDirectory() as tmp:
            rocm_dir = Path(tmp) / "rocm"
            torch_dir = Path(tmp) / "torch-lib"
            rocm_dir.mkdir()
            torch_dir.mkdir()
            (rocm_dir / "libamdhip64.so.7.2.70204").write_bytes(b"rocm-hip-build")
            vendored = torch_dir / "libamdhip64.so"
            vendored.write_bytes(b"a wheel copy of a different size")
            self.assertEqual(
                probe.vendored_runtime_libs([str(vendored)], str(rocm_dir)),
                [str(vendored)],
            )


class TestChildCommand(CustomTestCase):
    def test_workload_options_reach_the_phase_children(self):
        # --graph-nodes was accepted by the parent and dropped here, so a
        # calibration run at 4 nodes silently measured the default 64.
        probe = _load_probe()
        args = probe.build_parser().parse_args(
            ["--graph-nodes", "4", "--replays", "2", "--matmul-size", "256"]
        )
        cmd = probe.child_command("graph", args)
        self.assertEqual(
            cmd[-8:],
            [
                "--phase",
                "graph",
                "--replays",
                "2",
                "--matmul-size",
                "256",
                "--graph-nodes",
                "4",
            ],
        )

    def test_every_option_is_either_forwarded_or_parent_only(self):
        # The guard that keeps the bug above from coming back with the next knob.
        probe = _load_probe()
        options = {
            action.dest
            for action in probe.build_parser()._actions
            if action.dest != "help"
        }
        accounted = set(probe.PHASE_OPTIONS) | set(probe.PARENT_ONLY_OPTIONS)
        self.assertEqual(options - accounted, set())


class TestPreloadValue(CustomTestCase):
    def test_names_the_real_files_and_skips_the_symlinks(self):
        # Preloading the unversioned symlink is not what was measured, and the
        # versioned name is what pins the ROCm patch level in the value.
        probe = _load_probe()
        with tempfile.TemporaryDirectory() as lib_dir:
            hip = Path(lib_dir) / "libamdhip64.so.7.2.70204"
            tracer = Path(lib_dir) / "libroctracer64.so.4.1.70204"
            hip.touch()
            tracer.touch()
            (Path(lib_dir) / "libamdhip64.so.7").symlink_to(hip)
            self.assertEqual(probe.rocm_runtime_preload(lib_dir), f"{hip}:{tracer}")

    def test_no_value_when_the_image_has_no_rocm_runtime(self):
        probe = _load_probe()
        with tempfile.TemporaryDirectory() as lib_dir:
            self.assertIsNone(probe.rocm_runtime_preload(lib_dir))


if __name__ == "__main__":
    unittest.main()
