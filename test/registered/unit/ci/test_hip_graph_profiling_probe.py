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
import unittest
from pathlib import Path
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PROBE_PATH = (
    Path(__file__).resolve().parents[4] / "scripts/ci/amd/check_hip_graph_profiling.py"
)

ENV_OK = {
    "status": "ok",
    "device_count": 8,
    "torch": "2.9.1+rocm7.2.4",
    "torch_hip": "7.2.41134",
    "rocm_version": "7.2.4",
    "device_name": "AMD Instinct MI355X",
    "gcn_arch": "gfx950",
    "libs": ["libamdhip64.so.7.2.70204"],
}
TRACED = {
    "status": "ok",
    "kernels": 32,
    "self_device_time_us": 1234.5,
    "trace_events": 900,
    "libs": ["libroctracer64.so.4.1.70204"],
}
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

    def test_untraced_eager_launches_invalidate_the_graph_result(self):
        code, out = self._verdict(eager=UNTRACED, graph=UNTRACED)
        self.assertEqual(code, 1)
        self.assertIn("cannot trace GPU work at all", out)

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


if __name__ == "__main__":
    unittest.main()
