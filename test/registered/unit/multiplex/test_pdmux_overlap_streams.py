"""CPU-only tests for the PDMux overlapped-SM stream layout.

In overlap mode only prefill is capped by a green context; decode runs on a
plain full-device stream so its SM mask overlaps the prefill partition and also
reaches the SMs outside it. Collapsing that back to the paired green-context
call would silently re-cap decode -- a throughput regression with no error, so
the stream construction is pinned here rather than left to a live GPU run.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PDMUX_CONTEXT_PATH = REPO_ROOT / "python/sglang/srt/multiplex/pdmux_context.py"

TOTAL_SM = 132
PREFILL_CAP = 104


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


register_cpu_ci = _load_module(
    "_pdmux_overlap_ci_register", REPO_ROOT / "python/sglang/test/ci/ci_register.py"
).register_cpu_ci
register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeStream:
    """Stand-in for torch.cuda.Stream: the CI host has no CUDA device."""

    def __init__(self, device, priority=0):
        self.device = device
        self.priority = priority


class _FakeSpatial:
    """Records the (smA, smB) the green-context extension is asked for."""

    def __init__(self):
        self.calls = []

    def create_greenctx_stream_by_value(self, sm_a, sm_b, device_id):
        self.calls.append((sm_a, sm_b, device_id))
        return (f"greenctx(A={sm_a})", f"greenctx(B={sm_b})")

    def get_sm_available(self, device_id):
        return TOTAL_SM


def _install_stubs():
    spatial = _FakeSpatial()
    spatial_module = types.ModuleType("sgl_kernel.spatial")
    spatial_module.create_greenctx_stream_by_value = (
        spatial.create_greenctx_stream_by_value
    )
    spatial_module.get_sm_available = spatial.get_sm_available
    sgl_kernel = types.ModuleType("sgl_kernel")
    sgl_kernel.spatial = spatial_module
    sys.modules["sgl_kernel"] = sgl_kernel
    sys.modules["sgl_kernel.spatial"] = spatial_module
    return spatial


class PDMuxOverlapStreamTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spatial = _install_stubs()
        cls.pdmux = _load_module("_pdmux_context_overlap_test", PDMUX_CONTEXT_PATH)
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(
            Stream=_FakeStream,
            current_device=lambda: 0,
            get_device_capability=lambda device: (9, 0),
        )
        cls.pdmux.torch = fake_torch

    def setUp(self):
        self.spatial.calls.clear()

    def _write_config(self, body: str) -> str:
        path = Path(tempfile.mkdtemp()) / "pdmux.yaml"
        path.write_text(body, encoding="utf-8")
        return str(path)

    def _overlap_config(self):
        return self.pdmux.PDMuxConfig(
            sm_group_num=3,
            manual_divisions=[[PREFILL_CAP, TOTAL_SM, 0]],
            overlap_decode_full_sm=True,
        )

    def test_overlap_caps_prefill_and_leaves_decode_on_a_full_device_stream(self):
        self.pdmux.initialize_stream_groups(0, self._overlap_config())

        # The green context is cut for prefill only; its complement is the
        # decode-exclusive remainder and its stream is deliberately dropped.
        self.assertEqual(self.spatial.calls, [(PREFILL_CAP, TOTAL_SM - PREFILL_CAP, 0)])
        prefill_stream, decode_stream = self.pdmux.get_stream_groups()[1]
        self.assertEqual(prefill_stream, f"greenctx(A={PREFILL_CAP})")
        self.assertIsInstance(decode_stream, _FakeStream)
        self.assertEqual(decode_stream.priority, -1)

    def test_overlap_reports_the_full_device_as_the_decode_sm_count(self):
        # The configured decode_sm column is ignored: what decode can reach is
        # the whole device, and the startup log must not claim otherwise.
        config = self._overlap_config()
        config.manual_divisions = [[PREFILL_CAP, 8, 0]]
        self.pdmux.initialize_stream_groups(0, config)

        self.assertEqual(
            self.pdmux.get_sm_counts(),
            [(TOTAL_SM, 0), (PREFILL_CAP, TOTAL_SM), (0, TOTAL_SM)],
        )

    def test_mutually_exclusive_mode_is_unchanged(self):
        config = self.pdmux.PDMuxConfig(sm_group_num=3, manual_divisions=[[112, 20, 0]])
        self.pdmux.initialize_stream_groups(0, config)

        self.assertEqual(self.spatial.calls, [(112, 20, 0)])
        self.assertEqual(
            self.pdmux.get_stream_groups()[1], ("greenctx(A=112)", "greenctx(B=20)")
        )
        self.assertEqual(
            self.pdmux.get_sm_counts(), [(TOTAL_SM, 0), (112, 20), (0, TOTAL_SM)]
        )

    def test_overlap_rejects_a_cap_that_leaves_decode_no_exclusive_sms(self):
        for prefill_sm in (TOTAL_SM, 0):
            with self.subTest(prefill_sm=prefill_sm):
                config = self.pdmux.PDMuxConfig(
                    sm_group_num=3,
                    manual_divisions=[[prefill_sm, TOTAL_SM, 0]],
                    overlap_decode_full_sm=True,
                )
                with self.assertRaisesRegex(ValueError, "strictly inside"):
                    self.pdmux.initialize_stream_groups(0, config)

    def test_overlap_requires_manual_divisions(self):
        # divide_sm only enumerates mutually exclusive splits (prefill >= 50%),
        # which does not describe an overlapped layout.
        path = self._write_config("sm_group_num: 3\noverlap_decode_full_sm: true\n")
        with self.assertRaisesRegex(ValueError, "manual_divisions"):
            self.pdmux.load_pdmux_config(path)

    def test_overlap_flag_defaults_off(self):
        path = self._write_config("sm_group_num: 3\nmanual_divisions: [[112, 20, 0]]\n")
        self.assertFalse(self.pdmux.load_pdmux_config(path).overlap_decode_full_sm)


if __name__ == "__main__":
    unittest.main()
