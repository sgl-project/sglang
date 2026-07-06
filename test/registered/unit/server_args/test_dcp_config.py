"""Unit tests for DCP (Decode Context Parallelism) server args configuration.

Covers the ``--dcp-comm-backend`` field ({ag_rs, a2a, fi_a2a}) and its
validation in ``ServerArgs._handle_dcp_validation``:
  - a2a / fi_a2a require --dcp-size > 1
  - fi_a2a requires a CUDA platform (the authoritative MNNVL fabric probe runs
    later, at model-runner init)
  - dcp>1 requires CUDA or HIP (base behavior from the merged DCP PR)

Tests construct with safe defaults (dcp_size=1) then mutate the fields and call
``_handle_dcp_validation`` directly, so construction never trips the platform
gate; is_cuda / is_hip are patched per-test to pin the platform deterministically
(these are CPU-CI tests, where the real is_cuda() is False).
"""

import dataclasses
import unittest
from unittest.mock import patch

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

_mock_device = patch("sglang.srt.server_args.get_device", return_value="cuda")
_mock_device.start()


class TestDCPFieldDefaults(CustomTestCase):
    """Verify DCP-related dataclass fields exist with correct defaults."""

    def test_dcp_size_field_exists(self):
        fields = {f.name for f in dataclasses.fields(ServerArgs)}
        self.assertIn("dcp_size", fields)

    def test_dcp_comm_backend_field_exists(self):
        fields = {f.name for f in dataclasses.fields(ServerArgs)}
        self.assertIn("dcp_comm_backend", fields)

    def test_dcp_size_default(self):
        self.assertEqual(ServerArgs.dcp_size, 1)

    def test_dcp_comm_backend_default(self):
        self.assertEqual(ServerArgs.dcp_comm_backend, "ag_rs")


class TestDCPCommBackendValidation(CustomTestCase):
    """Verify ``_handle_dcp_validation`` accepts/rejects the right combos."""

    @staticmethod
    def _make_args(dcp_size, dcp_comm_backend):
        # Construct with safe defaults (dcp_size=1) so __post_init__ never trips
        # the dcp>1 platform gate, then set the fields under test.
        args = ServerArgs(model_path="dummy")
        args.dcp_size = dcp_size
        args.dcp_comm_backend = dcp_comm_backend
        return args

    def test_a2a_requires_dcp_size_gt_1(self):
        args = self._make_args(dcp_size=1, dcp_comm_backend="a2a")
        with self.assertRaises(ValueError):
            args._handle_dcp_validation()

    def test_fi_a2a_requires_dcp_size_gt_1(self):
        args = self._make_args(dcp_size=1, dcp_comm_backend="fi_a2a")
        with self.assertRaises(ValueError):
            args._handle_dcp_validation()

    @patch("sglang.srt.server_args.is_hip", return_value=False)
    @patch("sglang.srt.server_args.is_cuda", return_value=True)
    def test_a2a_with_dcp_size_2_on_cuda_passes(self, *_):
        args = self._make_args(dcp_size=2, dcp_comm_backend="a2a")
        args._handle_dcp_validation()  # no raise
        self.assertEqual(args.dcp_comm_backend, "a2a")

    @patch("sglang.srt.server_args.is_hip", return_value=False)
    @patch("sglang.srt.server_args.is_cuda", return_value=True)
    def test_fi_a2a_with_dcp_size_2_on_cuda_passes_server_args(self, *_):
        # server_args accepts fi_a2a on CUDA; the MNNVL fabric probe is deferred
        # to model-runner init (init_fi_a2a_workspace).
        args = self._make_args(dcp_size=2, dcp_comm_backend="fi_a2a")
        args._handle_dcp_validation()  # no raise
        self.assertEqual(args.dcp_comm_backend, "fi_a2a")

    @patch("sglang.srt.server_args.is_hip", return_value=False)
    @patch("sglang.srt.server_args.is_cuda", return_value=False)
    def test_fi_a2a_on_non_cuda_raises(self, *_):
        args = self._make_args(dcp_size=2, dcp_comm_backend="fi_a2a")
        with self.assertRaises(ValueError):
            args._handle_dcp_validation()

    @patch("sglang.srt.server_args.is_hip", return_value=False)
    @patch("sglang.srt.server_args.is_cuda", return_value=True)
    def test_ag_rs_with_dcp_size_8_on_cuda_passes(self, *_):
        args = self._make_args(dcp_size=8, dcp_comm_backend="ag_rs")
        args._handle_dcp_validation()  # no raise
        self.assertEqual(args.dcp_size, 8)


if __name__ == "__main__":
    unittest.main()
