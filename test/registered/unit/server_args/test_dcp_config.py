"""Unit tests for DCP (Decode Context Parallelism) server args configuration."""

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
    """Verify --dcp-comm-backend validation logic.

    ServerArgs.__post_init__ skips validation for model_path="dummy",
    so we call _handle_context_parallelism directly.
    """

    def _make_args(self, **kwargs):
        defaults = dict(model_path="dummy", dcp_size=1, dcp_comm_backend="ag_rs")
        defaults.update(kwargs)
        return ServerArgs(**defaults)

    def test_a2a_requires_dcp_size_gt_1(self):
        args = self._make_args(dcp_size=1, dcp_comm_backend="a2a")
        with self.assertRaises(ValueError):
            args._handle_context_parallelism()

    def test_a2a_with_dcp_size_2_passes(self):
        args = self._make_args(dcp_size=2, dcp_comm_backend="a2a")
        args._handle_context_parallelism()
        self.assertEqual(args.dcp_comm_backend, "a2a")

    def test_ag_rs_with_dcp_size_1(self):
        args = self._make_args(dcp_size=1, dcp_comm_backend="ag_rs")
        self.assertEqual(args.dcp_comm_backend, "ag_rs")

    def test_ag_rs_with_dcp_size_8(self):
        args = self._make_args(dcp_size=8, dcp_comm_backend="ag_rs")
        self.assertEqual(args.dcp_size, 8)


if __name__ == "__main__":
    unittest.main()
