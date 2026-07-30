"""Unit tests for DCP (Decode Context Parallelism) server args configuration.

Covers the ``--dcp-comm-backend`` field ({ag_rs, a2a, fi_a2a, vmm}), the
independent ``--dcp-query-backend`` field, and their validation in
``ServerArgs._handle_dcp_validation``:
  - a2a / fi_a2a / vmm require --dcp-size > 1
  - fi_a2a requires a CUDA platform (the authoritative MNNVL fabric probe runs
    later, at model-runner init)
  - peer Query backends require CUDA and a DSA model
  - dcp>1 requires CUDA or HIP (base behavior from the merged DCP PR)

Tests construct with safe defaults (dcp_size=1) then mutate the fields and call
``_handle_dcp_validation`` directly, so construction never trips the platform
gate; is_cuda / is_hip are patched per-test to pin the platform deterministically
(these are CPU-CI tests, where the real is_cuda() is False).
"""

import dataclasses
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

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

    def test_dcp_query_backend_field_exists(self):
        fields = {f.name for f in dataclasses.fields(ServerArgs)}
        self.assertIn("dcp_query_backend", fields)

    def test_dcp_indexer_backend_field_exists(self):
        fields = {f.name for f in dataclasses.fields(ServerArgs)}
        self.assertIn("dcp_indexer_backend", fields)

    def test_dcp_topk_backend_field_exists(self):
        fields = {f.name for f in dataclasses.fields(ServerArgs)}
        self.assertIn("dcp_topk_backend", fields)

    def test_dcp_size_default(self):
        self.assertEqual(ServerArgs.dcp_size, 1)

    def test_dcp_comm_backend_default(self):
        self.assertEqual(ServerArgs.dcp_comm_backend, "ag_rs")

    def test_dcp_query_backend_default(self):
        self.assertEqual(ServerArgs.dcp_query_backend, "allgather")

    def test_dcp_indexer_backend_default(self):
        self.assertEqual(ServerArgs.dcp_indexer_backend, "replicated")

    def test_dcp_topk_backend_default(self):
        self.assertEqual(ServerArgs.dcp_topk_backend, "allgather")


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

    def test_vmm_requires_dcp_size_gt_1(self):
        args = self._make_args(dcp_size=1, dcp_comm_backend="vmm")
        with self.assertRaises(ValueError):
            args._handle_dcp_validation()

    def test_query_direct_requires_dcp_size_gt_1(self):
        args = self._make_args(dcp_size=1, dcp_comm_backend="ag_rs")
        args.dcp_query_backend = "vmm_direct"
        with self.assertRaises(ValueError):
            args._handle_dcp_validation()

    def test_owner_sharded_indexer_requires_dcp_size_gt_1(self):
        args = self._make_args(dcp_size=1, dcp_comm_backend="ag_rs")
        args.dcp_indexer_backend = "owner_sharded"
        with self.assertRaises(ValueError):
            args._handle_dcp_validation()

    def test_topk_vmm_requires_owner_sharded_indexer(self):
        args = self._make_args(dcp_size=4, dcp_comm_backend="ag_rs")
        args.dcp_topk_backend = "vmm"
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
    @patch("sglang.srt.server_args.is_cuda", return_value=False)
    def test_vmm_on_non_cuda_raises(self, *_):
        args = self._make_args(dcp_size=2, dcp_comm_backend="vmm")
        with self.assertRaises(ValueError):
            args._handle_dcp_validation()

    @patch("sglang.srt.server_args.is_hip", return_value=False)
    @patch("sglang.srt.server_args.is_cuda", return_value=False)
    def test_query_direct_on_non_cuda_raises(self, *_):
        args = self._make_args(dcp_size=2, dcp_comm_backend="ag_rs")
        args.dcp_query_backend = "vmm_direct"
        with self.assertRaises(ValueError):
            args._handle_dcp_validation()

    @patch("sglang.srt.server_args.is_hip", return_value=False)
    @patch("sglang.srt.server_args.is_cuda", return_value=True)
    def test_vmm_rejects_non_dsa_model(self, *_):
        args = self._make_args(dcp_size=2, dcp_comm_backend="vmm")
        model_config = SimpleNamespace(
            hf_config={
                "architectures": ["DeepseekV2ForCausalLM"],
                "index_topk": None,
            }
        )
        with (
            patch.object(args, "get_model_config", return_value=model_config),
            self.assertRaises(ValueError),
        ):
            args._handle_dcp_validation()

    @patch("sglang.srt.server_args.is_hip", return_value=False)
    @patch("sglang.srt.server_args.is_cuda", return_value=True)
    def test_vmm_with_dsa_dcp_on_cuda_passes(self, *_):
        args = self._make_args(dcp_size=4, dcp_comm_backend="vmm")
        model_config = SimpleNamespace(
            hf_config={
                "architectures": ["GlmMoeDsaForCausalLM"],
                "index_topk": 2048,
            }
        )
        with patch.object(args, "get_model_config", return_value=model_config):
            args._handle_dcp_validation()
        self.assertEqual(args.dcp_comm_backend, "vmm")

    @patch("sglang.srt.server_args.is_hip", return_value=False)
    @patch("sglang.srt.server_args.is_cuda", return_value=True)
    def test_query_direct_with_dsa_dcp_on_cuda_passes(self, *_):
        args = self._make_args(dcp_size=4, dcp_comm_backend="ag_rs")
        args.dcp_query_backend = "vmm_direct"
        model_config = SimpleNamespace(
            hf_config={
                "architectures": ["GlmMoeDsaForCausalLM"],
                "index_topk": 2048,
            }
        )
        with patch.object(args, "get_model_config", return_value=model_config):
            args._handle_dcp_validation()
        self.assertEqual(args.dcp_query_backend, "vmm_direct")

    @patch("sglang.srt.server_args.is_hip", return_value=False)
    @patch("sglang.srt.server_args.is_cuda", return_value=True)
    def test_owner_sharded_indexer_with_topk_vmm_passes(self, *_):
        args = self._make_args(dcp_size=4, dcp_comm_backend="ag_rs")
        args.dcp_indexer_backend = "owner_sharded"
        args.dcp_topk_backend = "vmm"
        model_config = SimpleNamespace(
            hf_config={
                "architectures": ["GlmMoeDsaForCausalLM"],
                "index_topk": 2048,
            }
        )
        with patch.object(args, "get_model_config", return_value=model_config):
            args._handle_dcp_validation()
        self.assertEqual(args.dcp_indexer_backend, "owner_sharded")
        self.assertEqual(args.dcp_topk_backend, "vmm")

    @patch("sglang.srt.server_args.is_hip", return_value=False)
    @patch("sglang.srt.server_args.is_cuda", return_value=True)
    def test_topk_vmm_rejects_unsupported_index_topk(self, *_):
        args = self._make_args(dcp_size=4, dcp_comm_backend="ag_rs")
        args.dcp_indexer_backend = "owner_sharded"
        args.dcp_topk_backend = "vmm"
        model_config = SimpleNamespace(
            hf_config={
                "architectures": ["GlmMoeDsaForCausalLM"],
                "index_topk": 256,
            }
        )
        with (
            patch.object(args, "get_model_config", return_value=model_config),
            self.assertRaises(ValueError),
        ):
            args._handle_dcp_validation()

    def test_query_direct_conflicts_with_replicated_q_projection(self, *_):
        args = self._make_args(dcp_size=4, dcp_comm_backend="a2a")
        args.dcp_query_backend = "vmm_direct"
        args.dcp_replicate_q_proj = True
        model_config = SimpleNamespace(
            hf_config={
                "architectures": ["GlmMoeDsaForCausalLM"],
                "index_topk": 2048,
            }
        )
        with (
            patch.object(args, "get_model_config", return_value=model_config),
            self.assertRaises(ValueError),
        ):
            args._handle_dcp_validation()

    @patch("sglang.srt.server_args.is_hip", return_value=False)
    @patch("sglang.srt.server_args.is_cuda", return_value=True)
    def test_ag_rs_with_dcp_size_8_on_cuda_passes(self, *_):
        args = self._make_args(dcp_size=8, dcp_comm_backend="ag_rs")
        args._handle_dcp_validation()  # no raise
        self.assertEqual(args.dcp_size, 8)


if __name__ == "__main__":
    unittest.main()
