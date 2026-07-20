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

from sglang.srt.arg_groups.speculative_hook import _validate_dcp_spec_topk
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


class TestDCPSpecTopkGuard(CustomTestCase):
    """DCP + speculative decoding permits only chain drafts (topk == 1).

    The guard is ``arg_groups/speculative_hook._validate_dcp_spec_topk``,
    invoked at the end of ``handle_speculative_decoding`` after the
    per-algorithm handler has resolved ``speculative_eagle_topk``
    (auto-choose -> int, DFLASH forced to 1). Tested directly, mirroring the
    ``_handle_dcp_validation`` pattern above.
    """

    @staticmethod
    def _make_args(dcp_size, algorithm, topk):
        args = ServerArgs(model_path="dummy")
        args.dcp_size = dcp_size
        args.speculative_algorithm = algorithm
        args.speculative_eagle_topk = topk
        return args

    def test_eagle3_topk2_with_dcp_raises(self):
        args = self._make_args(8, "EAGLE3", 2)
        with self.assertRaisesRegex(ValueError, "speculative-eagle-topk"):
            _validate_dcp_spec_topk(args)

    def test_eagle3_topk1_with_dcp_passes(self):
        _validate_dcp_spec_topk(self._make_args(8, "EAGLE3", 1))  # no raise

    def test_dflash_with_dcp_passes(self):
        # _handle_dflash forces topk == 1 before the guard runs.
        _validate_dcp_spec_topk(self._make_args(8, "DFLASH", 1))  # no raise

    def test_mtp_nextn_chain_with_dcp_passes(self):
        # NEXTN resolves to EAGLE; MTP drafts run the EAGLE path.
        _validate_dcp_spec_topk(self._make_args(8, "EAGLE", 1))  # no raise

    def test_topk_none_with_dcp_passes(self):
        # Direct-call case for an unresolved topk. (In production, NGRAM
        # resolves topk from --speculative-ngram-max-bfs-breadth (default 10),
        # so real NGRAM+DCP trips the guard unless breadth is set to 1.)
        _validate_dcp_spec_topk(self._make_args(8, "STANDALONE", None))  # no raise

    def test_ngram_breadth_topk_with_dcp_raises_with_ngram_hint(self):
        # NGRAM's handler overwrites topk with the bfs breadth; the error must
        # name the actionable knob for that path.
        args = self._make_args(8, "NGRAM", 10)
        with self.assertRaisesRegex(ValueError, "ngram-max-bfs-breadth"):
            _validate_dcp_spec_topk(args)

    def test_duck_typed_args_without_dcp_size_pass(self):
        # handle_speculative_decoding is exercised by registry unit tests with
        # SimpleNamespace mocks that carry no dcp_size; the guard must not
        # assume the attribute exists.
        from types import SimpleNamespace

        _validate_dcp_spec_topk(
            SimpleNamespace(speculative_algorithm="EAGLE")
        )  # no raise

    def test_topk2_without_dcp_passes(self):
        _validate_dcp_spec_topk(self._make_args(1, "EAGLE3", 2))  # no raise

    def test_no_spec_algorithm_with_dcp_passes(self):
        _validate_dcp_spec_topk(self._make_args(8, None, None))  # no raise


if __name__ == "__main__":
    unittest.main()
