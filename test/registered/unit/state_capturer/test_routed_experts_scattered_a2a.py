"""DeepEP-family backend recognition in RoutedExpertsCapturer."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.state_capturer import routed_experts as re_mod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestScatteredA2ABackendHelper(CustomTestCase):
    def test_classification(self):
        expected = {
            "deepep": True,
            "deepep_v2": True,
            "none": False,
            "mooncake": False,
        }
        for value, exp in expected.items():
            with mock.patch.object(
                re_mod, "get_moe_a2a_backend", return_value=MoeA2ABackend(value)
            ):
                self.assertEqual(
                    re_mod._is_scattered_a2a_backend(), exp, f"backend={value}"
                )


class TestGetLocalSliceBackendBranch(CustomTestCase):
    T, L, K = 16, 3, 4

    def setUp(self):
        super().setUp()
        patches = (
            mock.patch.object(
                moe_utils,
                "get_moe_a2a_backend",
                side_effect=lambda: re_mod.get_moe_a2a_backend(),
            ),
            mock.patch.object(
                moe_utils,
                "should_use_flashinfer_cutlass_moe_fp4_allgather",
                return_value=False,
            ),
            mock.patch.object(
                moe_utils, "get_parallel", return_value=SimpleNamespace(dwdp_size=1)
            ),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _capturer(self):
        cap = object.__new__(re_mod.RoutedExpertsCapturer)
        buf = torch.arange(self.T * self.L * self.K, dtype=torch.int32).reshape(
            self.T, self.L, self.K
        )
        cap.device_cache = SimpleNamespace(buffer=buf)
        cap.topk_size = self.K
        return cap, buf

    def _slice(self, cap, n_local):
        fb = SimpleNamespace(out_cache_loc=torch.empty(n_local))
        return cap._get_local_slice(fb, can_run_graph=False, cuda_graph_batch=None)

    def test_deepep_v2_reads_buffer_head(self):
        cap, buf = self._capturer()
        with (
            mock.patch.object(re_mod, "is_dp_attention_enabled", return_value=True),
            mock.patch.object(
                re_mod, "get_moe_a2a_backend", return_value=MoeA2ABackend("deepep_v2")
            ),
        ):
            out = self._slice(cap, n_local=5)
        self.assertTrue(torch.equal(out, buf[0:5, :, : self.K]))

    def test_deepep_v2_matches_deepep(self):
        cap, _ = self._capturer()
        outs = []
        for backend in ("deepep", "deepep_v2"):
            with (
                mock.patch.object(re_mod, "is_dp_attention_enabled", return_value=True),
                mock.patch.object(
                    re_mod, "get_moe_a2a_backend", return_value=MoeA2ABackend(backend)
                ),
            ):
                outs.append(self._slice(cap, n_local=7))
        self.assertTrue(torch.equal(outs[0], outs[1]))

    def test_tp_moe_reads_global_offset(self):
        cap, buf = self._capturer()
        with (
            mock.patch.object(re_mod, "is_dp_attention_enabled", return_value=True),
            mock.patch.object(
                re_mod, "get_moe_a2a_backend", return_value=MoeA2ABackend("none")
            ),
            mock.patch.object(re_mod, "get_dp_local_slice_cpu", return_value=(6, 4)),
        ):
            out = self._slice(cap, n_local=999)
        self.assertTrue(torch.equal(out, buf[6:10, :, : self.K]))

    def test_dp_local_a2a_preserves_routes_on_nonzero_dp_rank(self):
        """DP-local routing must not read the zero-filled global-offset region."""
        for backend in ("flashinfer", "flashinfer_megamoe", "megamoe"):
            for can_run_graph in (False, True):
                with self.subTest(backend=backend, cuda_graph=can_run_graph):
                    cap, buf = self._capturer()
                    expected = buf[:3].clone()
                    buf[3:].zero_()
                    with (
                        mock.patch.object(
                            re_mod, "is_dp_attention_enabled", return_value=True
                        ),
                        mock.patch.object(
                            re_mod,
                            "get_moe_a2a_backend",
                            return_value=MoeA2ABackend(backend),
                        ),
                        mock.patch.object(
                            re_mod, "get_dp_local_slice_cpu", return_value=(8, 3)
                        ),
                    ):
                        out = cap._get_local_slice(
                            SimpleNamespace(out_cache_loc=torch.empty(3)),
                            can_run_graph=can_run_graph,
                            cuda_graph_batch=4 if can_run_graph else None,
                        )
                    torch.testing.assert_close(out, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
