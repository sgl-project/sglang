"""DeepEP-class backend recognition in RoutedExpertsCapturer.

The capturer keys its buffer layout on the a2a backend: DeepEP-class
dispatchers hand the MoE layer only the attention rank's DP-local tokens, so
``capture()`` must attn-TP-gather and ``_get_local_slice()`` must read the
buffer head instead of the global DP offset. These tests pin that DeepEP v2
is classified like DeepEP (it shares that token topology); a miss makes
dp_rank > 0 read unwritten rows (silent wrong data), see the DP>1 readback
test in test/registered/ep/test_routed_experts_dp_readback.py.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.state_capturer import routed_experts as re_mod
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")


class TestScatteredA2ABackendHelper(CustomTestCase):
    def test_classification(self):
        # deepep_v2 shares DeepEP's scattered token topology. Other backends
        # keep their existing classification (mooncake/mori are deliberately
        # not reclassified here).
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
    T, L, K = 16, 3, 4  # buffer tokens, layers, top-k

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
        with mock.patch.object(
            re_mod, "is_dp_attention_enabled", return_value=True
        ), mock.patch.object(
            re_mod, "get_moe_a2a_backend", return_value=MoeA2ABackend("deepep_v2")
        ):
            out = self._slice(cap, n_local=5)
        self.assertTrue(torch.equal(out, buf[0:5, :, : self.K]))

    def test_deepep_v2_matches_deepep(self):
        cap, _ = self._capturer()
        outs = []
        for backend in ("deepep", "deepep_v2"):
            with mock.patch.object(
                re_mod, "is_dp_attention_enabled", return_value=True
            ), mock.patch.object(
                re_mod, "get_moe_a2a_backend", return_value=MoeA2ABackend(backend)
            ):
                outs.append(self._slice(cap, n_local=7))
        self.assertTrue(torch.equal(outs[0], outs[1]))

    def test_tp_moe_reads_global_offset(self):
        cap, buf = self._capturer()
        with mock.patch.object(
            re_mod, "is_dp_attention_enabled", return_value=True
        ), mock.patch.object(
            re_mod, "get_moe_a2a_backend", return_value=MoeA2ABackend("none")
        ), mock.patch.object(
            re_mod, "get_dp_local_slice_cpu", return_value=(6, 4)
        ):
            out = self._slice(cap, n_local=999)
        self.assertTrue(torch.equal(out, buf[6:10, :, : self.K]))


if __name__ == "__main__":
    unittest.main()
