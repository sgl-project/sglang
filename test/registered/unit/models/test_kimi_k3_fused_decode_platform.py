"""Kimi-K3 prepares the fused KDA decode kernel only on CUDA.

The kernel is JIT-compiled CUDA, and ``kda_fused_decode.covered()`` checks only
shapes, dtypes and strides. Static inputs published on another platform would
route decode into a CUDA build whenever the local head count is supported (K3
at TP16 has 6). Pure Python: ``_prepare_fused_decode`` runs on a stand-in
attention module with the platform patched.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.models import kimi_k3
from sglang.srt.models.kimi_k3 import KimiK3DeltaAttention
from sglang.test.test_utils import CustomTestCase


def _attention(heads: int = 6) -> SimpleNamespace:
    seg = heads * 128
    layer = SimpleNamespace(
        conv_weights=torch.randn(3 * seg, 4),
        bias=None,
        A_log=torch.randn(heads),
        dt_bias=torch.randn(seg),
        _k3_fused_decode_args=None,
    )
    return SimpleNamespace(
        attn=layer,
        local_num_heads=heads,
        o_norm=SimpleNamespace(weight=torch.ones(128), eps=1e-5),
        _kda_fused_decode_ready=False,
    )


class TestKimiK3FusedDecodePlatform(CustomTestCase):
    def _prepare(self, attention: SimpleNamespace, is_cuda: bool) -> None:
        platform = SimpleNamespace(is_cuda=is_cuda)
        with (
            mock.patch.object(kimi_k3, "_is_hip", False),
            mock.patch.object(kimi_k3, "get_platform", return_value=platform),
        ):
            KimiK3DeltaAttention._prepare_fused_decode(attention)

    def test_non_cuda_platforms_leave_the_fused_decode_unarmed(self):
        attention = _attention()
        self._prepare(attention, is_cuda=False)
        self.assertIsNone(attention.attn._k3_fused_decode_args)
        self.assertFalse(attention._kda_fused_decode_ready)

    def test_cuda_arms_the_fused_decode_for_supported_heads(self):
        attention = _attention()
        self._prepare(attention, is_cuda=True)
        self.assertEqual(len(attention.attn._k3_fused_decode_args), 7)
        self.assertTrue(attention._kda_fused_decode_ready)


if __name__ == "__main__":
    unittest.main()
