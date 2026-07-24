"""CPU tests for Kimi-K3 fused-decode post-load topology preparation."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.kimi_k3 import KimiK3DeltaAttention
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _make_owner(heads: int):
    seg = heads * 128
    layer = SimpleNamespace(
        num_q_heads=heads,
        conv_weights=torch.randn(3 * seg, 4, dtype=torch.float32),
        bias=None,
        A_log=torch.randn(1, 1, heads, 1, dtype=torch.float32),
        dt_bias=torch.randn(seg, dtype=torch.float32),
    )
    owner = SimpleNamespace(
        attn=layer,
        o_norm=SimpleNamespace(
            weight=torch.ones(128, dtype=torch.float32),
            eps=1e-6,
        ),
        _kda_fused_decode_ready=False,
    )
    return owner


class TestKimiK3FusedDecodeLayout(CustomTestCase):
    def test_prepares_tp8_and_tp16_static_inputs(self):
        for heads in (6, 12):
            owner = _make_owner(heads)
            KimiK3DeltaAttention._prepare_fused_decode(owner)

            self.assertTrue(owner._kda_fused_decode_ready)
            fused = owner.attn._k3_fused_decode_args
            seg = heads * 128
            self.assertEqual(tuple(fused[0].shape), (4, seg))
            self.assertEqual(tuple(fused[1].shape), (4, seg))
            self.assertEqual(tuple(fused[2].shape), (4, seg))
            self.assertEqual(tuple(fused[3].shape), (3 * seg,))
            self.assertEqual(tuple(fused[4].shape), (heads,))

    def test_unsupported_tp32_layout_keeps_fallback(self):
        owner = _make_owner(3)
        KimiK3DeltaAttention._prepare_fused_decode(owner)

        self.assertFalse(owner._kda_fused_decode_ready)
        self.assertFalse(hasattr(owner.attn, "_k3_fused_decode_args"))


if __name__ == "__main__":
    unittest.main()
