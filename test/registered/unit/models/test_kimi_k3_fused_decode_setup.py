"""Unit tests for Kimi K3 fused KDA decode setup."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.kimi_k3 import KimiK3DeltaAttention
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestKimiK3FusedDecodeSetup(CustomTestCase):
    @staticmethod
    def _make_attention(num_heads: int):
        segment_size = num_heads * 128
        layer = SimpleNamespace(
            num_v_heads=num_heads,
            conv_weights=torch.empty(3 * segment_size, 4, dtype=torch.float32),
            A_log=torch.empty(1, 1, num_heads, 1, dtype=torch.float32),
            dt_bias=torch.empty(segment_size, dtype=torch.float32),
            bias=None,
        )
        return SimpleNamespace(
            attn=layer,
            o_norm=SimpleNamespace(
                weight=torch.nn.Parameter(torch.empty(segment_size)),
                eps=1e-5,
            ),
            _kda_fused_decode_ready=False,
        )

    def test_supports_all_compiled_tp_shapes(self):
        for num_heads in (3, 6, 12):
            with self.subTest(num_heads=num_heads):
                attention = self._make_attention(num_heads)

                KimiK3DeltaAttention._prepare_fused_decode(attention)

                self.assertTrue(attention._kda_fused_decode_ready)
                w_q_t, w_k_t, w_v_t, conv_bias, a_log, onorm_weight, _ = (
                    attention.attn._k3_fused_decode_args
                )
                segment_size = num_heads * 128
                self.assertEqual(w_q_t.shape, (4, segment_size))
                self.assertEqual(w_k_t.shape, (4, segment_size))
                self.assertEqual(w_v_t.shape, (4, segment_size))
                self.assertEqual(conv_bias.shape, (3 * segment_size,))
                self.assertEqual(a_log.shape, (num_heads,))
                self.assertEqual(onorm_weight.shape, (segment_size,))

    def test_rejects_uncompiled_head_count(self):
        attention = self._make_attention(num_heads=4)

        KimiK3DeltaAttention._prepare_fused_decode(attention)

        self.assertFalse(attention._kda_fused_decode_ready)
        self.assertFalse(hasattr(attention.attn, "_k3_fused_decode_args"))


if __name__ == "__main__":
    unittest.main()
