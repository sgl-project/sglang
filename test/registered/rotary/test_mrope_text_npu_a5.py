"""Ascend 950 framework text RoPE dispatch through the unified CANN API."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=30, suite="nightly-acc-test-4-npu-a5", nightly=True)


class TestTextMropeA5(CustomTestCase):
    def _check_text_rope(self, head_size, query_heads, key_heads):
        from sgl_kernel_npu.operators import rope as npu_rope_ops
        from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware
        from sgl_kernel_npu.operators.utils.registry import get_impl

        self.assertIs(get_hardware().generation, Generation.ARCH35)
        self.assertIs(
            get_impl(npu_rope_ops.mrope, "auto"),
            get_impl(npu_rope_ops.mrope, "ops-transformer"),
        )
        rot = RotaryEmbedding(
            head_size=head_size,
            rotary_dim=head_size,
            max_position_embeddings=32768,
            base=1000000.0,
            is_neox_style=True,
            dtype=torch.bfloat16,
        ).to("npu")
        torch.manual_seed(42)
        # Preserve split-QKV views, as in the model, rather than constructing
        # independent contiguous Q and K tensors.
        packed = torch.randn(
            16,
            (query_heads + 2 * key_heads) * head_size,
            dtype=torch.bfloat16,
            device="npu",
        )
        query, key, _ = packed.split(
            (query_heads * head_size, key_heads * head_size, key_heads * head_size), -1
        )
        positions = torch.arange(16, dtype=torch.int64, device="npu")
        ref_q, ref_k = rot.forward_native(positions, query.clone(), key.clone())
        with patch.object(npu_rope_ops, "mrope", wraps=npu_rope_ops.mrope) as public_op:
            out_q, out_k = rot.forward_npu(positions, query, key)
        public_op.assert_called_once()
        self.assertEqual(public_op.call_args.kwargs["mrope_section"], [0, 0, 0])
        self.assertEqual(public_op.call_args.kwargs["rotary_mode"], "half")
        for actual, expected in ((out_q, ref_q), (out_k, ref_k)):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.dtype, expected.dtype)
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_qwen2_05b_text_prefill_tp1(self):
        self._check_text_rope(64, 14, 2)

    def test_qwen25_7b_text_prefill_tp2(self):
        self._check_text_rope(128, 14, 2)


if __name__ == "__main__":
    unittest.main()
