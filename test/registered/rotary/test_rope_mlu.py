"""Essential MLU RoPE/MRoPE numerical and fallback checks."""

import unittest
from unittest.mock import patch

import torch
import torch_mlu_ops

from sglang.kernels.fused_op import get_fused_op_backend, set_fused_op_backend
from sglang.srt.environ import envs
from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.srt.layers.rotary_embedding.mrope import (
    Ernie4_5_VLRotaryEmbedding,
    MRotaryEmbedding,
)
from sglang.test.ci.ci_register import register_mlu_ci
from sglang.test.test_utils import CustomTestCase

register_mlu_ci(est_time=30, suite="pr-test-1-mlu")


class TestRotaryEmbeddingMLU(CustomTestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(42)
        context = envs.SGLANG_ROPE_CACHE_FP32.override(False)
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        backend = get_fused_op_backend()
        self.addCleanup(set_fused_op_backend, backend)
        set_fused_op_backend(None)

    def _check(self, rope, positions, offsets=None, native=False):
        tokens = positions.shape[-1]
        qkv = torch.randn(tokens, 512, device="mlu", dtype=torch.bfloat16)
        q, k, v = qkv.split((256, 128, 128), dim=-1)
        saved = qkv.clone()
        extra = (offsets,) if offsets is not None else ()
        expected = rope.forward_native(positions, q.clone(), k.clone(), *extra)
        with patch.object(
            torch_mlu_ops, "apply_rotary", wraps=torch_mlu_ops.apply_rotary
        ) as kernel:
            actual = rope(positions, q, k, *extra)
        self.assertEqual(kernel.call_count > 0, not native)
        for call in kernel.call_args_list:
            self.assertTrue(call.kwargs["compressed_cache"])
            self.assertEqual(call.kwargs["cos_cache"].shape[-1], rope.rotary_dim // 2)
        for result, reference, source in zip(actual, expected, (q, k)):
            torch.testing.assert_close(
                result,
                reference,
                rtol=0 if native else 0.03,
                atol=0 if native else 0.03,
            )
            if not native:
                self.assertEqual(result.data_ptr(), source.data_ptr())
        torch.testing.assert_close(v, saved[:, 384:], rtol=0, atol=0)
        torch.testing.assert_close(
            qkv[:, :384].view(tokens, 6, 64)[..., 32:],
            saved[:, :384].view(tokens, 6, 64)[..., 32:],
            rtol=0,
            atol=0,
        )

    def test_numerics(self):
        """Rotation layouts, offsets, partial rotation and MRoPE dispatch."""
        axes = torch.tensor([[1, 63, 127], [5, 17, 95], [9, 42, 121]], device="mlu")
        for neox in (True, False):
            with self.subTest(neox=neox):
                rope = RotaryEmbedding(64, 32, 128, 10000, neox, torch.bfloat16).to(
                    "mlu"
                )
                self._check(rope, (axes[0] - 1).int(), offsets=torch.ones_like(axes[0]))
        rope = MRotaryEmbedding(
            64, 32, 128, 10000, True, torch.bfloat16, mrope_section=[4, 4, 8]
        ).to("mlu")
        for positions in (axes[0], axes):
            self._check(rope, positions, native=positions.ndim == 2)

    def test_native_fallback(self):
        """FP32 coefficients and Ernie's specialized layout bypass the vendor op."""
        axes = torch.arange(9, device="mlu").reshape(3, 3)
        with envs.SGLANG_ROPE_CACHE_FP32.override(True):
            ordinary = RotaryEmbedding(64, 32, 128, 10000, True, torch.bfloat16).to(
                "mlu"
            )
        self._check(ordinary, axes[0], native=True)
        ernie = Ernie4_5_VLRotaryEmbedding(
            64, 32, 128, 10000, True, torch.bfloat16, mrope_section=[4, 4, 8]
        ).to("mlu")
        for positions in (axes[0], axes):
            self._check(ernie, positions, native=True)

    def test_rejects_fused_kv_write(self):
        rope = RotaryEmbedding(64, 32, 128, 10000, True, torch.bfloat16).to("mlu")
        positions = torch.zeros(1, device="mlu", dtype=torch.int32)
        q = torch.ones(1, 256, device="mlu", dtype=torch.bfloat16)
        k = torch.ones(1, 128, device="mlu", dtype=torch.bfloat16)
        with self.assertRaisesRegex(NotImplementedError, "fused_set_kv_buffer_arg"):
            rope(positions, q, k, fused_set_kv_buffer_arg=object())
        torch.testing.assert_close(q, torch.ones_like(q), rtol=0, atol=0)
        torch.testing.assert_close(k, torch.ones_like(k), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
