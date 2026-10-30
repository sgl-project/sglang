"""M-RoPE axis mapping on Ascend NPU.

``forward_npu`` runs the Triton kernel that ``forward_cuda`` uses for 3-axis
positions, and keeps the fused CANN op for 1-D positions. Both are compared
against ``forward_native`` here: exactly in fp32, and within bf16 rounding for
the deployment dtype.
"""

import unittest

import torch

from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
from sglang.srt.utils import is_npu
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=30, suite="full-1-npu-a3", nightly=True)


@unittest.skipUnless(is_npu(), "Ascend NPU not available")
class TestMRotaryEmbeddingNpuSection(CustomTestCase):
    def _build(self, dtype, **mrope_flags):
        return MRotaryEmbedding(
            head_size=128,
            rotary_dim=128,
            max_position_embeddings=8192,
            base=10000,
            is_neox_style=False,
            dtype=dtype,
            mrope_section=[16, 24, 24],
            **mrope_flags,
        ).to("npu")

    @staticmethod
    def _tokens(seq, dtype, heads=16):
        q = torch.randn(seq, heads * 128, dtype=dtype, device="npu")
        k = torch.randn(seq, (heads // 2) * 128, dtype=dtype, device="npu")
        return q, k

    @staticmethod
    def _positions_3d(seq):
        t = torch.arange(seq)
        return torch.stack([t, t * 2 + 1, t * 3 + 5]).to(torch.int64).npu()

    def _assert_2d_matches_native(self, dtype, **mrope_flags):
        rot = self._build(dtype, **mrope_flags)
        torch.manual_seed(0)
        seq = 8
        positions = self._positions_3d(seq)
        q, k = self._tokens(seq, dtype)

        out_q, out_k = rot.forward_npu(positions, q.clone(), k.clone())
        ref_q, ref_k = rot.forward_native(positions, q.clone(), k.clone())

        if dtype == torch.float32:
            # A wrong axis map moves these by several units, not by rounding.
            self.assertTrue(torch.equal(out_q, ref_q))
            self.assertTrue(torch.equal(out_k, ref_k))
            return

        # bf16 rounds the same fp32 arithmetic differently; measured 0.3% of
        # the reference norm and at most 2 ulp.
        for out, ref in ((out_q, ref_q), (out_k, ref_k)):
            delta = (out.float() - ref.float()).abs()
            self.assertLess(delta.norm() / ref.float().norm(), 5e-3)
            self.assertLessEqual(delta.max().item(), 0.05)

    def test_contiguous_axis_map(self):
        # GLM-OCR: lanes split as [0]*16 [1]*24 [2]*24.
        self._assert_2d_matches_native(torch.float32, mrope_interleaved=False)

    def test_interleaved_axis_map(self):
        # Qwen2-VL: axes assigned round-robin.
        self._assert_2d_matches_native(torch.float32, mrope_interleaved=True)

    def test_glm_interleaved_axis_map(self):
        # GLM style: round-robin over the axes, each capped by its section.
        self._assert_2d_matches_native(torch.float32, mrope_interleaved_glm=True)

    def test_bfloat16_deployment_dtype(self):
        for flags in (
            {"mrope_interleaved": False},
            {"mrope_interleaved": True},
            {"mrope_interleaved_glm": True},
        ):
            self._assert_2d_matches_native(torch.bfloat16, **flags)

    def test_text_positions_match_native(self):
        # 1-D positions carry no axis map; they must still get plain RoPE.
        rot = self._build(torch.float32)
        torch.manual_seed(0)
        seq = 16
        positions = torch.arange(seq).to(torch.int64).npu()
        q, k = self._tokens(seq, torch.float32)

        out_q, out_k = rot.forward_npu(positions, q.clone(), k.clone())
        ref_q, ref_k = rot.forward_native(positions, q.clone(), k.clone())

        torch.testing.assert_close(out_q, ref_q, rtol=0, atol=1e-5)
        torch.testing.assert_close(out_k, ref_k, rtol=0, atol=1e-5)

    def test_empty_tokens_short_circuit(self):
        # The Triton kernel launches a zero-block grid and aborts the process,
        # so a 0-token batch must return before dispatch.
        rot = self._build(torch.bfloat16)
        for positions in (
            torch.zeros((3, 0), dtype=torch.int64).npu(),
            torch.zeros((0,), dtype=torch.int64).npu(),
        ):
            q, k = self._tokens(0, torch.bfloat16)
            out_q, out_k = rot.forward_npu(positions, q, k)
            self.assertEqual(out_q.shape, q.shape)
            self.assertEqual(out_k.shape, k.shape)


if __name__ == "__main__":
    unittest.main()
