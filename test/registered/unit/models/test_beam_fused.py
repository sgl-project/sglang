"""The fused Beam norm/residual and output-gate kernels reproduce eager rounding."""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

import math
import unittest

import torch

from sglang.kernels.ops.elementwise import beam_fused
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.models.beam import apply_attention_gate, depth_scaled_residual
from sglang.test.test_utils import CustomTestCase

HIDDEN, HEADS, HEAD_DIM, EPS = 5120, 12, 128, 1e-6


def _rows(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, HIDDEN, generator=g).to(torch.bfloat16).cuda() * 3


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestBeamFusedOps(CustomTestCase):
    def test_sandwich_residual_rmsnorm_matches_eager(self):
        torch.manual_seed(0)
        post = RMSNorm(HIDDEN, eps=EPS).cuda().to(torch.bfloat16)
        nxt = RMSNorm(HIDDEN, eps=EPS).cuda().to(torch.bfloat16)
        post.weight.data.uniform_(0.5, 1.5)
        nxt.weight.data.uniform_(0.5, 1.5)
        scales = (0.25, 0.968)
        for rows in (1, 64, 1024):
            for combine_in_fp32 in (False, True):
                with self.subTest(rows=rows, combine_in_fp32=combine_in_fp32):
                    branch, residual = _rows(rows, 1), _rows(rows, 2)
                    ref_combined = depth_scaled_residual(
                        post(branch), residual, scales, combine_in_fp32
                    )
                    ref_out = nxt(ref_combined)
                    out, combined = beam_fused.fused_sandwich_residual_rmsnorm(
                        branch,
                        residual,
                        post.weight.data,
                        nxt.weight.data,
                        EPS,
                        scales,
                        combine_in_fp32,
                    )
                    # Bound error on these fixed samples. Reduction differences
                    # can exceed one ULP when the scaled operands nearly cancel.
                    mismatch = (combined != ref_combined).float().mean().item()
                    self.assertLess(mismatch, 2e-3)
                    torch.testing.assert_close(
                        combined, ref_combined, atol=1e-6, rtol=2**-7
                    )
                    torch.testing.assert_close(out, ref_out, atol=1e-6, rtol=2**-6)

    def test_combine_without_sandwich_is_exact(self):
        branch, residual = _rows(8, 3), _rows(8, 4)
        out, combined = beam_fused.fused_sandwich_residual_rmsnorm(
            branch, residual, None, None, EPS, (0.5, 0.5), False
        )
        self.assertIsNone(out)
        self.assertTrue(
            torch.equal(
                combined, depth_scaled_residual(branch, residual, (0.5, 0.5), False)
            )
        )

    def test_fp32_residual_cancellation_stays_zero(self):
        """Separate FP32 products must cancel before the next norm amplifies them."""
        branch = torch.ones(1, HIDDEN, device="cuda", dtype=torch.bfloat16)
        residual = -torch.ones_like(branch)
        post = RMSNorm(HIDDEN, eps=EPS).cuda().to(torch.bfloat16)
        nxt = RMSNorm(HIDDEN, eps=EPS).cuda().to(torch.bfloat16)
        post.weight.data.fill_(5)
        nxt.weight.data.fill_(1)
        scales = (1 / math.sqrt(26), 5 / math.sqrt(26))
        expected = depth_scaled_residual(post(branch), residual, scales, True)
        self.assertEqual(torch.count_nonzero(expected).item(), 0)
        out, combined = beam_fused.fused_sandwich_residual_rmsnorm(
            branch, residual, post.weight, nxt.weight, EPS, scales, True
        )
        torch.testing.assert_close(combined, expected, atol=0, rtol=0)
        torch.testing.assert_close(out, nxt(expected), atol=0, rtol=0)

    def test_fp16_gate_rounding_boundaries(self):
        """FP16 sigmoid must round the same way as eager near half-way values."""
        gate = torch.tensor(
            [[0.0068359375, -0.00634765625]], device="cuda", dtype=torch.float16
        )
        attention = torch.ones_like(gate)
        bias = torch.zeros(1, device="cuda", dtype=torch.float16)
        expected = apply_attention_gate(
            attention.view(1, 1, 2), gate.view(1, 1, 2), bias
        ).view_as(gate)
        actual = beam_fused.fused_attention_gate(attention, gate, bias, 2)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_attention_gate_matches_eager(self):
        g = torch.Generator().manual_seed(5)
        bias = torch.randn(HEADS, generator=g).to(torch.bfloat16).cuda()
        for rows in (1, 64, 1024):
            with self.subTest(rows=rows):
                attn = (
                    torch.randn(rows, HEADS * HEAD_DIM, generator=g)
                    .to(torch.bfloat16)
                    .cuda()
                )
                gate = (
                    torch.randn(rows, HEADS * HEAD_DIM, generator=g)
                    .to(torch.bfloat16)
                    .cuda()
                    * 2
                )
                ref = apply_attention_gate(
                    attn.view(-1, HEADS, HEAD_DIM), gate.view(-1, HEADS, HEAD_DIM), bias
                ).view(rows, -1)
                out = beam_fused.fused_attention_gate(attn, gate, bias, HEAD_DIM)
                self.assertTrue(torch.equal(out, ref))


if __name__ == "__main__":
    unittest.main()
