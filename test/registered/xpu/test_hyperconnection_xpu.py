"""XPU numerical regression test for the fused HC combine + next-norm branch."""

import unittest

import torch
import torch.nn.functional as F

from sglang.srt.layers.hyperconnection import (
    GatedResidual,
    GroupedGemmaRMSNorm,
    HyperConnectionConfig,
)
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=30, suite="stage-b-test-1-gpu-xpu")


def _fused_op_available():
    if not torch.xpu.is_available():
        return False
    try:
        from sgl_kernel import hc_combine_norm  # noqa: F401
    except ImportError:
        return False
    return True


def _reference(block_output, residual, normed, inject_w, norm, hc, hs):
    # Unfused: combine, round to BF16, then per-stream grouped Gemma RMSNorm.
    gate = 2 * torch.sigmoid(F.linear(normed, inject_w).float() / hc)
    combined = (
        residual.unflatten(-1, (hc, hs)).float()
        + block_output.float().unsqueeze(-2) * gate.unsqueeze(-1)
    ).flatten(-2)
    combined = combined.to(torch.bfloat16)
    x = combined.float().unflatten(-1, (hc, hs))
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
    weight = norm.weight.float().view(-1, hs) if norm.weight.numel() > hs else None
    scale = 1.0 + (norm.weight.float() if weight is None else weight)
    return combined, (x * scale).flatten(-2).to(torch.bfloat16)


@unittest.skipUnless(
    _fused_op_available(), "requires XPU and sgl_kernel hc_combine_norm"
)
class TestHyperConnectionCombineNormXPU(unittest.TestCase):
    def _run(self, num_tokens, per_stream_weight):
        hc, hs = 4, 512
        torch.manual_seed(0)
        module = GatedResidual(
            HyperConnectionConfig(
                hc_count=hc,
                hidden_size=hs,
                params_dtype=torch.bfloat16,
                hc_per_branch_norm=True,
            ),
            use_mix=False,
        ).to("xpu")
        with torch.no_grad():
            module.block_inject_weight.weight.normal_(0, 0.05)
        norm = GroupedGemmaRMSNorm(
            hc * hs if per_stream_weight else hs, eps=1e-6, group_size=hs
        ).to("xpu")
        with torch.no_grad():
            norm.weight.normal_(0, 0.1)
        norm.weight.data = norm.weight.data.to(torch.bfloat16)

        residual = torch.randn(num_tokens, hc * hs, device="xpu", dtype=torch.bfloat16)
        normed = torch.randn_like(residual)
        block_output = torch.randn(num_tokens, hs, device="xpu", dtype=torch.bfloat16)

        combined, normalized = module.combine_and_normalize(
            block_output, (residual, normed, None), norm
        )
        self.assertIsNotNone(normalized)
        ref_combined, ref_normalized = _reference(
            block_output,
            residual,
            normed,
            module.block_inject_weight.weight,
            norm,
            hc,
            hs,
        )
        torch.testing.assert_close(combined, ref_combined, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(
            normalized.reshape(num_tokens, -1),
            ref_normalized,
            rtol=3e-2,
            atol=3e-2,
        )

    def test_shared_norm_weight(self):
        for n in (1, 7, 64):
            with self.subTest(num_tokens=n):
                self._run(n, per_stream_weight=False)

    def test_per_stream_norm_weight(self):
        for n in (1, 7, 64):
            with self.subTest(num_tokens=n):
                self._run(n, per_stream_weight=True)


if __name__ == "__main__":
    unittest.main()
