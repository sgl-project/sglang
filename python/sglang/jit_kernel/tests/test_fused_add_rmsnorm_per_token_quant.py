"""Tests for fused Add+RMSNorm + per-token FP8 quantization kernel."""

import torch
import torch.nn.functional as F
import unittest


def reference_add_rmsnorm_per_token_quant(input_bf16, residual_bf16, weight_bf16, eps=1e-6):
    """Reference implementation using separate operations."""
    # Residual add
    residual_ref = residual_bf16.float() + input_bf16.float()

    # RMSNorm
    variance = residual_ref.pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(variance + eps)
    normed = (residual_ref * rstd * weight_bf16.float()).to(torch.bfloat16)

    # Per-token FP8 quant
    normed_f = normed.float()
    absmax = normed_f.abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(absmax / 448.0, min=1e-10)
    inv_scale = 1.0 / scale
    fp8_out = (normed_f * inv_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

    return normed, fp8_out, scale, residual_ref.to(torch.bfloat16)


class TestFusedAddRMSNormPerTokenQuant(unittest.TestCase):

    def _run_test(self, m, d, eps=1e-6):
        from sglang.jit_kernel.fused_add_rmsnorm_per_token_quant import (
            fused_add_rmsnorm_per_token_quant,
        )

        torch.manual_seed(42)
        input_bf16 = torch.randn(m, d, dtype=torch.bfloat16, device="cuda")
        residual_bf16 = torch.randn(m, d, dtype=torch.bfloat16, device="cuda")
        weight_bf16 = torch.randn(d, dtype=torch.bfloat16, device="cuda")

        # Clone residual for reference (it's modified in-place)
        residual_clone = residual_bf16.clone()

        # Reference
        ref_bf16, ref_fp8, ref_scale, ref_residual = reference_add_rmsnorm_per_token_quant(
            input_bf16, residual_clone, weight_bf16, eps
        )

        # Fused kernel
        fused_bf16, fused_fp8, fused_scale = fused_add_rmsnorm_per_token_quant(
            input_bf16, residual_bf16, weight_bf16, eps=eps
        )

        # Check residual updated correctly
        res_diff = (residual_bf16.float() - ref_residual.float()).abs().max().item()

        # Check BF16 normed output
        bf16_diff = (fused_bf16.float() - ref_bf16.float()).abs().max().item()

        # Check scale
        scale_diff = (fused_scale - ref_scale).abs().max().item()
        scale_rel = scale_diff / ref_scale.abs().max().item()

        # Check FP8 match rate
        match_rate = (fused_fp8.float() == ref_fp8.float()).float().mean().item()

        print(
            f"  m={m:4d} d={d:4d} | res_diff={res_diff:.6f} bf16_diff={bf16_diff:.6f} "
            f"scale_rel={scale_rel:.6f} fp8_match={match_rate:.2%}"
        )

        self.assertLess(res_diff, 0.01, f"Residual mismatch: {res_diff}")
        self.assertLess(bf16_diff, 0.1, f"BF16 normed mismatch: {bf16_diff}")
        self.assertLess(scale_rel, 0.01, f"Scale mismatch: {scale_rel}")
        self.assertGreater(match_rate, 0.90, f"FP8 match rate too low: {match_rate:.2%}")

    def test_qwen35_dims(self):
        """Qwen3.5-35B-A3B: hidden_size=2048"""
        self._run_test(1, 2048)

    def test_small_batch(self):
        self._run_test(4, 2048)

    def test_medium_batch(self):
        self._run_test(64, 2048)

    def test_large_hidden(self):
        self._run_test(8, 4096)

    def test_single_token(self):
        self._run_test(1, 512)

    def test_rmsnorm_only(self):
        """Test the no-residual variant."""
        from sglang.jit_kernel.fused_add_rmsnorm_per_token_quant import (
            fused_rmsnorm_per_token_quant,
        )

        m, d = 16, 2048
        torch.manual_seed(42)
        input_bf16 = torch.randn(m, d, dtype=torch.bfloat16, device="cuda")
        weight_bf16 = torch.randn(d, dtype=torch.bfloat16, device="cuda")

        fused_bf16, fused_fp8, fused_scale = fused_rmsnorm_per_token_quant(
            input_bf16, weight_bf16
        )

        # Reference
        inp_f = input_bf16.float()
        var = inp_f.pow(2).mean(dim=-1, keepdim=True)
        normed_ref = (inp_f * torch.rsqrt(var + 1e-6) * weight_bf16.float()).to(torch.bfloat16)

        bf16_diff = (fused_bf16.float() - normed_ref.float()).abs().max().item()
        print(f"  rmsnorm_only: bf16_diff={bf16_diff:.6f}")
        self.assertLess(bf16_diff, 0.1)

    def test_empty(self):
        from sglang.jit_kernel.fused_add_rmsnorm_per_token_quant import (
            fused_add_rmsnorm_per_token_quant,
        )
        inp = torch.empty(0, 2048, dtype=torch.bfloat16, device="cuda")
        res = torch.empty(0, 2048, dtype=torch.bfloat16, device="cuda")
        w = torch.ones(2048, dtype=torch.bfloat16, device="cuda")
        bf16, fp8, scale = fused_add_rmsnorm_per_token_quant(inp, res, w)
        self.assertEqual(bf16.shape, (0, 2048))


if __name__ == "__main__":
    unittest.main()
