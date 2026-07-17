"""Integration tests for the fused norm+quant wiring.

Covers the two integration points (the kernel itself is covered by
test_fused_add_rmsnorm_per_token_quant.py):
  1. RMSNorm.forward_cuda attaches the pre-quantized activation as tensor
     attributes when SGLANG_FUSED_NORM_FP8_QUANT=1.
  2. apply_fp8_linear consumes the attributes on the compressed-tensors
     (dynamic per-token) path instead of re-quantizing.
"""

import os
import unittest

import torch

os.environ["SGLANG_FUSED_NORM_FP8_QUANT"] = "1"


def _reference_per_token_quant(x_bf16):
    x = x_bf16.float()
    absmax = x.abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(absmax / 448.0, min=1e-10)
    fp8 = (x / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return fp8, scale


class TestRMSNormWiring(unittest.TestCase):
    def test_forward_cuda_attaches_fp8(self):
        import sglang.srt.layers.layernorm as layernorm_mod
        from sglang.srt.layers.layernorm import RMSNorm

        # reset the lazy cache in case another test imported with the env off
        layernorm_mod._fused_norm_fp8_quant_state = None

        m, d = 64, 8192
        torch.manual_seed(0)
        norm = RMSNorm(d).cuda().to(torch.bfloat16)
        x = torch.randn(m, d, dtype=torch.bfloat16, device="cuda")
        residual = torch.randn(m, d, dtype=torch.bfloat16, device="cuda")

        out, residual_out = norm.forward_cuda(x.clone(), residual.clone())

        fp8 = getattr(out, "_sglang_fp8_data", None)
        scale = getattr(out, "_sglang_fp8_scale", None)
        self.assertIsNotNone(fp8, "fused path did not attach _sglang_fp8_data")
        self.assertIsNotNone(scale, "fused path did not attach _sglang_fp8_scale")
        self.assertEqual(fp8.dtype, torch.float8_e4m3fn)
        self.assertEqual(scale.shape, (m, 1))

        # attached fp8 must dequantize back to the bf16 output within fp8 error
        dequant = fp8.float() * scale
        # e4m3 has 3 mantissa bits -> max relative rounding error 2^-4 = 6.25%
        torch.testing.assert_close(
            dequant, out.float(), atol=2e-2, rtol=7e-2
        )

        # and must agree with an independent per-token quant of the output
        ref_fp8, ref_scale = _reference_per_token_quant(out)
        # both sides are independently fp8-rounded -> errors can compound
        torch.testing.assert_close(
            fp8.float() * scale, ref_fp8.float() * ref_scale, atol=4e-2, rtol=1.3e-1
        )

    def test_disabled_without_env(self):
        import sglang.srt.layers.layernorm as layernorm_mod
        from sglang.srt.layers.layernorm import RMSNorm

        layernorm_mod._fused_norm_fp8_quant_state = None
        os.environ["SGLANG_FUSED_NORM_FP8_QUANT"] = "0"
        try:
            m, d = 8, 8192
            norm = RMSNorm(d).cuda().to(torch.bfloat16)
            x = torch.randn(m, d, dtype=torch.bfloat16, device="cuda")
            residual = torch.randn(m, d, dtype=torch.bfloat16, device="cuda")
            out, _ = norm.forward_cuda(x.clone(), residual.clone())
            self.assertIsNone(getattr(out, "_sglang_fp8_data", None))
        finally:
            os.environ["SGLANG_FUSED_NORM_FP8_QUANT"] = "1"
            layernorm_mod._fused_norm_fp8_quant_state = None


class TestApplyFp8LinearConsumesPrequant(unittest.TestCase):
    def _make_fp8_weight(self, k, n):
        # build [n, k] row-major then transpose: cuBLASLt fallback requires a
        # column-major B operand (matches process_weights_after_loading layout)
        w = torch.randn(n, k, dtype=torch.float32, device="cuda")
        w_scale = (w.abs().max() / 448.0).clamp(min=1e-10)
        w_fp8 = (w / w_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        return w_fp8.t(), w_scale.reshape(1)

    def test_poisoned_attrs_are_consumed(self):
        """Attach fp8 attrs derived from a DIFFERENT tensor; if the hook is
        live, the GEMM result must follow the attached data, not the input."""
        from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear

        torch.manual_seed(1)
        m, k, n = 32, 8192, 1024
        weight, weight_scale = self._make_fp8_weight(k, n)

        input_real = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        input_poison = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")

        kwargs = dict(
            weight=weight,
            weight_scale=weight_scale,
            input_scale=None,
            use_per_token_if_dynamic=True,
            compressed_tensor_quant=True,
        )

        # baseline: no attrs -> quantizes input_real
        out_real = apply_fp8_linear(input=input_real.clone(), **kwargs)
        # poison: attrs carry quant(input_poison)
        poisoned = input_real.clone()
        fp8_p, scale_p = _reference_per_token_quant(input_poison)
        poisoned._sglang_fp8_data = fp8_p
        poisoned._sglang_fp8_scale = scale_p
        out_poisoned = apply_fp8_linear(input=poisoned, **kwargs)
        # reference for the poison source
        out_poison_ref = apply_fp8_linear(input=input_poison.clone(), **kwargs)

        # hook must be live: poisoned result tracks the poison source...
        torch.testing.assert_close(
            out_poisoned.float(), out_poison_ref.float(), atol=5e-2, rtol=5e-2
        )
        # ...and clearly differs from the unpoisoned result
        self.assertGreater((out_poisoned - out_real).abs().max().item(), 0.1)

    def test_correct_attrs_match_unfused(self):
        """Attrs carrying the true quant of the input must reproduce the
        unfused result (same kernel-level rounding contract)."""
        from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear

        torch.manual_seed(2)
        m, k, n = 16, 8192, 512
        weight, weight_scale = self._make_fp8_weight(k, n)
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")

        kwargs = dict(
            weight=weight,
            weight_scale=weight_scale,
            input_scale=None,
            use_per_token_if_dynamic=True,
            compressed_tensor_quant=True,
        )
        out_unfused = apply_fp8_linear(input=x.clone(), **kwargs)

        x_attr = x.clone()
        fp8, scale = _reference_per_token_quant(x)
        x_attr._sglang_fp8_data = fp8
        x_attr._sglang_fp8_scale = scale
        out_fused = apply_fp8_linear(input=x_attr, **kwargs)

        # the two paths round independently (sgl kernel vs python reference)
        # before the same GEMM; allow one fp8 ulp of relative slack
        torch.testing.assert_close(
            out_fused.float(), out_unfused.float(), atol=5e-2, rtol=1e-1
        )


if __name__ == "__main__":
    unittest.main()
