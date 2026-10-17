"""MI35x parity test for the fused per-token FP8 tuple path in apply_fp8_linear.

Compares the "direct-write" pre-quantized tuple path (what the fused
RMSNorm+quant fold feeds into ``apply_fp8_linear``) against the standard
non-tuple path that quantizes the activation itself, and checks:

  * numerical parity: feeding a pre-quantized ``(fp8, per-token scale)`` tuple
    produces the same GEMM output as the standard path (same quant kernel, same
    ``gemm_a8w8_bpreshuffle``), and
  * dtype preservation: the tuple path honors the carried original dtype, so an
    FP16 activation stays FP16 and is NOT silently promoted to BF16, while a
    2-tuple (no dtype) falls back to ``pre_quant_output_dtype`` or BF16.

Requires ROCm/aiter on gfx95 (MI35x); skipped elsewhere.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

try:
    import sglang.srt.layers.quantization.fp8_utils as fp8u
    from sglang.srt.models.deepseek_common.utils import (
        _use_aiter,
        _use_aiter_bpreshuffle_gfx95,
    )

    _HAS_FOLD_PATH = (
        _use_aiter and _use_aiter_bpreshuffle_gfx95 and torch.cuda.is_available()
    )
except Exception:
    _HAS_FOLD_PATH = False


@unittest.skipUnless(
    _HAS_FOLD_PATH, "requires ROCm/aiter gfx95 (MI35x) with bpreshuffle GEMM"
)
class TestPerTokenFp8FoldParity(CustomTestCase):
    M, K, N = 128, 512, 2112  # 2112 = fused entry-proj output width for this ckpt

    def _make_case(self, dtype):
        device = "cuda"
        x = torch.randn(self.M, self.K, dtype=dtype, device=device) * 0.1
        # weight laid out (K, N) so apply_fp8_linear feeds weight.T -> (N, K).
        weight = (torch.randn(self.K, self.N, device=device) * 0.1).to(
            torch.float8_e4m3fn
        )
        # Per-channel weight scale, 1-D [N] (this checkpoint's layout).
        weight_scale = (
            torch.rand(self.N, dtype=torch.float32, device=device) * 0.05 + 0.01
        )
        return x, weight, weight_scale

    def _prequant(self, x):
        # Same per-token quant the standard non-tuple path uses internally, so
        # the tuple path and the reference share identical (qx, scale).
        qx, x_scale = fp8u.per_token_group_quant_fp8(x, group_size=x.shape[1])
        return qx, x_scale

    def test_fold_matches_standard_and_preserves_dtype(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                x, weight, weight_scale = self._make_case(dtype)

                ref = fp8u.apply_fp8_linear(
                    input=x,
                    weight=weight,
                    weight_scale=weight_scale,
                    use_per_token_if_dynamic=True,
                )

                qx, x_scale = self._prequant(x)
                fused = fp8u.apply_fp8_linear(
                    input=(qx, x_scale, dtype),
                    weight=weight,
                    weight_scale=weight_scale,
                )

                # dtype must be carried through (FP16 stays FP16).
                self.assertEqual(fused.dtype, dtype)
                self.assertEqual(ref.dtype, dtype)
                self.assertEqual(list(fused.shape), [self.M, self.N])

                # Same quant + same GEMM => outputs match tightly.
                torch.testing.assert_close(
                    fused.float(), ref.float(), rtol=2e-2, atol=2e-2
                )

    def test_two_tuple_defaults_dtype(self):
        # Backward compat: a 2-tuple carries no dtype -> pre_quant_output_dtype
        # (or BF16 when unset).
        x, weight, weight_scale = self._make_case(torch.float16)
        qx, x_scale = self._prequant(x)

        out_default = fp8u.apply_fp8_linear(
            input=(qx, x_scale),
            weight=weight,
            weight_scale=weight_scale,
        )
        self.assertEqual(out_default.dtype, torch.bfloat16)

        out_fp16 = fp8u.apply_fp8_linear(
            input=(qx, x_scale),
            weight=weight,
            weight_scale=weight_scale,
            pre_quant_output_dtype=torch.float16,
        )
        self.assertEqual(out_fp16.dtype, torch.float16)


if __name__ == "__main__":
    unittest.main()
