"""Unit tests for the row-padded quant path of the cutlass FP8 blockwise linear.

`cutlass_w8a8_block_fp8_linear_with_fallback` quantizes activations into
row-aligned buffers (`sglang_per_token_group_quant_fp8_row_padded`) so the
`fp8_blockwise_scaled_mm` wrapper's per-call mat_a/scales_a padding short-
circuits. These tests pin the invariant that this is numerically identical to
the legacy unpadded path, across both row-aligned and unaligned M.
"""

import unittest

import torch

from sglang.srt.layers.quantization.fp8_kernel import (
    fp8_dtype,
    per_token_group_quant_fp8,
    sglang_per_token_group_quant_fp8_row_padded,
)
from sglang.srt.layers.quantization.fp8_utils import (
    _check_cutlass_block_fp8_hardware_support,
    cutlass_w8a8_block_fp8_linear_with_fallback,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-large")

_FP8_MAX = torch.finfo(fp8_dtype).max
_BLOCK = 128
# Cover M == 1 (greedy decode), small unaligned M (speculative draft tokens),
# the 4-row alignment boundary, and a large aligned batch.
_M_VALUES = [1, 2, 3, 4, 5, 7, 13, 16, 31, 64, 256]


def _quant_weight_blockwise(weight_bf16: torch.Tensor, block: int = _BLOCK):
    """Block-quantize a (N, K) bf16 weight to fp8 with (N//block, K//block) fp32 scales."""
    n, k = weight_bf16.shape
    assert n % block == 0 and k % block == 0
    w = weight_bf16.float().reshape(n // block, block, k // block, block)
    amax = w.abs().amax(dim=(1, 3)).clamp(min=1e-12)  # (N//block, K//block)
    scale = amax / _FP8_MAX
    wq = (w / scale[:, None, :, None]).clamp(-_FP8_MAX, _FP8_MAX).to(fp8_dtype)
    return wq.reshape(n, k), scale.to(torch.float32)


def _legacy_cutlass_linear(x_2d, weight, weight_scale):
    """The pre-optimization path: unpadded quant, wrapper pads internally."""
    from sgl_kernel import fp8_blockwise_scaled_mm

    q_input, x_scale = per_token_group_quant_fp8(x_2d, _BLOCK, column_major_scales=True)
    return fp8_blockwise_scaled_mm(
        q_input, weight.T, x_scale, weight_scale.T, out_dtype=x_2d.dtype
    )


@unittest.skipUnless(
    _check_cutlass_block_fp8_hardware_support(),
    "cutlass block FP8 requires Hopper (SM90) or newer",
)
class TestFP8BlockwiseRowPadding(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.K = 512
        cls.N = 256
        torch.manual_seed(0)

    def test_quant_buffers_row_aligned(self):
        """Row-padded quant returns 4-aligned, M-major buffers whose live rows
        match the legacy column-major quant bit-for-bit."""
        for m in _M_VALUES:
            x = torch.randn(m, self.K, device="cuda", dtype=torch.bfloat16) * 0.1
            xq, xs = sglang_per_token_group_quant_fp8_row_padded(x, _BLOCK)
            m_pad = (m + 3) // 4 * 4

            self.assertEqual(xq.shape, (m_pad, self.K), f"M={m}")
            self.assertEqual(xs.shape[0], m_pad, f"M={m}")
            # scales_a must stay M-major (stride(0) == 1) for the kernel contract.
            self.assertEqual(xs.stride(0), 1, f"M={m}")

            xq_ref, xs_ref = per_token_group_quant_fp8(
                x, _BLOCK, column_major_scales=True
            )
            self.assertEqual(xq_ref.shape, (m, self.K), f"M={m}")
            # Live rows are produced by the same kernel, so they must be identical.
            self.assertTrue(
                torch.equal(xq[:m].view(torch.uint8), xq_ref.view(torch.uint8)),
                f"quantized activation mismatch at M={m}",
            )
            torch.testing.assert_close(xs[:m], xs_ref, atol=0.0, rtol=0.0)

    def test_gemm_bit_exact_vs_legacy(self):
        """The full linear (row-padded) is bit-identical to the legacy unpadded GEMM."""
        weight_bf16 = (
            torch.randn(self.N, self.K, device="cuda", dtype=torch.bfloat16) * 0.1
        )
        weight, weight_scale = _quant_weight_blockwise(weight_bf16)

        for m in _M_VALUES:
            x = torch.randn(m, self.K, device="cuda", dtype=torch.bfloat16) * 0.1

            out_ref = _legacy_cutlass_linear(x, weight, weight_scale)
            out_new = cutlass_w8a8_block_fp8_linear_with_fallback(
                input=x,
                weight=weight,
                block_size=[_BLOCK, _BLOCK],
                weight_scale=weight_scale,
            )

            self.assertEqual(out_new.shape, (m, self.N), f"M={m}")
            self.assertTrue(
                torch.equal(out_ref, out_new),
                f"row-padded GEMM differs from legacy at M={m}: "
                f"max_abs_diff={(out_ref.float() - out_new.float()).abs().max().item()}",
            )

    def test_linear_matches_bf16_reference(self):
        """Sanity: the FP8 linear stays close to a bf16 reference matmul."""
        weight_bf16 = (
            torch.randn(self.N, self.K, device="cuda", dtype=torch.bfloat16) * 0.1
        )
        weight, weight_scale = _quant_weight_blockwise(weight_bf16)

        for m in [1, 5, 64]:
            x = torch.randn(m, self.K, device="cuda", dtype=torch.bfloat16) * 0.1
            ref = (x.float() @ weight_bf16.float().T).to(torch.bfloat16)
            out = cutlass_w8a8_block_fp8_linear_with_fallback(
                input=x,
                weight=weight,
                block_size=[_BLOCK, _BLOCK],
                weight_scale=weight_scale,
            )
            torch.testing.assert_close(out, ref, atol=0.5, rtol=0.1)


if __name__ == "__main__":
    unittest.main()
