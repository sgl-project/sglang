"""Block-fp8 matmul on gfx115x (RDNA3.5): the default tiles against a reference.

gfx115x launches the bf16-upcast kernel with tiles narrower than the 128 x 128
scale block (16 x 64, 32 x 64 or 64 x 32 by row count), so several N tiles share
one row of weight scales and ragged M, N and K go through the load masks. Which
tile and kernel are launched is checked on CPU in
test/registered/unit/kernels/test_block_fp8_matmul_dispatch.py.
"""

import unittest
from unittest import mock

import torch

from sglang.kernels.ops.gemm import fp8_kernel
from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype, fp8_max
from sglang.srt.utils import use_gfx115_kernels
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="base-b-test-1-gpu-small-amd")

BLOCK = 128
# N is neither a multiple of the N tiles (64, 32) nor of the scale block, and
# K ends in a partial scale block.
N, K = 336, 400
TOLERANCE = 5e-3


def _quantize(values: torch.Tensor, *, rows: int, cols: int = BLOCK) -> tuple:
    """fp32 [n, k] -> fp8 codes, fp32 scale per (rows x cols) block, dequantized."""
    n, k = values.shape
    n_blocks, k_blocks = -(-n // rows), -(-k // cols)
    padded = torch.zeros(n_blocks * rows, k_blocks * cols, device=values.device)
    padded[:n, :k] = values
    blocks = padded.reshape(n_blocks, rows, k_blocks, cols)
    scale = (blocks.abs().amax(dim=(1, 3)) / fp8_max).clamp_min(1e-10)
    scaled = (blocks / scale[:, None, :, None]).reshape(n_blocks * rows, -1)
    codes = scaled[:n, :k].to(fp8_dtype)
    expanded = scale.repeat_interleave(rows, dim=0).repeat_interleave(cols, dim=1)
    return codes.contiguous(), scale.contiguous(), codes.float() * expanded[:n, :k]


@unittest.skipUnless(use_gfx115_kernels(), "needs a gfx115x GPU with its kernels on")
class TestBlockFp8MatmulGfx115(CustomTestCase):
    def test_default_tiles_match_the_dequantized_product(self):
        # The gate is read at import; it is not patched here.
        self.assertTrue(fp8_kernel._is_gfx115)
        g = torch.Generator(device="cuda").manual_seed(0)
        weight = torch.randn(N, K, device="cuda", generator=g) / K**0.5
        weight_q, weight_scale, weight_ref = _quantize(weight, rows=BLOCK)
        # No tuned config file: the default tiles are what is under test.
        with mock.patch.object(
            fp8_kernel, "get_w8a8_block_fp8_configs", return_value=None
        ):
            # One row count per tile, plus 70, which is not a multiple of the row tile;
            # the other two output types once.
            cases = [(num_rows, torch.bfloat16) for num_rows in (1, 17, 33, 70)]
            cases += [(17, torch.float16), (17, torch.float32)]
            for num_rows, output_dtype in cases:
                with self.subTest(num_rows=num_rows, output_dtype=output_dtype):
                    x = torch.randn(num_rows, K, device="cuda", generator=g)
                    x_q, x_scale, x_ref = _quantize(x, rows=1)
                    out = fp8_kernel.w8a8_block_fp8_matmul_triton(
                        x_q,
                        weight_q,
                        x_scale,
                        weight_scale,
                        [BLOCK, BLOCK],
                        output_dtype=output_dtype,
                    )
                    self.assertEqual(out.dtype, output_dtype)
                    reference = x_ref @ weight_ref.T
                    error = (out.float() - reference).norm() / reference.norm()
                    self.assertLess(error.item(), TOLERANCE)
            # A 64 x 32 scale block: the N and K scale steps differ, and the rows
            # of the weight get scales that differ by block (x 1, x 2, x 3, ...).
            with self.subTest(block=(64, 32)):
                ramp = 1.0 + torch.arange(N, device="cuda") // 64
                weight_q, weight_scale, weight_ref = _quantize(
                    weight * ramp[:, None], rows=64, cols=32
                )
                x = torch.randn(17, K, device="cuda", generator=g)
                x_q, x_scale, x_ref = _quantize(x, rows=1, cols=32)
                out = fp8_kernel.w8a8_block_fp8_matmul_triton(
                    x_q, weight_q, x_scale, weight_scale, [64, 32], torch.bfloat16
                )
                reference = x_ref @ weight_ref.T
                error = (out.float() - reference).norm() / reference.norm()
                self.assertLess(error.item(), TOLERANCE)


if __name__ == "__main__":
    unittest.main()
