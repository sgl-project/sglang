"""Block-scaled FP8 and INT8 Triton matmuls must address rows past 2^31 elements of A
and of C correctly; int32 row offsets used to wrap and fault."""

import unittest

import torch
import triton

from sglang.kernels.ops.gemm.fp8_kernel import w8a8_block_fp8_matmul_triton
from sglang.srt.layers.quantization.int8_utils import apply_w8a8_block_int8_linear
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

BLOCK = [128, 128]
# One row past 2^15, so with 2^16 columns the last row starts beyond 2^31 elements
M = 2**15 + 1
WIDE = 2**16

# Exact e4m3 encodings: 0x08 is 2^-6 and 0x38 is 1.0
FP8_2_POW_NEG6 = 0x08
FP8_ONE = 0x38


def _fp8_full(rows: int, cols: int, code: int) -> torch.Tensor:
    # Filling uint8 and viewing avoids a large fp32 temporary
    return torch.full((rows, cols), code, dtype=torch.uint8, device="cuda").view(
        torch.float8_e4m3fn
    )


def _unit_scales(m: int, n: int, k: int):
    a_s = torch.ones(m, k // BLOCK[1], device="cuda")
    b_s = torch.ones(triton.cdiv(n, BLOCK[0]), k // BLOCK[1], device="cuda")
    return a_s, b_s


class TestBlockMatmulLargeM(CustomTestCase):
    def test_fp8_rows_of_a_past_2_pow_31(self):
        n, k = 16, WIDE
        a = _fp8_full(M, k, FP8_2_POW_NEG6)
        b = _fp8_full(n, k, FP8_ONE)
        a_s, b_s = _unit_scales(M, n, k)
        out = w8a8_block_fp8_matmul_triton(
            A=a, B=b, As=a_s, Bs=b_s, block_size=BLOCK, output_dtype=torch.float32
        )
        # 2^16 products of 2^-6 * 1.0
        self.assertTrue((out[-1] == 1024.0).all().item())

    def test_fp8_rows_of_c_past_2_pow_31(self):
        n, k = WIDE, 128
        a = _fp8_full(M, k, FP8_ONE)
        b = _fp8_full(n, k, FP8_ONE)
        a_s, b_s = _unit_scales(M, n, k)
        out = w8a8_block_fp8_matmul_triton(
            A=a, B=b, As=a_s, Bs=b_s, block_size=BLOCK, output_dtype=torch.bfloat16
        )
        self.assertTrue((out[-1] == 128.0).all().item())

    def test_int8_linear_rows_of_a_past_2_pow_31(self):
        # Serving path: per-token-group activation quantization, then the matmul
        n, k = 16, WIDE
        x = torch.ones(M, k, dtype=torch.bfloat16, device="cuda")
        w = torch.ones(n, k, dtype=torch.int8, device="cuda")
        w_s = torch.ones(triton.cdiv(n, BLOCK[0]), k // BLOCK[1], device="cuda")
        out = apply_w8a8_block_int8_linear(
            input=x, weight=w, block_size=BLOCK, weight_scale=w_s
        )
        # Each 128-element group quantizes 1.0 to 127 with scale 1/127
        torch.testing.assert_close(
            out[-1].float(),
            torch.full((n,), float(WIDE), device="cuda"),
            rtol=1e-3,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
