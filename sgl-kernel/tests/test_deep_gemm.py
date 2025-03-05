import unittest
from typing import Optional, Type

import torch
from sgl_kernel import fp8_blockwise_scaled_mm
from sgl_kernel import gemm_fp8_fp8_bf16_nt


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


def scale_shape(shape, group_shape):
    assert len(shape) == len(group_shape)
    return tuple(cdiv(shape[i], group_shape[i]) for i in range(len(group_shape)))


class TestFp8Gemm(unittest.TestCase):
    def _test_accuracy_once(self, M, N, K, out_dtype, device):
        fp8_info = torch.finfo(torch.float8_e4m3fn)
        fp8_max, fp8_min = fp8_info.max, fp8_info.min

        a_fp32 = (
            (torch.rand(M, K, dtype=torch.float32, device=device) - 0.5) * 2 * fp8_max
        )
        a_fp8 = a_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        b_fp32 = (
            (torch.rand(N, K, dtype=torch.float32, device=device) - 0.5) * 2 * fp8_max
        )
        b_fp8 = b_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn).t()

        scale_a_group_shape = (1, 128)
        scale_b_group_shape = (128, 128)
        scale_a_shape = scale_shape(a_fp8.shape, scale_a_group_shape)
        scale_b_shape = scale_shape(b_fp8.shape, scale_b_group_shape)

        scale_a = torch.randn(scale_a_shape, device=device, dtype=torch.float32) * 0.001
        scale_b = torch.randn(scale_b_shape, device=device, dtype=torch.float32) * 0.001
        scale_a = scale_a.t().contiguous().t()
        scale_b = scale_b.t().contiguous().t()

        o1 = fp8_blockwise_scaled_mm(a_fp8, b_fp8, scale_a, scale_b, out_dtype)
        o2 = gemm_fp8_fp8_bf16_nt(a_fp8, b_fp8, scale_a, scale_b)

        rtol = 0.02
        atol = 1
        torch.testing.assert_close(o, o1, rtol=rtol, atol=atol)
        print(f"M={M}, N={N}, K={K}, out_dtype={out_dtype}: OK")

    def test_accuracy(self):
        Ms = [1, 128, 512, 1024, 4096]
        NKs = [
            # (36, 7168),
            (1536, 7168),
            (1536, 1536),
            (24576, 7168),
            (2048, 512),
            (7168, 1024),
            (2304, 7168),
            (256, 7168),
            (7168, 1152),
            (7168, 256),
        ]
        out_dtypes = [torch.bfloat16]
        for M in Ms:
            for N, K in NKs:
                    for out_dtype in out_dtypes:
                        self._test_accuracy_once(M, N, K, out_dtype, "cuda")


if __name__ == "__main__":
    unittest.main()
