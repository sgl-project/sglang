import itertools
import unittest
from typing import Optional

import torch
import torch.testing

from sglang.srt.layers.quantization.fp8_kernel import triton_scaled_mm
from sglang.test.test_utils import CustomTestCase


def torch_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference implementation using float32 for stability"""
    out = torch.mm(a.to(torch.float32), b.to(torch.float32))
    out = scale_a.to(torch.float32) * out * scale_b.to(torch.float32).T
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out.to(out_dtype)


class TestScaledMM(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("This test requires a CUDA device.")
        torch.set_default_device("cuda")

    def _make_inputs(self, M, K, N, in_dtype):
        if in_dtype == torch.int8:
            a = torch.randint(-8, 8, (M, K), dtype=in_dtype, device="cuda")
            b = torch.randint(-8, 8, (K, N), dtype=in_dtype, device="cuda")
        else:  # fp8
            a = torch.clamp(
                0.1 * torch.randn((M, K), dtype=torch.float16, device="cuda"), -0.3, 0.3
            ).to(in_dtype)
            b = torch.clamp(
                0.1 * torch.randn((K, N), dtype=torch.float16, device="cuda"), -0.3, 0.3
            ).to(in_dtype)
        return a, b

    def test_basic_cases(self):
        """Test core functionality with reduced precision requirements"""
        test_configs = [
            (32, 32, 32, torch.int8, torch.float16, False),
            (64, 64, 64, torch.int8, torch.float16, True),
        ]

        try:
            torch.tensor([1.0], dtype=torch.float8_e4m3fn, device="cuda")
            test_configs.append((32, 32, 32, torch.float8_e4m3fn, torch.float16, False))
        except:
            print("FP8 not supported, skipping")

        for M, K, N, in_dtype, out_dtype, with_bias in test_configs:
            with self.subTest(M=M, K=K, N=N, dtype=in_dtype, bias=with_bias):
                print(f"Currently testing with in_dtype: {in_dtype}")
                torch.manual_seed(42)

                input, weight = self._make_inputs(M, K, N, in_dtype)
                scale_a = 0.1 + 0.05 * torch.rand(
                    (M, 1), dtype=torch.float32, device="cuda"
                )
                scale_b = 0.1 + 0.05 * torch.rand(
                    (N, 1), dtype=torch.float32, device="cuda"
                )
                bias = (
                    0.01 * torch.randn((M, N), dtype=out_dtype, device="cuda")
                    if with_bias
                    else None
                )

                triton_out = triton_scaled_mm(
                    input, weight, scale_a, scale_b, out_dtype, bias
                )
                ref_out = torch_scaled_mm(
                    input, weight, scale_a, scale_b, out_dtype, bias
                )

                # Use relaxed tolerances
                rtol = 0.15 if in_dtype == torch.int8 else 0.25
                atol = 0.1 if in_dtype == torch.int8 else 0.15

                torch.testing.assert_close(triton_out, ref_out, rtol=rtol, atol=atol)


if __name__ == "__main__":
    unittest.main(verbosity=2)
