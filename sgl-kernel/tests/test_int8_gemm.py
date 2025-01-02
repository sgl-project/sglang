import unittest

import torch
from sgl_kernel import int8_scaled_mm
from vllm._custom_ops import cutlass_scaled_mm as vllm_scaled_mm


def to_int8(tensor: torch.Tensor) -> torch.Tensor:
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)


class TestInt8Gemm(unittest.TestCase):
    def _test_accuracy_once(self, M, N, K, with_bias, out_dtype, device):
        a = to_int8(torch.randn((M, K), device=device) * 5)
        b = to_int8(torch.randn((N, K), device=device).t() * 5)
        o = torch.empty((M, N), device=device, dtype=out_dtype)
        scale_a = torch.ones((M,), device="cuda", dtype=torch.float32)
        scale_b = torch.ones((N,), device="cuda", dtype=torch.float32)
        if with_bias:
            bias = torch.zeros((N,), device="cuda", dtype=out_dtype)
        else:
            bias = None

        int8_scaled_mm(o, a, b, scale_a, scale_b, bias)
        print(o)
        o1 = vllm_scaled_mm(a, b, scale_a, scale_b, out_dtype)
        print(o1)
        self.assertTrue(torch.allclose(o, o1))

    def test_accuracy(self):
        M, N, K = 1024, 2048, 1024
        out_dtypes = [torch.float16, torch.bfloat16]
        for out_dtype in out_dtypes:
            self._test_accuracy_once(M, N, K, False, out_dtype, "cuda")


if __name__ == "__main__":
    unittest.main()
