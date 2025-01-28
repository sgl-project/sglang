import unittest

import torch
from sgl_kernel import int8_scaled_mm
from vllm._custom_ops import cutlass_scaled_mm as vllm_scaled_mm


def to_int8(tensor: torch.Tensor) -> torch.Tensor:
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)


def torch_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias):
    o = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    if bias is not None:
        o = o.to(torch.float32) * scale_a.view(-1, 1) * scale_b.view(1, -1) + bias
    else:
        o = o.to(torch.float32) * scale_a.view(-1, 1) * scale_b.view(1, -1)
    return o.to(out_dtype)


class TestInt8Gemm(unittest.TestCase):
    def _test_accuracy_once(self, M, N, K, with_bias, out_dtype, device):
        a = to_int8(torch.randn((M, K), device=device) * 5)
        b = to_int8(torch.randn((N, K), device=device).t() * 5)
        scale_a = torch.randn((M,), device="cuda", dtype=torch.float32)
        scale_b = torch.randn((N,), device="cuda", dtype=torch.float32)
        if with_bias:
            bias = torch.randn((N,), device="cuda", dtype=out_dtype) * 10
        else:
            bias = None

        o = int8_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)
        o1 = torch_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)
        o2 = vllm_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)
        torch.testing.assert_close(o, o1)
        torch.testing.assert_close(o, o2)
        print(f"M={M}, N={N}, K={K}, with_bias={with_bias}, out_dtype={out_dtype}: OK")

    def test_accuracy(self):
        Ms = [1, 128, 512, 1024, 4096, 8192]
        Ns = [16, 128, 512, 1024, 4096, 8192, 16384]
        Ks = [512, 1024, 4096, 8192, 16384]
        bias_opts = [True, False]
        out_dtypes = [torch.float16, torch.bfloat16]
        for M in Ms:
            for N in Ns:
                for K in Ks:
                    for with_bias in bias_opts:
                        for out_dtype in out_dtypes:
                            self._test_accuracy_once(
                                M, N, K, with_bias, out_dtype, "cuda"
                            )


if __name__ == "__main__":
    unittest.main()
