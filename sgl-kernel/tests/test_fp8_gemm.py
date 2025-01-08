import unittest

import torch
from vllm._custom_ops import cutlass_scaled_mm as vllm_scaled_mm
from vllm._custom_ops import scaled_fp8_quant as vllm_scaled_fp8_quant
from sgl_kernel import fp8_scaled_mm


def torch_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias):
    o = torch.matmul(a.to(torch.float32), b.to(torch.float32))

    o = o.to(torch.float32)
    temp1 = o * scale_a.view(-1, 1)
    temp2 = temp1 * scale_b.view(1, -1)
    final = temp2.to(out_dtype)

    return final


class TestFp8Gemm(unittest.TestCase):
    def _test_accuracy_once(self, M, N, K, with_bias, out_dtype, device):
        a = torch.randn((M, K), device=device)
        b = torch.randn((N, K), device=device)

        scale_a = torch.randn((M,), device="cuda", dtype=torch.float32) * 0.01
        scale_b = torch.randn((N,), device="cuda", dtype=torch.float32) * 0.01
        if with_bias:
            # bias = torch.ones((N,), device="cuda", dtype=out_dtype) * 10
            bias = torch.randn((N,), device="cuda", dtype=out_dtype)
        else:
            bias = None
        o1 = torch.empty((a.shape[0], b.shape[1]), device="cuda", dtype=torch.bfloat16)
        b_fp8, scale_b_fp8 = vllm_scaled_fp8_quant(b, scale_b)
        b_fp8 = b_fp8.t()
        a_fp8, scale_a_fp8 = vllm_scaled_fp8_quant(a, scale_a)
        o = torch_scaled_mm(a_fp8, b_fp8, scale_a_fp8, scale_b_fp8, out_dtype, bias)
        # o1 = vllm_scaled_mm(a_fp8, b_fp8, scale_a_fp8, scale_b_fp8, out_dtype, bias)
        o1 = fp8_scaled_mm(a_fp8, b_fp8, scale_a_fp8, scale_b_fp8, out_dtype, bias)
        max_val = max(o.abs().max().item(), o1.abs().max().item())
        rtol = 4e-3
        atol = max_val * rtol
        torch.testing.assert_close(o, o1, rtol=rtol, atol=atol)
        print(f"M={M}, N={N}, K={K}, with_bias={with_bias}, out_dtype={out_dtype}: OK")

    def test_accuracy(self):
        Ms = [1, 128, 512, 1024, 4096]
        Ns = [16, 128, 512, 1024, 4096]
        Ks = [512, 1024, 4096, 8192, 16384]
        # Ms = [128]
        # Ns = [512]
        # Ks = [4096]
        # bias_opts = [True, False]
        bias_opts = [False]
        out_dtypes = [torch.bfloat16, torch.float16]
        # out_dtypes = [torch.float16]
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
