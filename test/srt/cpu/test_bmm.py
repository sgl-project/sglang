import itertools
import unittest

# TODO: use interface in cpu.py
import torch
import torch.nn as nn
from utils import precision

from sglang.srt.layers.quantization.fp8_utils import input_to_float8
from sglang.test.test_utils import CustomTestCase

torch.manual_seed(1234)


class Mod(nn.Module):
    def __init__(self, input_channel, output_channel, has_bias):
        super(Mod, self).__init__()
        self.linear = torch.nn.Linear(input_channel, output_channel, has_bias)

    def forward(self, x):
        return self.linear(x)


class TestBmm(CustomTestCase):
    M = [1, 2, 11, 111]
    N = [128 + 32, 512]
    K = [512 + 32, 128 + 32]
    B = [1, 16, 17]
    chunk = [True, False]

    def _get_bmm_inputs(self, B, M, N, K, chunk, dtype):
        if chunk:
            mat1 = (
                torch.randn(M, B, K + 64, dtype=dtype).narrow(2, 0, K).transpose_(0, 1)
            )
            mat2 = torch.randn(B, N, K, dtype=dtype).transpose_(1, 2)
            mat3 = (
                torch.randn(M, B, N + 64, dtype=dtype).narrow(2, 0, N).transpose_(0, 1)
            )
        else:
            mat1 = torch.randn(M, B, K, dtype=dtype).transpose_(0, 1)
            mat2 = torch.randn(B, N, K, dtype=dtype).transpose_(1, 2)
            mat3 = torch.randn(M, B, N, dtype=dtype).transpose_(0, 1)
        return mat1, mat2, mat3

    def _bf16_bmm(self, B, M, N, K, chunk, dtype=torch.bfloat16):
        mat1, mat2, mat3 = self._get_bmm_inputs(B, M, N, K, chunk, dtype)
        ref = torch.bmm(mat1, mat2)
        mat2_t = mat2.transpose_(1, 2)
        mat3.zero_()
        torch.ops.sgl_kernel.bmm_cpu(mat3, mat1, mat2, False, None)
        atol = rtol = precision[ref.dtype]
        torch.testing.assert_close(ref, mat3, atol=atol, rtol=rtol)

        packed_B = torch.ops.sgl_kernel.convert_weight_packed(mat2_t)
        mat3.zero_()
        torch.ops.sgl_kernel.bmm_cpu(mat3, mat1, packed_B, True, None)
        torch.testing.assert_close(ref, mat3, atol=atol, rtol=rtol)

    def _fp8_bmm(self, B, M, N, K, chunk, dtype=torch.bfloat16):
        mat1, mat2, mat3 = self._get_bmm_inputs(B, M, N, K, chunk, dtype)
        mat2_q, mat2_s = input_to_float8(mat2)
        ref = torch.bmm(mat1, mat2_q.to(torch.bfloat16)) * mat2_s
        mat2_q_t = mat2_q.transpose_(1, 2).contiguous()
        mat3.zero_()
        atol = rtol = precision[ref.dtype]
        torch.ops.sgl_kernel.bmm_cpu(mat3, mat1, mat2_q_t, False, mat2_s)
        torch.testing.assert_close(ref, mat3, atol=atol, rtol=rtol)

        packed_B_q = torch.ops.sgl_kernel.convert_weight_packed(mat2_q_t)
        mat3.zero_()
        torch.ops.sgl_kernel.bmm_cpu(mat3, mat1, packed_B_q, True, mat2_s)
        torch.testing.assert_close(ref, mat3, atol=atol, rtol=rtol)

    def test_bmm(self):
        for params in itertools.product(
            self.B,
            self.M,
            self.N,
            self.K,
            self.chunk,
        ):
            with self.subTest(
                B=params[0],
                M=params[1],
                N=params[2],
                K=params[3],
                chunk=params[4],
            ):
                self._bf16_bmm(*params)
                self._fp8_bmm(*params)


if __name__ == "__main__":
    unittest.main()
