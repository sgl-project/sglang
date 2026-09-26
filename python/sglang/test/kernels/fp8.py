"""Shared input/reference construction for FP8 quantization and GEMM tests."""

import torch

from sglang.srt.utils import get_device
from sglang.test.test_utils import CustomTestCase

device = get_device()


class TestFP8Base(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.M = 256
        # test non-aligned
        cls.N = 1024 + 64
        cls.K = 512
        cls.group_size = 128
        cls.quant_type = torch.float8_e4m3fn
        cls.output_type = torch.bfloat16

    @staticmethod
    def _make_A(M, K, group_size, out_dtype):
        quant_A = torch.rand(
            M, K // group_size, group_size, dtype=torch.float32, device=device
        )
        # -1 ~ 1
        quant_A = quant_A * 2 - 1
        # scaling abs max to fmax
        finfo = torch.finfo(out_dtype)
        fmax = finfo.max
        scaling = fmax / quant_A.abs().amax(-1, keepdim=True)
        quant_A *= scaling
        quant_A = quant_A.to(out_dtype).to(torch.float32)

        # create scale and A
        scale = torch.rand(M, K // group_size, dtype=torch.float32, device=device)
        scale /= fmax
        A = quant_A * scale[..., None]

        A = A.reshape(M, K)
        quant_A = quant_A.reshape(M, K).to(out_dtype)
        return A, quant_A, scale

    @staticmethod
    def _make_B(K, N, group_size, out_dtype):
        def _aligned_size(a, b):
            return (a + b - 1) // b * b

        K_aligned = _aligned_size(K, group_size)
        N_aligned = _aligned_size(N, group_size)

        quant_B = torch.rand(
            K_aligned // group_size,
            group_size,
            N_aligned // group_size,
            group_size,
            dtype=torch.float32,
            device=device,
        )
        quant_B = quant_B * 2 - 1

        # scaling abs max to fmax
        finfo = torch.finfo(out_dtype)
        fmax = finfo.max
        scaling = fmax / quant_B.abs().amax((1, 3), keepdim=True)
        quant_B *= scaling
        quant_B = quant_B.to(out_dtype).to(torch.float32)

        scale = torch.rand(
            K_aligned // group_size,
            1,
            N_aligned // group_size,
            1,
            dtype=torch.float32,
            device=device,
        )
        scale /= fmax

        B = quant_B * scale

        B = B.reshape(K_aligned, N_aligned)[:K, :N]
        quant_B = quant_B.reshape(K_aligned, N_aligned).to(out_dtype)[:K, :N]
        scale = scale.reshape(K_aligned // group_size, N_aligned // group_size)
        return B, quant_B, scale
