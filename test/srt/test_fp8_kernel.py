import unittest

import torch

from sglang.srt.layers.quantization.fp8_kernel import (
    per_token_group_quant_fp8,
    w8a8_block_fp8_matmul,
)
from sglang.srt.utils import get_device
from sglang.test.test_utils import CustomTestCase


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
        cls.device = get_device()

    @staticmethod
    def _make_A(M, K, group_size, out_dtype, device):
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
    def _make_B(K, N, group_size, out_dtype, device):
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


class TestPerTokenGroupQuantFP8(TestFP8Base):
    def test_per_token_group_quant_fp8(self):
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 9:
            return
        A, A_quant_gt, scale_gt = self._make_A(
            M=self.M,
            K=self.K,
            group_size=self.group_size,
            out_dtype=self.quant_type,
            device=self.device,
        )
        A_quant, scale = per_token_group_quant_fp8(x=A, group_size=self.group_size)
        torch.testing.assert_close(scale, scale_gt)
        diff = (A_quant.to(torch.float16) - A_quant_gt.to(torch.float16)).abs()
        diff_count = (diff > 1e-5).count_nonzero()
        assert diff_count / diff.numel() < 1e-4


class TestW8A8BlockFP8Matmul(TestFP8Base):
    def test_w8a8_block_fp8_matmul(self):
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 9:
            return
        A, A_quant_gt, A_scale_gt = self._make_A(
            M=self.M,
            K=self.K,
            group_size=self.group_size,
            out_dtype=self.quant_type,
            device=self.device,
        )
        B, B_quant_gt, B_scale_gt = self._make_B(
            K=self.K,
            N=self.N,
            group_size=self.group_size,
            out_dtype=self.quant_type,
            device=self.device,
        )
        C_gt = A.to(self.output_type) @ B.to(self.output_type)
        C = w8a8_block_fp8_matmul(
            A=A_quant_gt,
            B=B_quant_gt.T.contiguous(),
            As=A_scale_gt,
            Bs=B_scale_gt.T.contiguous(),
            block_size=[128, 128],
            output_dtype=self.output_type,
        )
        torch.testing.assert_close(C, C_gt, atol=0.5, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
