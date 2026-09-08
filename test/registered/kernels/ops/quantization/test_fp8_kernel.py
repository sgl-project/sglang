import unittest

import torch

from sglang.kernels.ops.quantization.fp8_kernel import (
    per_token_group_quant_fp8,
    sglang_per_token_quant_fp8,
)
from sglang.srt.utils import is_cuda
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.fp8 import TestFP8Base, device

register_cuda_ci(est_time=4, stage="base-b", runner_config="1-gpu-large")


_is_cuda = is_cuda()


class TestPerTokenGroupQuantFP8(TestFP8Base):
    def test_per_token_quant_fp8_empty_input(self):
        x = torch.empty((0, 3072), dtype=torch.bfloat16, device=device)

        x_q, x_s = sglang_per_token_quant_fp8(x)
        torch.cuda.synchronize()

        self.assertEqual(x_q.shape, x.shape)
        self.assertEqual(x_s.shape, (0, 1))

    def test_per_token_group_quant_fp8(self):
        if _is_cuda and torch.cuda.get_device_capability()[0] < 9:
            return

        A, A_quant_gt, scale_gt = self._make_A(
            M=self.M, K=self.K, group_size=self.group_size, out_dtype=self.quant_type
        )
        A_quant, scale = per_token_group_quant_fp8(
            x=A.to(torch.bfloat16), group_size=self.group_size
        )
        torch.testing.assert_close(scale, scale_gt)
        diff = (A_quant.to(torch.float16) - A_quant_gt.to(torch.float16)).abs()
        diff_count = (diff > 1e-5).count_nonzero()
        assert diff_count / diff.numel() < 1e-4


if __name__ == "__main__":
    unittest.main()
