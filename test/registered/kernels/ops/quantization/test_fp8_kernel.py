import unittest

import torch

from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-large")

from sglang.srt.utils import get_device, is_cuda, is_xpu

_is_cuda = is_cuda()
_is_xpu = is_xpu()

device = get_device()


from sglang.test.kernels.fp8 import TestFP8Base


class TestPerTokenGroupQuantFP8(TestFP8Base):
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
