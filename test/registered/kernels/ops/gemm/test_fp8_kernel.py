import unittest

import torch

from sglang.kernels.ops.gemm.fp8_kernel import w8a8_block_fp8_matmul
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-large")

from sglang.srt.utils import get_device, is_cuda, is_xpu

_is_cuda = is_cuda()
_is_xpu = is_xpu()

device = get_device()


from sglang.test.kernels.fp8 import TestFP8Base


class TestW8A8BlockFP8Matmul(TestFP8Base):
    def test_w8a8_block_fp8_matmul(self):
        if _is_cuda and torch.cuda.get_device_capability()[0] < 9:
            return
        elif _is_xpu:
            # XPU doesn't provide traditional capability info like CUDA
            pass
        else:
            return

        A, A_quant_gt, A_scale_gt = self._make_A(
            M=self.M, K=self.K, group_size=self.group_size, out_dtype=self.quant_type
        )
        B, B_quant_gt, B_scale_gt = self._make_B(
            K=self.K, N=self.N, group_size=self.group_size, out_dtype=self.quant_type
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
