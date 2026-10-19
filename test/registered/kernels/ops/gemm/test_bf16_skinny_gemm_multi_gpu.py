"""The BF16 skinny GEMM launches on its operands' device, not the current one."""

import unittest

import torch

from sglang.kernels.ops.gemm.bf16_skinny_gemm import bf16_skinny_gemm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="2-gpu-large")


def _error_bound(x, w):
    """FP64 result and the bound for an FP32-accumulated dot product rounded once to
    BF16, as in test_bf16_skinny_gemm.py."""
    ref = x.double() @ w.double().T
    k = x.shape[1]
    gamma = k * 2.0**-24 / (1 - k * 2.0**-24)
    ulp = torch.exp2(torch.floor(torch.log2(ref.abs().clamp_min(2.0**-126))) - 7)
    return ref, gamma * (x.double().abs() @ w.double().abs().T) + ulp


@unittest.skipUnless(
    torch.cuda.device_count() >= 2 and torch.version.hip is None, "needs two CUDA GPUs"
)
class TestBf16SkinnyGemmMultiGpu(CustomTestCase):
    def test_operands_on_a_non_current_device(self):
        g = torch.Generator(device="cuda:1").manual_seed(6)
        x = torch.randn(6, 256, device="cuda:1", generator=g).bfloat16()
        w = (torch.randn(1000, 256, device="cuda:1", generator=g) / 16).bfloat16()
        ref, bound = _error_bound(x, w)
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        with torch.cuda.device(0):
            # Keep device 0 busy, so a launch on its stream instead of device 1's
            # would not have run when device 1 is synchronized below.
            torch.cuda._sleep(2**31)
            out = bf16_skinny_gemm(x, w)
        torch.cuda.synchronize(1)
        self.assertEqual(out.device, x.device)
        self.assertTrue(bool(((out.double() - ref).abs() <= bound).all()))
        torch.cuda.synchronize(0)


if __name__ == "__main__":
    unittest.main()
