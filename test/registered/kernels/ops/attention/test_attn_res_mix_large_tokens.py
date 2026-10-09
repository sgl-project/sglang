"""The attention-residual score/combine kernels index the [T, 8, 7168] bank with
pid_t * stride_bm, which overflows int32 at 37,451 rows."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attn_residual import _mix_fused
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_H, _NB, _NVB = 7168, 8, 7
_T = 2**31 // (_NB * _H) + 2  # 37,451: the first row count whose offset overflows


class TestAttnResMixLargeTokens(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")

    def test_rows_past_int32_offset(self):
        g = torch.Generator(device="cuda").manual_seed(0)
        prefix = torch.randn(_T, _H, generator=g, device="cuda").bfloat16()
        bank = torch.randn(_T, _NB, _H, generator=g, device="cuda").bfloat16()
        proj = SimpleNamespace(weight=torch.randn(1, _H, device="cuda") * _H**-0.5)
        norm = SimpleNamespace(
            weight=1 + 0.1 * torch.randn(_H, device="cuda"), variance_epsilon=1e-6
        )

        out = _mix_fused(prefix, bank, _NVB, proj, norm)
        torch.cuda.synchronize()

        # Rows are independent, so the tail must match a run on the tail alone,
        # whose offsets fit in int32.
        n = 4096
        tail = _mix_fused(prefix[-n:], bank[-n:], _NVB, proj, norm)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.equal(out[-n:], tail))


if __name__ == "__main__":
    unittest.main()
