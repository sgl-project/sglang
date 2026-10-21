"""Quark W4A4 MXFP4 dense GEMM on gfx950 when K is not a whole number of AITER's tuned BLOCK_SIZE_K: outputs must match a dequantized reference."""

import unittest

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

# (N, K, M): AITER's config for each of these takes the masked K-tail path, which
# returned wrong elements before the even-K narrowing. K=768 is the Kimi-K3 TP8
# shared-expert down_proj; N=7168, K=1536 has no tuned file and uses DEFAULT.json.
CASES = [
    (7168, 768, 96),
    (7168, 768, 128),
    (7168, 768, 256),
    (7168, 1536, 16),
    (7168, 1536, 32),
]


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "gfx950 MXFP4 GEMM")
class TestQuarkAfp4wfp4KTail(CustomTestCase):
    def test_matches_dequantized_reference(self):
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.utility.fp4_utils import e8m0_to_f32, mxfp4_to_f32

        from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4 import (
            gemm_afp4wfp4,
        )

        def dequant(q, scale):
            values = mxfp4_to_f32(q)
            scale = e8m0_to_f32(scale).repeat_interleave(32, dim=-1)
            return values * scale[:, : values.shape[-1]]

        torch.manual_seed(0)
        for n, k, m in CASES:
            with self.subTest(n=n, k=k, m=m):
                w_q, w_s = dynamic_mxfp4_quant(
                    (torch.randn(n, k, device="cuda") * 0.05).to(torch.bfloat16)
                )
                x_q, x_s = dynamic_mxfp4_quant(
                    (torch.randn(m, k, device="cuda") * 0.05).to(torch.bfloat16)
                )
                ref = dequant(x_q, x_s) @ dequant(w_q, w_s).t()
                # NaN-filled output so an element the GEMM leaves stale fails too.
                y = torch.full(
                    (m, n), float("nan"), dtype=torch.bfloat16, device="cuda"
                )
                gemm_afp4wfp4(x_q, w_q, x_s, w_s, torch.bfloat16, y)
                err = (y.float() - ref).abs().nan_to_num(float("inf"))
                self.assertLessEqual(err.max().item(), 0.02 * ref.abs().max().item())


if __name__ == "__main__":
    unittest.main()
