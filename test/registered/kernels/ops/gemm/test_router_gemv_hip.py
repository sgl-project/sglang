"""The ROCm bf16 x bf16 -> fp32 GEMMs of DeepSeek-V4 keep fp32 accuracy."""

import unittest

import torch

from sglang.kernels.ops.gemm.bf16_fp32 import linear_bf16_fp32
from sglang.kernels.ops.gemm.router_gemv_hip import (
    ROCM_ROUTER_MAX_TOKENS,
    rocm_router_gemv_split_k,
    rocm_router_reduce_partials,
)
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=5, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

NUM_EXPERTS, HIDDEN = 384, 5120


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "ROCm gfx95 only")
class TestRouterGemvHip(CustomTestCase):
    def test_gemv_accuracy_and_batch_invariance(self):
        """The split-K GEMV is within fp32 rounding of fp64, and every M runs the same
        16-row tile, so a row's result does not depend on the batch."""
        gen = torch.Generator(device="cuda").manual_seed(0)
        weight = (
            torch.randn(NUM_EXPERTS, HIDDEN, device="cuda", generator=gen) * 0.02
        ).to(torch.bfloat16)
        x = torch.randn(
            ROCM_ROUTER_MAX_TOKENS, HIDDEN, device="cuda", generator=gen
        ).to(torch.bfloat16)
        ref = (x.double() @ weight.double().T).float()
        full = torch.empty(ROCM_ROUTER_MAX_TOKENS, NUM_EXPERTS, device="cuda")
        rocm_router_reduce_partials(rocm_router_gemv_split_k(x, weight), full)
        self.assertTrue(torch.allclose(full, ref, atol=2e-3, rtol=1e-4))
        part = torch.empty(17, NUM_EXPERTS, device="cuda")
        rocm_router_reduce_partials(rocm_router_gemv_split_k(x[:17], weight), part)
        self.assertTrue(torch.equal(part, full[:17]))

    def test_linear_bf16_fp32_is_not_rounded_to_bf16(self):
        """linear_bf16_fp32 feeds the V4 compressor and router logits in fp32; a route
        that rounds its output to bf16 on ROCm lands far outside fp32 accumulation error."""
        torch.manual_seed(0)
        x = torch.randn(37, HIDDEN, device="cuda", dtype=torch.bfloat16)
        w = (torch.randn(512, HIDDEN, device="cuda") * 0.02).to(torch.bfloat16)
        out = linear_bf16_fp32(x, w)
        self.assertEqual(out.dtype, torch.float32)
        # bf16 x bf16 products are exact in fp64, so this is the reference up to fp32 order
        ref = x.double() @ w.double().t()
        err = (out.double() - ref).abs().max()
        bf16_err = (out.bfloat16().double() - ref).abs().max()
        self.assertLess(err, bf16_err)
        torch.testing.assert_close(out.double(), ref, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
