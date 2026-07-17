import unittest

import torch

from sglang.srt.utils import (
    is_gfx1201_supported,
    is_hip,
    is_triton_kernels_available,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")


@unittest.skipUnless(
    is_hip() and is_gfx1201_supported(), "gfx1201 ROCm GPU required"
)
class TestGfx1201Quantization(unittest.TestCase):
    def test_fp8_scaled_mm(self):
        torch.manual_seed(1)
        x = torch.randn(16, 256, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)

        x_scale = x.float().abs().max().clamp_min(1e-12) / 448.0
        weight_scale = weight.float().abs().max().clamp_min(1e-12) / 448.0
        x_fp8 = (x.float() / x_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        weight_fp8 = (
            (weight.float() / weight_scale)
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
        )

        actual = torch._scaled_mm(
            x_fp8,
            weight_fp8.t(),
            scale_a=x_scale.reshape(1),
            scale_b=weight_scale.reshape(1),
            out_dtype=torch.bfloat16,
        )
        if isinstance(actual, tuple):
            actual = actual[0]
        expected = (x_fp8.float() * x_scale) @ (
            weight_fp8.float() * weight_scale
        ).t()
        torch.testing.assert_close(actual.float(), expected, atol=0.2, rtol=0.03)

    def test_bf16_mxfp4_matmul(self):
        if not is_triton_kernels_available():
            self.skipTest("triton_kernels is not installed")

        from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig, matmul_ogs
        from triton_kernels.numerics_details.mxfp import (
            downcast_to_mxfp_torch,
            upcast_from_mxfp_torch,
        )

        from sglang.srt.layers.quantization.mxfp4 import _swizzle_mxfp4

        torch.manual_seed(2)
        x = torch.randn(16, 256, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        weight_fp4, weight_scale = downcast_to_mxfp_torch(
            weight, torch.uint8, axis=-1
        )
        wrapped_weight, weight_flex, prepared_scale = _swizzle_mxfp4(
            weight_fp4, weight_scale, num_warps=8, use_hbm_swizzle=False
        )
        precision = PrecisionConfig(
            flex_ctx=FlexCtx(rhs_data=weight_flex),
            weight_scale=prepared_scale,
            out_dtype=torch.bfloat16,
        )

        actual = matmul_ogs(x, wrapped_weight, None, precision_config=precision)
        dequant_weight = upcast_from_mxfp_torch(
            weight_fp4, weight_scale, target_dtype=torch.bfloat16, axis=-1
        )
        expected = x @ dequant_weight.t()
        torch.testing.assert_close(actual, expected, atol=0.2, rtol=0.03)

    def test_mxfp4_moe_routing(self):
        if not is_triton_kernels_available():
            self.skipTest("triton_kernels is not installed")

        from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig
        from triton_kernels.numerics_details.mxfp import downcast_to_mxfp_torch

        from sglang.srt.layers.moe.fused_moe_triton.triton_kernels_moe import (
            triton_kernel_fused_experts_with_bias,
        )
        from sglang.srt.layers.moe.topk import routing
        from sglang.srt.layers.quantization.mxfp4 import _swizzle_mxfp4

        torch.manual_seed(3)
        tokens, experts, hidden_size, intermediate_size = 4, 4, 256, 128
        hidden_states = torch.randn(
            tokens, hidden_size, device="cuda", dtype=torch.bfloat16
        )
        logits = torch.randn(tokens, experts, device="cuda", dtype=torch.float32)
        routing_data, gather_indx, scatter_indx = routing(logits, n_expts_act=2)

        w13 = torch.randn(
            experts,
            2 * intermediate_size,
            hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        w2 = torch.randn(
            experts,
            hidden_size,
            intermediate_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        w13_fp4, w13_scale = downcast_to_mxfp_torch(w13, torch.uint8, axis=-1)
        w2_fp4, w2_scale = downcast_to_mxfp_torch(w2, torch.uint8, axis=-1)
        w13_wrapped, w13_flex, w13_scale = _swizzle_mxfp4(
            w13_fp4, w13_scale, num_warps=8, use_hbm_swizzle=False
        )
        w2_wrapped, w2_flex, w2_scale = _swizzle_mxfp4(
            w2_fp4, w2_scale, num_warps=8, use_hbm_swizzle=False
        )

        actual = triton_kernel_fused_experts_with_bias(
            hidden_states=hidden_states,
            w1=w13_wrapped,
            w1_pcg=PrecisionConfig(
                weight_scale=w13_scale, flex_ctx=FlexCtx(rhs_data=w13_flex)
            ),
            b1=torch.zeros(
                experts, 2 * intermediate_size, device="cuda", dtype=torch.float32
            ),
            w2=w2_wrapped,
            w2_pcg=PrecisionConfig(
                weight_scale=w2_scale, flex_ctx=FlexCtx(rhs_data=w2_flex)
            ),
            b2=torch.zeros(
                experts, hidden_size, device="cuda", dtype=torch.float32
            ),
            routing_data=routing_data,
            gather_indx=gather_indx,
            scatter_indx=scatter_indx,
            gemm1_alpha=1.702,
            gemm1_clamp_limit=7.0,
        )

        self.assertEqual(actual.shape, hidden_states.shape)
        self.assertTrue(torch.isfinite(actual).all())


if __name__ == "__main__":
    unittest.main()
