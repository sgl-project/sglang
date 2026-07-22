import unittest
from unittest import mock

import torch

from sglang.srt.utils import (
    is_gfx1201_supported,
    is_hip,
    is_triton_kernels_available,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")


@unittest.skipUnless(is_hip() and is_gfx1201_supported(), "gfx1201 ROCm GPU required")
class TestGfx1201Quantization(unittest.TestCase):
    def test_moe_topk(self):
        from sglang.kernels.ops.moe import topk_softmax
        from sglang.srt.layers.moe.topk import fused_topk

        torch.manual_seed(0)
        for num_experts in (4, 6):
            logits = torch.randn(4, num_experts, device="cuda", dtype=torch.bfloat16)
            actual_weights = torch.empty(4, 2, device="cuda", dtype=torch.float32)
            actual_ids = torch.empty(4, 2, device="cuda", dtype=torch.int32)

            topk_softmax(actual_weights, actual_ids, logits)
            expected_weights, expected_ids = torch.softmax(logits.float(), dim=-1).topk(
                2, dim=-1
            )

            torch.testing.assert_close(
                actual_weights, expected_weights, atol=1e-6, rtol=1e-5
            )
            torch.testing.assert_close(actual_ids, expected_ids.to(torch.int32))

        hidden_states = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
        for num_experts in (8, 6):
            logits = torch.randn(4, num_experts, device="cuda", dtype=torch.bfloat16)
            actual_weights, actual_ids = fused_topk(
                hidden_states,
                logits,
                topk=2,
                renormalize=True,
                scoring_func="sigmoid",
            )
            expected_weights, expected_ids = torch.sigmoid(logits.float()).topk(
                2, dim=-1
            )
            expected_weights /= expected_weights.sum(dim=-1, keepdim=True)
            torch.testing.assert_close(
                actual_weights, expected_weights, atol=1e-3, rtol=1e-3
            )
            torch.testing.assert_close(actual_ids, expected_ids.to(torch.int32))

    def test_bf16_and_fp8_fused_moe(self):
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            fused_moe,
        )
        from sglang.srt.layers.moe.topk import TopKConfig, select_experts
        from sglang.srt.server_args import (
            ServerArgs,
            set_global_server_args_for_scheduler,
        )

        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        torch.manual_seed(1)
        tokens, experts, hidden_size, intermediate_size, top_k = 4, 4, 256, 128, 2
        hidden_states = torch.randn(
            tokens, hidden_size, device="cuda", dtype=torch.bfloat16
        )
        logits = torch.randn(tokens, experts, device="cuda", dtype=torch.bfloat16)
        w13_bf16 = (
            torch.randn(
                experts,
                2 * intermediate_size,
                hidden_size,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.01
        )
        w2_bf16 = (
            torch.randn(
                experts,
                hidden_size,
                intermediate_size,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.01
        )
        topk_output = select_experts(
            hidden_states=hidden_states,
            router_logits=logits,
            topk_config=TopKConfig(top_k=top_k, renormalize=False),
        )

        fp8_modes = (
            (False, False, None, None, None),
            (
                True,
                False,
                torch.ones(experts, device="cuda", dtype=torch.float32),
                torch.ones(experts, device="cuda", dtype=torch.float32),
                None,
            ),
            (
                True,
                True,
                torch.ones(
                    experts,
                    2 * intermediate_size,
                    device="cuda",
                    dtype=torch.float32,
                ),
                torch.ones(experts, hidden_size, device="cuda", dtype=torch.float32),
                None,
            ),
            (
                True,
                False,
                torch.ones(experts, 2, 2, device="cuda", dtype=torch.float32),
                torch.ones(experts, 2, 1, device="cuda", dtype=torch.float32),
                [128, 128],
            ),
        )
        for use_fp8, per_channel_quant, w13_scale, w2_scale, block_shape in fp8_modes:
            w13 = w13_bf16.to(torch.float8_e4m3fn) if use_fp8 else w13_bf16
            w2 = w2_bf16.to(torch.float8_e4m3fn) if use_fp8 else w2_bf16

            actual = fused_moe(
                hidden_states,
                w13,
                w2,
                topk_output,
                use_fp8_w8a8=use_fp8,
                per_channel_quant=per_channel_quant,
                w1_scale=w13_scale,
                w2_scale=w2_scale,
                block_shape=block_shape,
            )

            topk_weights, topk_ids, _ = topk_output
            expected = torch.zeros_like(hidden_states)
            for token_id in range(tokens):
                for route_id in range(top_k):
                    expert_id = int(topk_ids[token_id, route_id])
                    gate_up = (
                        hidden_states[token_id] @ w13[expert_id].to(torch.bfloat16).t()
                    )
                    gate, up = gate_up.chunk(2)
                    expert_output = (torch.nn.functional.silu(gate) * up) @ w2[
                        expert_id
                    ].to(torch.bfloat16).t()
                    expected[token_id] += (
                        expert_output * topk_weights[token_id, route_id]
                    )

            torch.testing.assert_close(
                actual.float(), expected.float(), atol=0.35, rtol=0.08
            )

    def test_fp8_scaled_mm(self):
        torch.manual_seed(1)
        x = torch.randn(16, 256, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)

        x_scale = x.float().abs().max().clamp_min(1e-12) / 448.0
        weight_scale = weight.float().abs().max().clamp_min(1e-12) / 448.0
        x_fp8 = (x.float() / x_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        weight_fp8 = (
            (weight.float() / weight_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
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
        expected = (x_fp8.float() * x_scale) @ (weight_fp8.float() * weight_scale).t()
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
        weight_fp4, weight_scale = downcast_to_mxfp_torch(weight, torch.uint8, axis=-1)
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
        from triton_kernels.numerics_details.mxfp import (
            downcast_to_mxfp_torch,
            upcast_from_mxfp_torch,
        )

        from sglang.srt.layers.moe.fused_moe_triton import (
            triton_kernels_moe,
        )
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

        torch.testing.assert_close(gather_indx.src_indx, scatter_indx.dst_indx)
        with mock.patch.object(
            triton_kernels_moe, "_MATMUL_OGS_USES_RAGGED_METADATA", True
        ):
            new_api_gather_indx = triton_kernels_moe._gather_indx_for_matmul(
                gather_indx, routing_data
            )
        torch.testing.assert_close(
            new_api_gather_indx.src_indx,
            torch.div(gather_indx.src_indx, 2, rounding_mode="trunc"),
        )
        torch.testing.assert_close(new_api_gather_indx.dst_indx, gather_indx.dst_indx)

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
        w13_fp4, w13_scale_raw = downcast_to_mxfp_torch(w13, torch.uint8, axis=-1)
        w2_fp4, w2_scale_raw = downcast_to_mxfp_torch(w2, torch.uint8, axis=-1)
        w13_wrapped, w13_flex, w13_scale = _swizzle_mxfp4(
            w13_fp4, w13_scale_raw, num_warps=8, use_hbm_swizzle=False
        )
        w2_wrapped, w2_flex, w2_scale = _swizzle_mxfp4(
            w2_fp4, w2_scale_raw, num_warps=8, use_hbm_swizzle=False
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
            b2=torch.zeros(experts, hidden_size, device="cuda", dtype=torch.float32),
            routing_data=routing_data,
            gather_indx=gather_indx,
            scatter_indx=scatter_indx,
            gemm1_alpha=1.702,
            gemm1_clamp_limit=7.0,
        )

        self.assertEqual(actual.shape, hidden_states.shape)
        w13_dequant = upcast_from_mxfp_torch(
            w13_fp4, w13_scale_raw, target_dtype=torch.bfloat16, axis=-1
        )
        w2_dequant = upcast_from_mxfp_torch(
            w2_fp4, w2_scale_raw, target_dtype=torch.bfloat16, axis=-1
        )
        topk_logits, topk_ids = logits.topk(2, dim=-1)
        topk_weights = torch.softmax(topk_logits, dim=-1)
        expected = torch.zeros_like(hidden_states, dtype=torch.float32)
        for token_id in range(tokens):
            for route_id in range(2):
                expert_id = int(topk_ids[token_id, route_id])
                gate_up = (
                    hidden_states[token_id].float() @ w13_dequant[expert_id].float().t()
                )
                gate = gate_up[::2].clamp(max=7.0)
                linear = gate_up[1::2].clamp(min=-7.0, max=7.0)
                activated = gate * torch.sigmoid(1.702 * gate) * (linear + 1)
                expert_output = activated @ w2_dequant[expert_id].float().t()
                expected[token_id] += expert_output * topk_weights[token_id, route_id]

        torch.testing.assert_close(
            actual.float(),
            expected,
            atol=2.0,
            rtol=0.05,
        )


if __name__ == "__main__":
    unittest.main()
