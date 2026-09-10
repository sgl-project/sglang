"""NVFP4 scale-layout and Standard-input regressions for the CUTLASS Adapter."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0),
    "Requires an SM100 GPU",
)
class TestFlashinferCutlassNvfp4Scales(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
            model_parallel_is_initialized,
        )
        from sglang.test.layer_ut_utils import init_single_process_dist

        if not torch.distributed.is_initialized():
            cls.addClassCleanup(destroy_distributed_environment)
        if not model_parallel_is_initialized():
            cls.addClassCleanup(destroy_model_parallel)
        init_single_process_dist()

    def _prepare_inputs(self, hidden_size):
        from flashinfer import fp4_quantize

        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
            FlashInferCutlassMoeQuantInfo,
        )
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardDispatchOutput,
        )
        from sglang.srt.layers.moe.topk import TopKConfig, select_experts
        from sglang.test.quant_ref_utils import FLOAT4_E2M1_MAX, FLOAT8_E4M3_MAX

        torch.manual_seed(42)
        num_tokens, intermediate_size = 32, 128
        num_experts, top_k = 4, 2
        self.config = MoeRunnerConfig(
            num_experts=num_experts,
            num_local_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size,
            top_k=top_k,
        )

        def quantize_weights(rows, cols):
            weights = (
                torch.randn(
                    num_experts, rows, cols, device="cuda", dtype=torch.bfloat16
                )
                / 10
            )
            scales = (
                FLOAT8_E4M3_MAX
                * FLOAT4_E2M1_MAX
                / weights.float().abs().amax(dim=(1, 2))
            )
            packed, block_scales = zip(
                *(fp4_quantize(w, gs) for w, gs in zip(weights, scales))
            )
            return torch.stack(packed), torch.stack(block_scales), scales

        w13, w13_sf, w13_gs = quantize_weights(2 * intermediate_size, hidden_size)
        w2, w2_sf, w2_gs = quantize_weights(hidden_size, intermediate_size)
        activation_scale = torch.ones((), device="cuda", dtype=torch.float32)
        self.quant_info = FlashInferCutlassMoeQuantInfo(
            quant_type="fp4",
            w13_weight=w13,
            w2_weight=w2,
            output_dtype=torch.bfloat16,
            quant_scales=[
                activation_scale,
                w13_sf,
                1 / (activation_scale * w13_gs),
                activation_scale,
                w2_sf,
                1 / (activation_scale * w2_gs),
            ],
        )
        x = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
        topk_output = select_experts(
            hidden_states=x,
            router_logits=torch.randn(num_tokens, num_experts, device="cuda"),
            topk_config=TopKConfig(top_k=top_k, renormalize=True),
        )
        self.standard = StandardDispatchOutput(x, None, topk_output)

    def test_linear_scales(self):
        from flashinfer import fp4_quantize, nvfp4_block_scale_interleave
        from flashinfer.fused_moe import cutlass_fused_moe

        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
            fused_experts_flashinfer_to_flashinfer_cutlass,
        )
        from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
            FlashinferCombineInput,
            FlashinferDispatchOutput,
        )

        # H=288 exercises the unaligned interleave fallback.
        for hidden_size in (128, 288):
            with self.subTest(hidden_size=hidden_size):
                self._prepare_inputs(hidden_size=hidden_size)
                quant_info = self.quant_info
                scales = quant_info.quant_scales
                x, linear_sf = fp4_quantize(
                    self.standard.hidden_states, scales[0], is_sf_swizzled_layout=False
                )
                output = torch.empty_like(self.standard.hidden_states)
                dispatch = FlashinferDispatchOutput(
                    x, linear_sf, self.standard.topk_output, output
                )
                expected = torch.empty_like(output)
                # Independent swizzled baseline with the same packed input and routing.
                cutlass_fused_moe(
                    input=x,
                    input_sf=nvfp4_block_scale_interleave(linear_sf),
                    swizzled_input_sf=True,
                    token_selected_experts=dispatch.topk_output.topk_ids.to(
                        torch.int32
                    ),
                    token_final_scales=dispatch.topk_output.topk_weights,
                    fc1_expert_weights=quant_info.w13_weight.view(torch.int64),
                    fc2_expert_weights=quant_info.w2_weight.view(torch.int64),
                    quant_scales=[
                        scales[0],
                        scales[1].view(torch.int32),
                        scales[2],
                        scales[3],
                        scales[4].view(torch.int32),
                        scales[5],
                    ],
                    output_dtype=output.dtype,
                    output=expected,
                    tune_max_num_tokens=x.shape[0],
                    enable_alltoall=True,
                    use_fused_finalize=envs.SGLANG_FLASHINFER_MOE_FUSED_FINALIZE.get(),
                )
                actual = fused_experts_flashinfer_to_flashinfer_cutlass(
                    dispatch_output=dispatch,
                    quant_info=quant_info,
                    runner_config=self.config,
                )
                self.assertIsInstance(actual, FlashinferCombineInput)
                self.assertEqual(actual.hidden_states.data_ptr(), output.data_ptr())
                torch.testing.assert_close(
                    actual.hidden_states, expected, rtol=1e-3, atol=1e-3
                )

    def test_standard_dispatch(self):
        from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
            fused_experts_flashinfer_to_flashinfer_cutlass,
            fused_experts_none_to_flashinfer_cutlass,
        )
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

        self._prepare_inputs(hidden_size=128)
        expected = fused_experts_none_to_flashinfer_cutlass(
            dispatch_output=self.standard,
            quant_info=self.quant_info,
            runner_config=self.config,
        )
        actual = fused_experts_flashinfer_to_flashinfer_cutlass(
            dispatch_output=self.standard,
            quant_info=self.quant_info,
            runner_config=self.config,
        )
        self.assertIsInstance(actual, StandardCombineInput)
        torch.testing.assert_close(
            actual.hidden_states, expected.hidden_states, rtol=1e-3, atol=1e-3
        )


if __name__ == "__main__":
    unittest.main()
