import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm as runner
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFlashInferTrtllmFp4PreQuant(CustomTestCase):
    def test_fused_ar_pair_reaches_runner_without_requantizing(self):
        num_tokens, hidden_size = 3, 32
        hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16)
        packed_fp4 = torch.empty(num_tokens, hidden_size // 2, dtype=torch.uint8)
        linear_scales = torch.empty(
            num_tokens, hidden_size // 16, dtype=torch.float8_e4m3fn
        )
        fused_pair = (packed_fp4, linear_scales)
        topk_output = types.SimpleNamespace(
            router_logits=torch.randn(num_tokens, 8),
            topk_config=types.SimpleNamespace(
                correction_bias=None,
                top_k=2,
                num_expert_group=1,
                topk_group=1,
            ),
        )
        quant_info = MagicMock(use_per_token_activation=False, routing_method_type=0)
        runner_config = MagicMock(
            activation="silu", is_gated=True, routed_scaling_factor=1.0
        )

        routed_output = torch.empty_like(hidden_states)
        expert_weights = torch.empty(num_tokens, 2, dtype=torch.float32)
        expanded_idx_to_permuted_idx = torch.empty(num_tokens, 2, dtype=torch.int32)
        moe_kernel = MagicMock(
            return_value=(
                routed_output,
                expert_weights,
                expanded_idx_to_permuted_idx,
            )
        )
        standalone_quant = MagicMock(
            side_effect=AssertionError("standalone FP4 quantization was called")
        )
        import flashinfer.fused_moe as flashinfer_fused_moe

        from sglang.srt.layers.moe import topk as topk_module

        def run_moe_core(*, dispatch_output):
            self.assertIs(dispatch_output.hidden_states_pre_quant, fused_pair)
            return runner.fused_experts_none_to_flashinfer_trtllm_fp4(
                dispatch_output,
                quant_info,
                runner_config,
            )

        dispatcher = MagicMock()
        dispatcher.dispatch.return_value = StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=topk_output,
        )
        dispatcher.combine.return_value = routed_output
        moe = types.SimpleNamespace(
            quant_method=object(),
            dispatcher=dispatcher,
            run_moe_core=MagicMock(side_effect=run_moe_core),
        )

        with (
            patch.object(
                flashinfer_fused_moe,
                "trtllm_fp4_block_scale_moe",
                moe_kernel,
            ),
            patch.object(runner, "quantize_hidden_states_fp4", standalone_quant),
            patch.object(runner, "get_activation_type", return_value=7),
            patch.object(runner, "trtllm_moe_enable_pdl", return_value=False),
            patch.object(
                topk_module.TopKOutputChecker,
                "format_is_standard",
                return_value=False,
            ),
            patch.object(
                topk_module.TopKOutputChecker,
                "format_is_bypassed",
                return_value=True,
            ),
        ):
            result = FusedMoE.forward_deferred_finalize(
                moe,
                hidden_states,
                topk_output,
                pre_quant_input=fused_pair,
            )

        self.assertIs(result, routed_output)
        standalone_quant.assert_not_called()
        combine_input = dispatcher.combine.call_args.kwargs["combine_input"]
        deferred_output = combine_input.hidden_states
        self.assertIs(deferred_output.gemm2_out, routed_output)
        self.assertIs(deferred_output.expert_weights, expert_weights)
        self.assertIs(
            deferred_output.expanded_idx_to_permuted_idx,
            expanded_idx_to_permuted_idx,
        )
        kernel_kwargs = moe_kernel.call_args.kwargs
        self.assertIs(kernel_kwargs["hidden_states"], packed_fp4)
        self.assertEqual(
            kernel_kwargs["hidden_states_scale"].data_ptr(), linear_scales.data_ptr()
        )
        self.assertFalse(kernel_kwargs["do_finalize"])


if __name__ == "__main__":
    unittest.main()
