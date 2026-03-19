import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
import torch.nn.functional as F

from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.moe.fused_moe_native import fused_moe_forward_native_grouped_mm
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
    _swiglu_gpt_oss_sigmoid_alpha,
)
from sglang.srt.layers.moe.moe_runner import InductorGroupedMMRunner
from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import initialize_moe_config
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, suite="stage-b-test-large-1-gpu")


class TestInductorGroupedMMFusedMoE(CustomTestCase):
    NUM_EXPERTS = 32
    TOP_K = 4
    HIDDEN_SIZE = 16
    INTERMEDIATE_SIZE = 24
    GEMM1_ALPHA = 1.702
    GEMM1_CLAMP_LIMIT = 7.0

    def _make_layer(
        self,
        device: str,
        num_experts: int | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        torch.manual_seed(1234)
        if num_experts is None:
            num_experts = self.NUM_EXPERTS
        return SimpleNamespace(
            num_experts=num_experts,
            num_local_experts=num_experts,
            w13_weight=torch.randn(
                num_experts,
                2 * self.INTERMEDIATE_SIZE,
                self.HIDDEN_SIZE,
                device=device,
                dtype=dtype,
            ),
            w2_weight=torch.randn(
                num_experts,
                self.HIDDEN_SIZE,
                self.INTERMEDIATE_SIZE,
                device=device,
                dtype=dtype,
            ),
            w13_weight_bias=torch.randn(
                num_experts,
                2 * self.INTERMEDIATE_SIZE,
                device=device,
                dtype=torch.float32,
            ),
            w2_weight_bias=torch.randn(
                num_experts,
                self.HIDDEN_SIZE,
                device=device,
                dtype=torch.float32,
            ),
        )

    def _make_inputs(
        self,
        device: str,
        num_experts: int | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        if num_experts is None:
            num_experts = self.NUM_EXPERTS
        hidden_states = torch.randn(5, self.HIDDEN_SIZE, device=device, dtype=dtype)
        topk_ids = (
            torch.arange(hidden_states.shape[0] * self.TOP_K, device=device)
            .reshape(hidden_states.shape[0], self.TOP_K)
            % num_experts
        ).to(torch.int64)
        topk_weights = torch.tensor(
            [
                [0.40, 0.30, 0.20, 0.10],
                [0.35, 0.25, 0.20, 0.20],
                [0.50, 0.20, 0.20, 0.10],
                [0.25, 0.25, 0.25, 0.25],
                [0.45, 0.20, 0.20, 0.15],
            ],
            device=device,
            dtype=torch.float32,
        )
        topk_output = StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=torch.randn(
                hidden_states.shape[0],
                num_experts,
                device=device,
                dtype=torch.float32,
            ),
        )
        dispatch_output = StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=topk_output,
        )
        return hidden_states, topk_output, dispatch_output

    def _reference_grouped_mm(self, layer, hidden_states, topk_output):
        topk_weights, topk_ids, _ = topk_output
        output = torch.zeros_like(hidden_states)

        for token_idx in range(hidden_states.shape[0]):
            token_hidden = hidden_states[token_idx : token_idx + 1]
            for route_idx in range(topk_ids.shape[1]):
                expert_id = int(topk_ids[token_idx, route_idx].item())
                gate_up = F.linear(token_hidden, layer.w13_weight[expert_id])
                gate_up = gate_up + layer.w13_weight_bias[expert_id]
                gate = gate_up[..., ::2].clamp(max=self.GEMM1_CLAMP_LIMIT)
                up = gate_up[..., 1::2].clamp(
                    min=-self.GEMM1_CLAMP_LIMIT, max=self.GEMM1_CLAMP_LIMIT
                )
                hidden_after_activation = gate * torch.sigmoid(
                    gate * self.GEMM1_ALPHA
                ) * (up + 1)
                hidden_after_activation = hidden_after_activation.to(hidden_states.dtype)
                expert_out = F.linear(hidden_after_activation, layer.w2_weight[expert_id])
                expert_out = expert_out + layer.w2_weight_bias[expert_id]
                weighted_out = (
                    expert_out * topk_weights[token_idx, route_idx].to(expert_out.dtype)
                ).to(output.dtype)
                output[token_idx] = (output[token_idx] + weighted_out.squeeze(0)).to(
                    output.dtype
                )

        return output

    def test_grouped_mm_matches_reference_for_gpt_oss_layout(self):
        device = "cuda"
        layer = self._make_layer(device)
        hidden_states, topk_output, _ = self._make_inputs(device)
        moe_runner_config = MoeRunnerConfig(
            activation="silu",
            gemm1_alpha=self.GEMM1_ALPHA,
            gemm1_clamp_limit=self.GEMM1_CLAMP_LIMIT,
        )

        grouped_mm_output = fused_moe_forward_native_grouped_mm(
            layer=layer,
            hidden_states=hidden_states,
            topk_output=topk_output,
            moe_runner_config=moe_runner_config,
            activation_fn=_swiglu_gpt_oss_sigmoid_alpha,
            activation_fn_args=InductorGroupedMMRunner.create_native_activation_fn_args,
        )
        reference_output = self._reference_grouped_mm(layer, hidden_states, topk_output)

        torch.testing.assert_close(
            grouped_mm_output.float(),
            reference_output.float(),
            rtol=5e-2,
            atol=5e-2,
        )

    def test_forward_cuda_uses_grouped_mm_only_on_decode(self):
        server_args = ServerArgs(
            model_path="dummy",
            moe_runner_backend="inductor_groupedmm",
        )
        set_global_server_args_for_scheduler(server_args)
        initialize_moe_config(server_args)

        device = "cuda"
        layer = self._make_layer(device, num_experts=8, dtype=torch.float16)
        _, _, dispatch_output = self._make_inputs(
            device, num_experts=8, dtype=torch.float16
        )
        method = UnquantizedFusedMoEMethod()
        method.create_moe_runner(
            layer,
            MoeRunnerConfig(
                activation="silu",
                gemm1_alpha=self.GEMM1_ALPHA,
                gemm1_clamp_limit=self.GEMM1_CLAMP_LIMIT,
            ),
        )
        self.assertIsInstance(method.runner, InductorGroupedMMRunner)
        default_output = StandardCombineInput(
            hidden_states=torch.full_like(dispatch_output.hidden_states, 3)
        )
        method.runner.default_runner.run = MagicMock(return_value=default_output)
        method.runner._forward_cuda_grouped_mm = MagicMock(
            return_value=StandardCombineInput(
                hidden_states=torch.full_like(dispatch_output.hidden_states, 1)
            )
        )

        decode_output = method.forward_cuda(layer, dispatch_output, is_decode=True)
        self.assertTrue(method.runner._forward_cuda_grouped_mm.called)
        self.assertFalse(method.runner.default_runner.run.called)
        torch.testing.assert_close(
            decode_output.hidden_states,
            torch.full_like(dispatch_output.hidden_states, 1),
        )

        method.runner._forward_cuda_grouped_mm.reset_mock()
        method.runner.default_runner.run.reset_mock()

        prefill_output = method.forward_cuda(layer, dispatch_output, is_decode=False)
        method.runner._forward_cuda_grouped_mm.assert_not_called()
        self.assertTrue(method.runner.default_runner.run.called)
        torch.testing.assert_close(
            prefill_output.hidden_states,
            torch.full_like(dispatch_output.hidden_states, 3),
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
