"""CPU contracts for MiniMax-M3 on ordinary FlashInfer TRT-LLM MoE."""

import sys
import types
import unittest
from contextlib import nullcontext
from enum import IntEnum
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm as flashinfer_runner
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import BypassedTopKOutput, TopKConfig
from sglang.srt.layers.moe.utils import RoutingMethodType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Fp8QuantizationType(IntEnum):
    DeepSeekFp8 = 1
    MxFp8 = 2


def _fake_flashinfer_modules():
    flashinfer = types.ModuleType("flashinfer")
    flashinfer.__path__ = []
    fused_moe = types.ModuleType("flashinfer.fused_moe")
    fused_moe.Fp8QuantizationType = _Fp8QuantizationType
    flashinfer.fused_moe = fused_moe
    return {
        "flashinfer": flashinfer,
        "flashinfer.fused_moe": fused_moe,
    }


class TestMiniMaxFlashInferTrtllmRouting(CustomTestCase):
    def test_minimax2_prefers_topk_routed_scale(self):
        cases = (
            (RoutingMethodType.MiniMax2, 2.0, 1.25, 2.0),
            (RoutingMethodType.Renormalize, 2.0, 1.25, 1.25),
            (RoutingMethodType.MiniMax2, None, None, 1.0),
        )
        for routing_method, topk_scale, runner_scale, expected in cases:
            with self.subTest(routing_method=routing_method):
                actual = (
                    flashinfer_runner.resolve_flashinfer_trtllm_routed_scaling_factor(
                        routing_method,
                        topk_scale,
                        MoeRunnerConfig(routed_scaling_factor=runner_scale),
                        default=1.0,
                    )
                )
                self.assertEqual(actual, expected)

    def test_ordinary_mxfp8_forwards_minimax_topk_scale(self):
        hidden_states = torch.randn(2, 4, dtype=torch.bfloat16)
        routing_bias = torch.randn(3, dtype=torch.float32)
        topk_config = TopKConfig(
            top_k=2,
            renormalize=True,
            routed_scaling_factor=2.0,
            correction_bias=routing_bias,
        )
        dispatch_output = StandardDispatchOutput(
            hidden_states,
            None,
            BypassedTopKOutput(
                hidden_states,
                torch.randn(2, 3, dtype=torch.float32),
                topk_config,
            ),
        )
        quant_info = flashinfer_runner.FlashInferTrtllmFp8MoeQuantInfo(
            w13_weight=torch.empty(3, 16, 4),
            w2_weight=torch.empty(3, 4, 8),
            global_num_experts=3,
            local_expert_offset=0,
            local_num_experts=3,
            intermediate_size=8,
            routing_method_type=int(RoutingMethodType.MiniMax2),
            block_quant=True,
            use_mxfp8=True,
            weight_block_k=32,
            w13_weight_scale_inv=torch.empty(3, 1, 1),
            w2_weight_scale_inv=torch.empty(3, 1, 1),
        )
        runner_config = MoeRunnerConfig(
            num_local_experts=3,
            intermediate_size_per_partition=8,
            top_k=2,
            activation="silu",
            is_gated=True,
        )
        kernel = MagicMock(return_value=None)

        from sglang.srt.layers.quantization import fp8_utils

        with (
            patch.dict(sys.modules, _fake_flashinfer_modules()),
            patch.object(
                fp8_utils,
                "flashinfer_mxfp8_quantize",
                return_value=(
                    hidden_states.clone(),
                    torch.ones(2, 1, dtype=torch.uint8),
                ),
            ),
            patch.object(
                flashinfer_runner,
                "trtllm_fp8_block_scale_moe_out_wrapper",
                kernel,
            ),
            patch.object(
                flashinfer_runner,
                "use_symmetric_memory",
                side_effect=lambda *args, **kwargs: nullcontext(),
            ),
            patch.object(flashinfer_runner, "get_tp_group", return_value=None),
            patch.object(
                flashinfer_runner, "is_allocation_symmetric", return_value=False
            ),
        ):
            flashinfer_runner.fused_experts_none_to_flashinfer_trtllm_fp8(
                dispatch_output, quant_info, runner_config
            )

        self.assertEqual(kernel.call_args.kwargs["routed_scaling_factor"], 2.0)
        self.assertEqual(
            kernel.call_args.kwargs["routing_method_type"],
            RoutingMethodType.MiniMax2,
        )

    def test_model_marks_experts_as_minimax2_routing(self):
        from sglang.srt.models import minimax_m3 as minimax_module

        config = SimpleNamespace(
            n_shared_experts=None,
            num_local_experts=4,
            num_experts_per_tok=2,
            hidden_size=8,
            intermediate_size=16,
            swiglu_alpha=1.702,
            swiglu_limit=7.0,
            scoring_func="sigmoid",
            routed_scaling_factor=2.0,
            use_routing_bias=False,
        )
        experts_factory = MagicMock(return_value=torch.nn.Identity())

        with (
            patch.object(
                minimax_module,
                "get_parallel",
                return_value=SimpleNamespace(tp_size=1),
            ),
            patch.object(
                minimax_module,
                "get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(ep_num_redundant_experts=0)
                ),
            ),
            patch.object(
                minimax_module,
                "is_shared_experts_fusion_disabled",
                return_value=True,
            ),
            patch.object(
                minimax_module, "get_moe_impl_class", return_value=experts_factory
            ),
            patch.object(minimax_module, "TopK", return_value=torch.nn.Identity()),
        ):
            minimax_module.MiniMaxM3MoE(
                config=config,
                quant_config=None,
                prefix="model.layers.0.mlp",
                layer_id=0,
            )

        self.assertEqual(
            experts_factory.call_args.kwargs["routing_method_type"],
            RoutingMethodType.MiniMax2,
        )


if __name__ == "__main__":
    unittest.main()
