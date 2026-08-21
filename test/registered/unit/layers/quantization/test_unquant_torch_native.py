"""Regression tests for unquantized torch-native MoE dispatch."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe import MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.fused_moe_native import fused_moe_forward_native
from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.quantization import unquant
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


class TestUnquantizedTorchNativeMoE(CustomTestCase):
    def test_cuda_forward_matches_native_forward(self):
        config = MoeRunnerConfig(
            num_experts=4,
            num_local_experts=4,
            hidden_size=16,
            intermediate_size_per_partition=32,
            top_k=2,
            params_dtype=torch.bfloat16,
            activation="silu",
            is_gated=True,
        )
        generator = torch.Generator(device="cuda").manual_seed(0)
        layer = SimpleNamespace(
            w13_weight=torch.randn(
                4, 64, 16, dtype=torch.bfloat16, device="cuda", generator=generator
            ),
            w2_weight=torch.randn(
                4, 16, 32, dtype=torch.bfloat16, device="cuda", generator=generator
            ),
            moe_runner_config=config,
        )
        dispatch_output = StandardDispatchOutput(
            hidden_states=torch.randn(
                3, 16, dtype=torch.bfloat16, device="cuda", generator=generator
            ),
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(
                topk_weights=torch.tensor(
                    [[0.7, 0.3], [0.6, 0.4], [0.8, 0.2]],
                    dtype=torch.float32,
                    device="cuda",
                ),
                topk_ids=torch.tensor([[0, 1], [2, 3], [1, 2]], device="cuda"),
                router_logits=torch.zeros(3, 4, device="cuda"),
            ),
        )
        expected = fused_moe_forward_native(layer, dispatch_output).hidden_states

        with patch.object(
            unquant,
            "get_moe_runner_backend",
            return_value=MoeRunnerBackend.TORCH_NATIVE,
        ):
            method = unquant.UnquantizedFusedMoEMethod()
            method.create_moe_runner(layer, config)
            actual = method.forward_cuda(layer, dispatch_output).hidden_states

        torch.testing.assert_close(actual, expected, rtol=1e-1, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
