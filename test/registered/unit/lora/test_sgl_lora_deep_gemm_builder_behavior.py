"""Behavior tests for the compact SGL LoRA DeepGEMM hook builder."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import sglang.srt.lora.sgl_lora.deep_gemm_bf16 as builder_module
from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.lora.layers import FusedMoEWithLoRA
from sglang.srt.lora.sgl_lora.bf16 import Bf16MoeLaunchConfig
from sglang.srt.lora.sgl_lora.deep_gemm_bf16 import DeepGemmBf16LoRAHookBuilder
from sglang.srt.lora.sgl_lora.hooks import (
    DownHookContext,
    GateUpHookContext,
    MoeLoRAHookBuildContext,
)
from sglang.srt.server_args import ServerArgs


def _launch_config() -> Bf16MoeLaunchConfig:
    return Bf16MoeLaunchConfig(
        routing_block_size=16,
        lora_a={
            "BLOCK_SIZE_N": 16,
            "BLOCK_SIZE_K": 16,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 2,
        },
        lora_b={
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 16,
            "BLOCK_SIZE_K": 16,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 2,
        },
    )


def _context(
    *,
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    shared_outer: bool,
    experts: int = 2,
    adapters: int = 2,
    rank: int = 16,
    intermediate: int = 16,
):
    lora_info = SimpleNamespace(
        gate_up_lora_a_weights=torch.randn(
            adapters,
            1 if shared_outer else experts,
            2 * rank,
            hidden.shape[1],
            dtype=torch.bfloat16,
            device=hidden.device,
        ),
        gate_up_lora_b_weights=torch.randn(
            adapters,
            experts,
            2 * intermediate,
            rank,
            dtype=torch.bfloat16,
            device=hidden.device,
        ),
        down_lora_a_weights=torch.randn(
            adapters,
            experts,
            rank,
            intermediate,
            dtype=torch.bfloat16,
            device=hidden.device,
        ),
        down_lora_b_weights=torch.randn(
            adapters,
            1 if shared_outer else experts,
            hidden.shape[1],
            rank,
            dtype=torch.bfloat16,
            device=hidden.device,
        ),
        token_lora_mapping=token_lora_mapping,
        experts_shared_outer_loras=shared_outer,
    )
    dispatch = SimpleNamespace(
        hidden_states=hidden,
        topk_output=SimpleNamespace(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        ),
    )
    return MoeLoRAHookBuildContext(
        dispatch_output=dispatch,
        lora_info=lora_info,
        runner_backend=MoeRunnerBackend.DEEP_GEMM,
    )


class TestSglLoraDeepGemmBuilderBehavior(unittest.TestCase):
    def test_server_args_reject_unmapped_distributed_domains(self):
        base = dict(
            max_loras_per_batch=2,
            lora_execution_engine="sgl_lora",
            lora_use_virtual_experts=True,
            enable_dp_attention=False,
            dp_size=1,
            enable_eplb=False,
            init_expert_location="trivial",
            ep_num_redundant_experts=0,
        )
        cases = (
            ("elastic_ep", dict(ep_join_mode="scale")),
            ("dp_attention", dict(enable_dp_attention=True, dp_size=2)),
            ("eplb", dict(enable_eplb=True)),
            ("placement", dict(init_expert_location="nontrivial")),
            ("redundant", dict(ep_num_redundant_experts=1)),
        )
        for name, changes in cases:
            with self.subTest(name=name):
                args = SimpleNamespace(**(base | changes))
                with self.assertRaises(ValueError):
                    ServerArgs.check_lora_server_args(args)

    def test_factor_domains_match_normal_and_shared_outer_contracts(self):
        hidden = torch.randn(2, 8, dtype=torch.bfloat16)
        topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
        topk_weights = torch.rand(2, 2)
        token_lora_mapping = torch.tensor([0, 1], dtype=torch.int32)
        layer = object.__new__(FusedMoEWithLoRA)
        torch.nn.Module.__init__(layer)
        layer.lora_execution_engine = "sgl_lora"
        layer._sgl_lora_factor_experts = 2

        normal = _context(
            hidden=hidden,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_lora_mapping=token_lora_mapping,
            shared_outer=False,
        )
        normal.lora_info.gate_up_lora_a_weights = (
            normal.lora_info.gate_up_lora_a_weights[:, :1]
        )
        layer.experts_shared_outer_loras = False
        with self.assertRaisesRegex(ValueError, "factor domains"):
            layer.set_lora_info(
                normal.lora_info.gate_up_lora_a_weights,
                normal.lora_info.gate_up_lora_b_weights,
                normal.lora_info.down_lora_a_weights,
                normal.lora_info.down_lora_b_weights,
            )

        shared = _context(
            hidden=hidden,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_lora_mapping=token_lora_mapping,
            shared_outer=True,
        )
        shared.lora_info.down_lora_b_weights = torch.randn(
            2, 2, 8, 16, dtype=torch.bfloat16
        )
        layer.experts_shared_outer_loras = True
        with self.assertRaisesRegex(ValueError, "factor domains"):
            layer.set_lora_info(
                shared.lora_info.gate_up_lora_a_weights,
                shared.lora_info.gate_up_lora_b_weights,
                shared.lora_info.down_lora_a_weights,
                shared.lora_info.down_lora_b_weights,
            )

    def test_layer_forwards_explicit_output_dtype_to_typed_runner(self):
        layer = object.__new__(FusedMoEWithLoRA)
        torch.nn.Module.__init__(layer)
        dispatch_output = SimpleNamespace()
        final_output = torch.randn(2, 8)
        dispatcher = SimpleNamespace(
            dispatch=Mock(return_value=dispatch_output),
            combine=Mock(return_value=final_output),
        )
        combine_input = StandardCombineInput(torch.empty(2, 8))
        finalize = Mock(return_value=final_output)
        layer.base_layer = SimpleNamespace(
            dispatcher=dispatcher,
            combine_and_finalize=finalize,
        )
        layer._quant_info = SimpleNamespace()
        layer._lora_runner_backend = MoeRunnerBackend.DEEP_GEMM
        layer._lora_runner = Mock(run=Mock(return_value=combine_input))
        layer.lora_execution_engine = "sgl_lora"

        hidden = torch.randn(2, 8)
        topk_output = SimpleNamespace()
        lora_info = SimpleNamespace()
        result = layer._forward_with_lora(
            hidden,
            topk_output,
            lora_info,
            output_dtype=torch.float32,
        )

        self.assertIs(result, final_output)
        self.assertEqual(result.dtype, torch.float32)
        layer._lora_runner.run.assert_called_once_with(
            dispatch_output,
            layer._quant_info,
            lora_info=lora_info,
            output_dtype=torch.float32,
        )
        finalize.assert_called_once_with(
            combine_input=combine_input,
            origin_hidden_states_dim=hidden.shape[-1],
        )
        dispatcher.combine.assert_not_called()

    def test_callbacks_share_routing_and_preserve_base_provider_rows(self):
        hidden = torch.randn(2, 8, dtype=torch.bfloat16)
        topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
        topk_weights = torch.rand(2, 2)
        adapters = torch.tensor([0, -1], dtype=torch.int32)
        context = _context(
            hidden=hidden,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_lora_mapping=adapters,
            shared_outer=False,
            rank=4,
            intermediate=6,
        )
        builder = DeepGemmBf16LoRAHookBuilder(_launch_config())

        def fake_a(input, weight, output, routing, **kwargs):
            output.zero_()
            output[routing.virtual_topk_ids.reshape(-1) >= 0] = 2

        def fake_b(intermediate, weight, destination, routing, **kwargs):
            destination.zero_()
            destination[routing.virtual_topk_ids.reshape(-1) >= 0] = 3

        real_build_routing = builder_module.build_virtual_expert_routing
        with (
            patch.object(builder_module, "grouped_lora_a", side_effect=fake_a) as a,
            patch.object(builder_module, "stock_grouped_lora_b", side_effect=fake_b),
            patch.object(
                builder_module,
                "build_virtual_expert_routing",
                wraps=real_build_routing,
            ) as build_routing,
        ):
            hooks = builder(context)
            self.assertEqual(a.call_count, 1)
            self.assertEqual(build_routing.call_count, 1)

            provider_gate_up = torch.full((2, 3, 12), 7, dtype=torch.bfloat16)
            pair_to_provider_row = torch.tensor([0, 3, 4, 1], dtype=torch.int32)
            hooks.inject_gate_up(
                GateUpHookContext(
                    provider_gate_up=provider_gate_up,
                    pair_to_provider_row=pair_to_provider_row,
                    provider_layout="gate_then_up",
                )
            )
            rows = provider_gate_up.view(-1, 12)
            torch.testing.assert_close(rows[0], torch.full_like(rows[0], 10))
            torch.testing.assert_close(rows[3], torch.full_like(rows[3], 10))
            torch.testing.assert_close(rows[1], torch.full_like(rows[1], 7))
            torch.testing.assert_close(rows[4], torch.full_like(rows[4], 7))

            contribution = hooks.build_down_pair_contribution(
                DownHookContext(
                    provider_activation=torch.randn(2, 3, 6, dtype=torch.bfloat16),
                    pair_to_provider_row=pair_to_provider_row,
                )
            )
            self.assertEqual(a.call_count, 2)
            self.assertEqual(tuple(contribution.values.shape), (2, 2, 8))
            torch.testing.assert_close(
                contribution.values[0],
                torch.full_like(contribution.values[0], 3),
            )
            torch.testing.assert_close(
                contribution.values[1],
                torch.zeros_like(contribution.values[1]),
            )

    def test_shared_outer_uses_two_domains_and_excludes_physical_shared_ids(self):
        hidden = torch.randn(2, 8, dtype=torch.bfloat16)
        # Physical expert 2 is outside the two routed-expert factor slots.
        topk_ids = torch.tensor([[0, 2], [1, 2]], dtype=torch.int32)
        topk_weights = torch.rand(2, 2)
        context = _context(
            hidden=hidden,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_lora_mapping=torch.tensor([0, 1], dtype=torch.int32),
            shared_outer=True,
            rank=4,
            intermediate=6,
        )
        builder = DeepGemmBf16LoRAHookBuilder(_launch_config())

        seen_routings = []

        def fake_a(input, weight, output, routing, **kwargs):
            seen_routings.append(routing.virtual_topk_ids.clone())
            output.zero_()

        real_build_routing = builder_module.build_virtual_expert_routing
        with (
            patch.object(builder_module, "grouped_lora_a", side_effect=fake_a),
            patch.object(
                builder_module,
                "build_virtual_expert_routing",
                wraps=real_build_routing,
            ) as build_routing,
        ):
            builder(context)

        self.assertEqual(build_routing.call_count, 2)
        self.assertEqual(seen_routings[0].tolist(), [[0, -1], [1, -1]])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestSglLoraDeepGemmBuilderGraphMutation(unittest.TestCase):
    def test_one_graph_replays_base_mixed_and_active_adapter_mappings(self):
        torch.manual_seed(19)
        device = torch.device("cuda")
        tokens, topk, experts = 4, 2, 2
        hidden_size, intermediate, rank, adapters = 32, 32, 16, 2
        hidden = torch.randn(tokens, hidden_size, dtype=torch.bfloat16, device=device)
        topk_ids = torch.tensor(
            [[0, 1], [1, 0], [0, 1], [1, 0]],
            dtype=torch.int32,
            device=device,
        )
        topk_weights = torch.rand(tokens, topk, device=device)
        token_lora_mapping = torch.full((tokens,), -1, dtype=torch.int32, device=device)
        context = _context(
            hidden=hidden,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_lora_mapping=token_lora_mapping,
            shared_outer=False,
            experts=experts,
            adapters=adapters,
            rank=rank,
            intermediate=intermediate,
        )
        builder = DeepGemmBf16LoRAHookBuilder(_launch_config())
        pair_to_provider_row = torch.tensor(
            [0, 4, 5, 1, 2, 6, 7, 3],
            dtype=torch.int32,
            device=device,
        )
        provider_base = torch.randn(
            experts,
            4,
            2 * intermediate,
            dtype=torch.bfloat16,
            device=device,
        )
        provider_gate_up = torch.empty_like(provider_base)
        provider_activation = torch.randn(
            experts,
            4,
            intermediate,
            dtype=torch.bfloat16,
            device=device,
        )
        down_result = torch.empty(
            tokens,
            topk,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )

        def run_hooks():
            hooks = builder(context)
            provider_gate_up.copy_(provider_base)
            hooks.inject_gate_up(
                GateUpHookContext(
                    provider_gate_up=provider_gate_up,
                    pair_to_provider_row=pair_to_provider_row,
                    provider_layout="gate_then_up",
                )
            )
            contribution = hooks.build_down_pair_contribution(
                DownHookContext(
                    provider_activation=provider_activation,
                    pair_to_provider_row=pair_to_provider_row,
                )
            )
            down_result.copy_(contribution.values)

        mappings = (
            torch.tensor([-1, -1, -1, -1], dtype=torch.int32, device=device),
            torch.tensor([0, -1, 1, -1], dtype=torch.int32, device=device),
            torch.tensor([0, 1, 0, 1], dtype=torch.int32, device=device),
        )
        eager = []
        for mapping in mappings:
            token_lora_mapping.copy_(mapping)
            run_hooks()
            torch.cuda.synchronize()
            eager.append((provider_gate_up.clone(), down_result.clone()))

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            run_hooks()
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_hooks()
        for mapping, (expected_gate, expected_down) in zip(mappings, eager):
            token_lora_mapping.copy_(mapping)
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(provider_gate_up, expected_gate))
            self.assertTrue(torch.equal(down_result, expected_down))

        token_lora_mapping.fill_(-1)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(provider_gate_up, provider_base))
        self.assertTrue(torch.equal(down_result, torch.zeros_like(down_result)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
