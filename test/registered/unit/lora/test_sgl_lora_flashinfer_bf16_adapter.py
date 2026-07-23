"""CPU behavior tests for the public FlashInfer BF16 SGL-LoRA adapter."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import sglang.srt.lora.sgl_lora.flashinfer_bf16 as adapter_module
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.lora.layers import FusedMoEWithLoRA
from sglang.srt.lora.sgl_lora.bf16 import Bf16MoeLaunchConfig
from sglang.srt.lora.sgl_lora.flashinfer_bf16 import (
    build_flashinfer_bf16_lora_factor_maps,
    run_flashinfer_bf16_lora,
)


def _kernel_config() -> dict[str, int]:
    return {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 16,
        "BLOCK_SIZE_K": 16,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2,
    }


def _case(
    *,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    num_experts: int,
    local_expert_offset: int = 0,
    local_num_experts: int | None = None,
    hidden_size: int = 3,
    intermediate_size: int = 2,
    rank: int = 2,
    routed_scaling_factor: float = 1.0,
) -> tuple[SimpleNamespace, SimpleNamespace, MoeRunnerConfig, SimpleNamespace]:
    adapters = 2
    if local_num_experts is None:
        local_num_experts = num_experts
    dispatch = SimpleNamespace(
        hidden_states=torch.ones(topk_ids.shape[0], hidden_size, dtype=torch.bfloat16),
        topk_output=SimpleNamespace(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            packed_topk_ids=topk_ids,
        ),
    )
    quant_info = SimpleNamespace(
        gemm1_weights=torch.empty(0),
        gemm2_weights=torch.empty(0),
        global_num_experts=num_experts,
        local_expert_offset=local_expert_offset,
    )
    config = MoeRunnerConfig(
        num_experts=num_experts,
        num_local_experts=local_num_experts,
        hidden_size=hidden_size,
        intermediate_size_per_partition=intermediate_size,
        top_k=topk_ids.shape[1],
        routed_scaling_factor=routed_scaling_factor,
        activation="silu",
        is_gated=True,
    )
    lora_info = SimpleNamespace(
        gate_up_lora_a_weights=torch.ones(
            adapters,
            local_num_experts,
            2 * rank,
            hidden_size,
            dtype=torch.bfloat16,
        ),
        gate_up_lora_b_weights=torch.ones(
            adapters,
            local_num_experts,
            2 * intermediate_size,
            rank,
            dtype=torch.bfloat16,
        ),
        down_lora_a_weights=torch.ones(
            adapters,
            local_num_experts,
            rank,
            intermediate_size,
            dtype=torch.bfloat16,
        ),
        down_lora_b_weights=torch.zeros(
            adapters,
            local_num_experts,
            hidden_size,
            rank,
            dtype=torch.bfloat16,
        ),
        token_lora_mapping=token_lora_mapping,
        experts_shared_outer_loras=False,
    )
    return dispatch, quant_info, config, lora_info


def _fill_valid_a(_input, _weight, output, routing, **_kwargs):
    output.zero_()
    output[routing.virtual_topk_ids.reshape(-1) >= 0] = 1


def _write_distinct_gate_up(
    _intermediate,
    weight,
    destination,
    routing,
    *,
    destination_offsets,
    **_kwargs,
):
    destination.zero_()
    valid = routing.virtual_topk_ids.reshape(-1) >= 0
    width = weight.shape[1] // len(destination_offsets)
    for slice_id, offset in enumerate(destination_offsets):
        value = 3 if slice_id == 0 else 7
        destination[valid, offset : offset + width] = value


def _provider_result(*, output, gate_up_delta, **_kwargs):
    output.fill_(5)
    num_pairs = gate_up_delta.shape[0] * gate_up_delta.shape[1]
    pair_to_provider_row = torch.arange(num_pairs, dtype=torch.int32)
    provider_activation = torch.ones(
        num_pairs,
        gate_up_delta.shape[2] // 2,
        dtype=torch.bfloat16,
    )
    return output, pair_to_provider_row, provider_activation


class TestSglLoraFlashInferBf16Adapter(unittest.TestCase):
    def test_layer_dispatches_flashinfer_with_provider_neutral_config(self):
        hidden = torch.ones(2, 3, dtype=torch.bfloat16)
        topk_output = SimpleNamespace()
        dispatch_output = SimpleNamespace()
        provider_result = StandardCombineInput(torch.full_like(hidden, 4))
        final_result = torch.full_like(hidden, 5)
        dispatcher = SimpleNamespace(
            dispatch=Mock(return_value=dispatch_output),
            combine=Mock(return_value=final_result),
        )
        finalize = Mock(return_value=final_result)
        runner_config = SimpleNamespace()
        factor_map = torch.tensor([0, 1], dtype=torch.int32)
        shared_map = torch.tensor([0, 0], dtype=torch.int32)
        launch_config = Bf16MoeLaunchConfig(
            routing_block_size=16,
            lora_a=_kernel_config(),
            lora_b=_kernel_config(),
        )
        layer = object.__new__(FusedMoEWithLoRA)
        torch.nn.Module.__init__(layer)
        layer.base_layer = SimpleNamespace(
            dispatcher=dispatcher,
            moe_runner_config=runner_config,
            combine_and_finalize=finalize,
        )
        layer.lora_execution_engine = "sgl_lora"
        layer._lora_runner_backend = MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED
        layer._quant_info = SimpleNamespace()
        layer._sgl_lora_factor_maps = (factor_map, shared_map)
        layer._sgl_lora_launch_config = launch_config
        lora_info = SimpleNamespace()

        with patch.object(
            adapter_module,
            "run_flashinfer_bf16_lora",
            return_value=provider_result,
        ) as run_provider:
            result = layer._forward_with_lora(
                hidden,
                topk_output,
                lora_info,
                output_dtype=torch.bfloat16,
            )

        self.assertIs(result, final_result)
        dispatcher.dispatch.assert_called_once_with(
            hidden_states=hidden,
            topk_output=topk_output,
        )
        run_provider.assert_called_once_with(
            dispatch_output,
            layer._quant_info,
            runner_config,
            lora_info,
            routing_block_size=16,
            lora_a_config=launch_config.lora_a,
            lora_b_config=launch_config.lora_b,
            routed_expert_to_factor_id=factor_map,
            routed_expert_to_shared_factor_id=shared_map,
            output_dtype=torch.bfloat16,
        )
        finalize.assert_called_once_with(
            combine_input=provider_result,
            origin_hidden_states_dim=hidden.shape[-1],
        )
        dispatcher.combine.assert_not_called()

    def test_gate_up_order_and_resident_delta_for_all_traffic_modes(self):
        topk_ids = torch.tensor([[4, 5], [5, 6]], dtype=torch.int32)
        topk_weights = torch.tensor([[0.625, 0.375], [0.75, 0.25]], dtype=torch.float32)
        factor_map, _ = build_flashinfer_bf16_lora_factor_maps(
            global_num_experts=7,
            local_expert_offset=4,
            local_num_experts=3,
            device=torch.device("cpu"),
        )
        modes = {
            "base": torch.tensor([-1, -1], dtype=torch.int32),
            "mixed": torch.tensor([0, -1], dtype=torch.int32),
            "active": torch.tensor([0, 1], dtype=torch.int32),
        }

        for mode, token_mapping in modes.items():
            with self.subTest(mode=mode):
                dispatch, quant_info, config, lora_info = _case(
                    topk_ids=topk_ids,
                    topk_weights=topk_weights,
                    token_lora_mapping=token_mapping,
                    num_experts=7,
                    local_expert_offset=4,
                    local_num_experts=3,
                )
                captured = {}

                def provider(**kwargs):
                    captured["delta"] = kwargs["gate_up_delta"].clone()
                    return _provider_result(**kwargs)

                with (
                    patch.object(
                        adapter_module,
                        "grouped_lora_a",
                        side_effect=_fill_valid_a,
                    ),
                    patch.object(
                        adapter_module,
                        "stock_grouped_lora_b",
                        side_effect=_write_distinct_gate_up,
                    ),
                    patch.object(
                        adapter_module,
                        "_allocate_finalized_output",
                        return_value=torch.empty_like(dispatch.hidden_states),
                    ),
                    patch.object(
                        adapter_module,
                        "_invoke_public_flashinfer_bf16",
                        side_effect=provider,
                    ),
                ):
                    result = run_flashinfer_bf16_lora(
                        dispatch,
                        quant_info,
                        config,
                        lora_info,
                        routing_block_size=16,
                        lora_a_config=_kernel_config(),
                        lora_b_config=_kernel_config(),
                        routed_expert_to_factor_id=factor_map,
                    )

                delta = captured["delta"].reshape(-1, 4)
                valid = token_mapping[:, None].expand(-1, 2).reshape(-1) >= 0
                torch.testing.assert_close(
                    delta[valid, :2],
                    torch.full_like(delta[valid, :2], 7),
                )
                torch.testing.assert_close(
                    delta[valid, 2:],
                    torch.full_like(delta[valid, 2:], 3),
                )
                torch.testing.assert_close(
                    delta[~valid],
                    torch.zeros_like(delta[~valid]),
                )
                torch.testing.assert_close(
                    result.hidden_states,
                    torch.full_like(result.hidden_states, 5),
                )

    def test_down_route_weights_are_bf16_rounded_and_applied_once(self):
        topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
        topk_weights = torch.tensor([[0.3, 0.6]], dtype=torch.float32)
        dispatch, quant_info, config, lora_info = _case(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_lora_mapping=torch.tensor([0], dtype=torch.int32),
            num_experts=2,
            routed_scaling_factor=3.0,
        )
        lora_info.down_lora_b_weights.fill_(1)

        def provider(**kwargs):
            output, pair_map, activation = _provider_result(**kwargs)
            output.zero_()
            return output, pair_map, activation

        with (
            patch.object(
                adapter_module,
                "grouped_lora_a",
                side_effect=_fill_valid_a,
            ),
            patch.object(
                adapter_module,
                "stock_grouped_lora_b",
                side_effect=_write_distinct_gate_up,
            ),
            patch.object(
                adapter_module,
                "_allocate_finalized_output",
                return_value=torch.empty_like(dispatch.hidden_states),
            ),
            patch.object(
                adapter_module,
                "_invoke_public_flashinfer_bf16",
                side_effect=provider,
            ),
        ):
            result = run_flashinfer_bf16_lora(
                dispatch,
                quant_info,
                config,
                lora_info,
                routing_block_size=16,
                lora_a_config=_kernel_config(),
                lora_b_config=_kernel_config(),
            )

        expected = (
            topk_weights.to(torch.bfloat16).float().sum()
            * lora_info.down_lora_a_weights.shape[2]
        )
        torch.testing.assert_close(
            result.hidden_states.float(),
            torch.full_like(result.hidden_states.float(), expected),
        )

    def test_provider_padding_is_zero_and_tp_slices_use_factor_width(self):
        topk_ids = torch.tensor([[0]], dtype=torch.int32)
        dispatch, quant_info, config, lora_info = _case(
            topk_ids=topk_ids,
            topk_weights=torch.ones_like(topk_ids, dtype=torch.float32),
            token_lora_mapping=torch.tensor([0], dtype=torch.int32),
            num_experts=1,
            intermediate_size=4,
        )
        lora_info.gate_up_lora_b_weights = lora_info.gate_up_lora_b_weights[
            :, :, :6, :
        ].contiguous()
        lora_info.down_lora_a_weights = lora_info.down_lora_a_weights[
            :, :, :, :3
        ].contiguous()
        captured = {}

        def write_semantic_slices(
            _intermediate,
            weight,
            destination,
            _routing,
            *,
            destination_offsets,
            **_kwargs,
        ):
            width = weight.shape[1] // len(destination_offsets)
            for slice_id, offset in enumerate(destination_offsets):
                destination[:, offset : offset + width] = 3 + 4 * slice_id

        def provider(**kwargs):
            captured["delta"] = kwargs["gate_up_delta"].clone()
            return _provider_result(**kwargs)

        with (
            patch.object(adapter_module, "grouped_lora_a", side_effect=_fill_valid_a),
            patch.object(
                adapter_module,
                "stock_grouped_lora_b",
                side_effect=write_semantic_slices,
            ),
            patch.object(
                adapter_module,
                "_allocate_finalized_output",
                return_value=torch.empty_like(dispatch.hidden_states),
            ),
            patch.object(
                adapter_module,
                "_invoke_public_flashinfer_bf16",
                side_effect=provider,
            ),
        ):
            run_flashinfer_bf16_lora(
                dispatch,
                quant_info,
                config,
                lora_info,
                routing_block_size=16,
                lora_a_config=_kernel_config(),
                lora_b_config=_kernel_config(),
            )

        torch.testing.assert_close(
            captured["delta"].reshape(-1),
            torch.tensor([7, 7, 7, 0, 3, 3, 3, 0], dtype=torch.bfloat16),
        )

        layer = object.__new__(FusedMoEWithLoRA)
        torch.nn.Module.__init__(layer)
        layer.tp_size = 2
        layer.intermediate_size_per_partition = 128
        layer._uses_interleaved_gate_up = False
        layer.base_layer = SimpleNamespace(
            moe_runner_config=SimpleNamespace(is_gated=True)
        )
        down_a = torch.arange(2 * 192).reshape(2, 192)
        gate_up_b = torch.arange(2 * 192 * 2).reshape(2 * 192, 2)
        torch.testing.assert_close(
            layer.slice_moe_lora_a_weights(down_a, 1, "down_proj_moe"),
            down_a[:, 96:192],
        )
        torch.testing.assert_close(
            layer.slice_moe_lora_b_weights(gate_up_b, 1, "gate_up_proj_moe"),
            torch.cat((gate_up_b[96:192], gate_up_b[288:384])),
        )

    def test_ep_factor_maps_localize_owned_ids_and_mask_nonlocal_ids(self):
        per_expert, shared = build_flashinfer_bf16_lora_factor_maps(
            global_num_experts=8,
            local_expert_offset=4,
            local_num_experts=2,
            device=torch.device("cpu"),
        )
        torch.testing.assert_close(
            per_expert,
            torch.tensor([-1, -1, -1, -1, 0, 1, -1, -1], dtype=torch.int32),
        )
        torch.testing.assert_close(
            shared,
            torch.tensor([-1, -1, -1, -1, 0, 0, -1, -1], dtype=torch.int32),
        )

    def test_explicit_fp32_output_is_rejected_before_provider_launch(self):
        dispatch = SimpleNamespace(
            hidden_states=torch.empty(1, 4, dtype=torch.bfloat16)
        )
        with self.assertRaisesRegex(TypeError, "deferred-finalize"):
            run_flashinfer_bf16_lora(
                dispatch,
                None,
                None,
                None,
                routing_block_size=16,
                lora_a_config=_kernel_config(),
                lora_b_config=_kernel_config(),
                output_dtype=torch.float32,
            )


if __name__ == "__main__":
    unittest.main()
