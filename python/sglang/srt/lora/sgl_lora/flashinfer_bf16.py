"""Public FlashInfer BF16 routed-MoE adapter for SGL LoRA.

This module owns only the provider boundary. Kernel schedules are supplied by
the caller and remain candidates until end-to-end benchmarking selects policy.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from sglang.kernels.ops.moe.pack_topk_ids import PackTopkIds
from sglang.srt.lora.sgl_lora.bf16 import (
    grouped_lora_a,
    stock_grouped_lora_b,
    token_owned_lora_b_add,
)
from sglang.srt.lora.sgl_lora.routing import build_virtual_expert_routing
from sglang.srt.utils.common import next_power_of_2


def _get_packed_topk_ids(topk_output: Any) -> torch.Tensor:
    packed_topk_ids = getattr(topk_output, "packed_topk_ids", None)
    if packed_topk_ids is not None:
        return packed_topk_ids
    return PackTopkIds.execute(
        topk_output.topk_ids.contiguous(),
        topk_output.topk_weights.contiguous(),
    )


def build_flashinfer_bf16_lora_factor_maps(
    *,
    global_num_experts: int,
    local_expert_offset: int,
    local_num_experts: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map FlashInfer global expert IDs to EP-local LoRA factor slots."""
    stop = local_expert_offset + local_num_experts
    factor_map = torch.full((global_num_experts,), -1, dtype=torch.int32, device=device)
    factor_map[local_expert_offset:stop] = torch.arange(
        local_num_experts,
        dtype=torch.int32,
        device=device,
    )
    shared_factor_map = torch.full_like(factor_map, -1)
    shared_factor_map[local_expert_offset:stop] = 0
    return factor_map, shared_factor_map


def _allocate_finalized_output(hidden_states: torch.Tensor) -> torch.Tensor:
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        use_symmetric_memory,
    )
    from sglang.srt.layers.dp_attention import is_allocation_symmetric

    with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):
        return torch.empty_like(
            hidden_states, dtype=torch.bfloat16, memory_format=torch.contiguous_format
        )


def _routing_method_type(runner_config: Any) -> int:
    from sglang.srt.layers.moe.utils import RoutingMethodType

    routing_method = runner_config.routing_method_type
    if routing_method is None:
        routing_method = RoutingMethodType.Default
    elif routing_method == RoutingMethodType.DeepSeekV3:
        # Routing is already materialized in packed top-k form.
        routing_method = RoutingMethodType.TopK
    return int(routing_method)


def _invoke_public_flashinfer_bf16(
    *,
    packed_topk_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    quant_info: Any,
    runner_config: Any,
    gate_up_delta: torch.Tensor,
    output: torch.Tensor,
    enable_pdl: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from flashinfer.fused_moe import (
        ActivationType,
        WeightLayout,
        trtllm_bf16_routed_moe,
    )

    result = trtllm_bf16_routed_moe(
        topk_ids=packed_topk_ids,
        hidden_states=hidden_states,
        gemm1_weights=quant_info.gemm1_weights,
        gemm2_weights=quant_info.gemm2_weights,
        num_experts=quant_info.global_num_experts,
        top_k=packed_topk_ids.shape[1],
        n_group=None,
        topk_group=None,
        intermediate_size=runner_config.intermediate_size_per_partition,
        local_expert_offset=quant_info.local_expert_offset,
        local_num_experts=runner_config.num_local_experts,
        # Packed TopK routing carries the final BF16 combine coefficients.
        # Passing a second scale is currently inert in FlashInfer and would be
        # ambiguous if that provider contract changed.
        routed_scaling_factor=1.0,
        routing_method_type=_routing_method_type(runner_config),
        use_shuffled_weight=True,
        weight_layout=WeightLayout.BlockMajorK,
        do_finalize=True,
        enable_pdl=enable_pdl,
        gemm1_lora_delta=gate_up_delta,
        tune_max_num_tokens=next_power_of_2(hidden_states.shape[0]),
        activation_type=ActivationType.Swiglu.value,
        output=output,
    )
    return result[0], result[1], result[2]


def run_flashinfer_bf16_lora(
    dispatch_output: Any,
    quant_info: Any,
    runner_config: Any,
    lora_info: Any,
    *,
    routing_block_size: int,
    lora_a_config: Mapping[str, int],
    lora_b_config: Mapping[str, int],
    routed_expert_to_factor_id: torch.Tensor | None = None,
    routed_expert_to_shared_factor_id: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
    enable_pdl: bool = True,
) -> Any:
    """Run one finalized public-FlashInfer BF16 MoE plus SGL LoRA.

    The delta argument is always resident, including base-only batches, so the
    caller can capture one CUDA graph for base, mixed, and active requests.
    """
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
    )

    hidden_states = dispatch_output.hidden_states
    resolved_output_dtype = (
        hidden_states.dtype if output_dtype is None else output_dtype
    )
    if resolved_output_dtype != torch.bfloat16:
        raise TypeError(
            "finalized public FlashInfer BF16 MoE requires BF16 output; "
            "FP32 requires the deferred-finalize candidate"
        )
    if runner_config.activation != "silu" or not runner_config.is_gated:
        raise NotImplementedError(
            "FlashInfer GEMM1 LoRA delta currently requires gated SiLU"
        )

    topk_output = dispatch_output.topk_output
    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights
    num_tokens, top_k = topk_ids.shape
    num_pairs = num_tokens * top_k
    intermediate_size = runner_config.intermediate_size_per_partition
    shared_outer = bool(lora_info.experts_shared_outer_loras)
    if shared_outer and routed_expert_to_shared_factor_id is None:
        raise ValueError("shared-outer LoRA requires a resident shared factor map")

    def route_for(weight: torch.Tensor, factor_map: torch.Tensor | None):
        return build_virtual_expert_routing(
            topk_ids,
            lora_info.token_lora_mapping,
            factor_expert_count=weight.shape[1],
            max_loras=weight.shape[0],
            block_size=routing_block_size,
            routed_expert_to_factor_id=factor_map,
        )

    shared_map = routed_expert_to_shared_factor_id if shared_outer else None
    expert_route = route_for(
        lora_info.gate_up_lora_b_weights,
        routed_expert_to_factor_id,
    )
    outer_route = (
        route_for(lora_info.gate_up_lora_a_weights, shared_map)
        if shared_outer
        else expert_route
    )

    gate_a_output = torch.empty(
        (num_pairs, lora_info.gate_up_lora_a_weights.shape[2]),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    grouped_lora_a(
        hidden_states,
        lora_info.gate_up_lora_a_weights.flatten(0, 1),
        gate_a_output,
        outer_route,
        config=lora_a_config,
    )

    factor_intermediate_size = lora_info.gate_up_lora_b_weights.shape[2] // 2
    gate_up_delta_factory = (
        torch.empty if factor_intermediate_size == intermediate_size else torch.zeros
    )
    gate_up_delta = gate_up_delta_factory(
        (num_pairs, 2 * intermediate_size),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    # SGL factors are [gate, up]; FlashInfer computes silu(second) * first.
    stock_grouped_lora_b(
        gate_a_output,
        lora_info.gate_up_lora_b_weights.flatten(0, 1),
        gate_up_delta,
        expert_route,
        destination_offsets=(intermediate_size, 0),
        config=lora_b_config,
    )

    output = _allocate_finalized_output(hidden_states)
    output, pair_to_provider_row, provider_activation = _invoke_public_flashinfer_bf16(
        packed_topk_ids=_get_packed_topk_ids(topk_output),
        hidden_states=hidden_states,
        quant_info=quant_info,
        runner_config=runner_config,
        gate_up_delta=gate_up_delta.view(num_tokens, top_k, 2 * intermediate_size),
        output=output,
        enable_pdl=enable_pdl,
    )

    down_a_output = torch.empty(
        (num_pairs, lora_info.down_lora_a_weights.shape[2]),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    grouped_lora_a(
        provider_activation,
        lora_info.down_lora_a_weights.flatten(0, 1),
        down_a_output,
        expert_route,
        config=lora_a_config,
        input_row_map=pair_to_provider_row,
    )

    token_owned_lora_b_add(
        down_a_output,
        lora_info.down_lora_b_weights.flatten(0, 1),
        output,
        outer_route,
        topk_weights,
        config=lora_b_config,
    )
    return StandardCombineInput(hidden_states=output)


__all__ = [
    "build_flashinfer_bf16_lora_factor_maps",
    "run_flashinfer_bf16_lora",
]
