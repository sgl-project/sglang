"""SGL LoRA hooks for the stock masked-BF16 DeepGEMM MoE runner."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.lora.sgl_lora.bf16 import (
    Bf16MoeLaunchConfig,
    grouped_lora_a,
    stock_grouped_lora_b,
)
from sglang.srt.lora.sgl_lora.hooks import (
    DownHookContext,
    GateUpHookContext,
    MoeLoRAHookBuildContext,
    PairDomainLoRAContribution,
    SglMoeLoRAHooks,
)
from sglang.srt.lora.sgl_lora.routing import (
    VirtualExpertRouting,
    build_virtual_expert_routing,
)


@triton.jit
def _scatter_add_pair_rows_kernel(
    pair_delta_ptr,
    provider_ptr,
    pair_to_provider_row_ptr,
    virtual_ids_ptr,
    num_pairs,
    width,
    stride_dm,
    stride_dn,
    stride_pm,
    stride_pn,
    BLOCK_N: tl.constexpr,
):
    pair_id = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    active = (pair_id < num_pairs) & (
        tl.load(virtual_ids_ptr + pair_id, mask=pair_id < num_pairs, other=-1) >= 0
    )
    provider_row = tl.load(
        pair_to_provider_row_ptr + pair_id,
        mask=active,
        other=0,
    ).to(tl.int64)
    mask = active & (columns < width)
    delta = tl.load(
        pair_delta_ptr + pair_id * stride_dm + columns * stride_dn,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    destination = provider_ptr + provider_row * stride_pm + columns * stride_pn
    base = tl.load(destination, mask=mask, other=0.0).to(tl.float32)
    tl.store(destination, base + delta, mask=mask)


def scatter_add_pair_rows(
    pair_delta: torch.Tensor,
    provider_output: torch.Tensor,
    pair_to_provider_row: torch.Tensor,
    routing: VirtualExpertRouting,
) -> None:
    """Add active canonical pair rows to an injectively mapped provider output."""
    provider_rows = provider_output.view(-1, provider_output.shape[-1])
    num_pairs, width = pair_delta.shape
    if num_pairs == 0:
        return

    if not pair_delta.is_cuda:
        active = routing.virtual_topk_ids.reshape(-1) >= 0
        rows = pair_to_provider_row.reshape(-1)[active].to(torch.int64)
        provider_rows[rows] = (
            provider_rows[rows].float() + pair_delta[active].float()
        ).to(provider_rows.dtype)
        return

    block_n = 256
    _scatter_add_pair_rows_kernel[(num_pairs, triton.cdiv(width, block_n))](
        pair_delta,
        provider_rows,
        pair_to_provider_row,
        routing.virtual_topk_ids,
        num_pairs,
        width,
        pair_delta.stride(0),
        pair_delta.stride(1),
        provider_rows.stride(0),
        provider_rows.stride(1),
        BLOCK_N=block_n,
        num_warps=4,
    )


def _flatten_factors(weight: torch.Tensor) -> torch.Tensor:
    return weight.reshape(
        weight.shape[0] * weight.shape[1], weight.shape[2], weight.shape[3]
    )


class DeepGemmBf16LoRAHookBuilder:
    """Build deterministic LoRA hooks while canonical dispatch tensors exist."""

    def __init__(self, launch_config: Bf16MoeLaunchConfig):
        self.launch_config = launch_config
        if launch_config.lora_b["BLOCK_SIZE_M"] != launch_config.routing_block_size:
            raise ValueError("LoRA-B and routing block sizes must match")
        self._shared_factor_maps: dict[
            tuple[torch.device, torch.dtype, int], torch.Tensor
        ] = {}

    def _factor_map(
        self,
        topk_ids: torch.Tensor,
        *,
        factor_count: int,
        physical_factor_domain: int,
        shared: bool,
    ) -> torch.Tensor | None:
        if not shared:
            return None
        if factor_count != 1:
            raise ValueError("shared outer LoRA factors must have factor dimension 1")
        key = (topk_ids.device, topk_ids.dtype, physical_factor_domain)
        factor_map = self._shared_factor_maps.get(key)
        if factor_map is None:
            # The map is deliberately bounded by the provider's EP-local expert
            # domain. Global, nonlocal, and physical shared IDs outside it map -1.
            factor_map = torch.zeros(
                physical_factor_domain,
                dtype=topk_ids.dtype,
                device=topk_ids.device,
            )
            self._shared_factor_maps[key] = factor_map
        return factor_map

    def _routing(
        self,
        topk_ids: torch.Tensor,
        token_lora_mapping: torch.Tensor,
        weight: torch.Tensor,
        *,
        physical_factor_domain: int,
        shared: bool,
    ) -> VirtualExpertRouting:
        factor_count = weight.shape[1]
        factor_map = self._factor_map(
            topk_ids,
            factor_count=factor_count,
            physical_factor_domain=physical_factor_domain,
            shared=shared,
        )
        return build_virtual_expert_routing(
            topk_ids,
            token_lora_mapping,
            factor_expert_count=factor_count,
            max_loras=weight.shape[0],
            block_size=self.launch_config.routing_block_size,
            routed_expert_to_factor_id=factor_map,
        )

    def __call__(self, context: MoeLoRAHookBuildContext) -> SglMoeLoRAHooks:
        if context.runner_backend != MoeRunnerBackend.DEEP_GEMM:
            raise NotImplementedError("this hook builder requires DeepGEMM")

        dispatch_output = context.dispatch_output
        hidden_states = dispatch_output.hidden_states
        device = hidden_states.device
        topk_ids = dispatch_output.topk_output.topk_ids
        lora_info = context.lora_info
        shared_outer = bool(lora_info.experts_shared_outer_loras)

        gate_a_weight = lora_info.gate_up_lora_a_weights
        gate_b_weight = lora_info.gate_up_lora_b_weights
        down_a_weight = lora_info.down_lora_a_weights
        down_b_weight = lora_info.down_lora_b_weights
        token_lora_mapping = lora_info.token_lora_mapping

        routing_cache: dict[tuple[int, int, int, bool], VirtualExpertRouting] = {}

        def routing_for(
            weight: torch.Tensor, physical_factor_domain: int, shared: bool
        ) -> VirtualExpertRouting:
            key = (
                weight.shape[0],
                weight.shape[1],
                physical_factor_domain,
                shared,
            )
            if key not in routing_cache:
                routing_cache[key] = self._routing(
                    topk_ids,
                    token_lora_mapping,
                    weight,
                    physical_factor_domain=physical_factor_domain,
                    shared=shared,
                )
            return routing_cache[key]

        gate_a_routing = routing_for(
            gate_a_weight, gate_b_weight.shape[1], shared_outer
        )
        gate_b_routing = routing_for(gate_b_weight, gate_b_weight.shape[1], False)
        down_a_routing = routing_for(down_a_weight, down_a_weight.shape[1], False)
        down_b_routing = routing_for(
            down_b_weight, down_a_weight.shape[1], shared_outer
        )

        num_pairs = topk_ids.numel()
        gate_a_output = torch.empty(
            num_pairs,
            gate_a_weight.shape[2],
            dtype=torch.bfloat16,
            device=device,
        )
        # This launch must happen before provider pre-permute disposes hidden_states.
        grouped_lora_a(
            hidden_states,
            _flatten_factors(gate_a_weight),
            gate_a_output,
            gate_a_routing,
            config=self.launch_config.lora_a,
        )

        def inject_gate_up(hook_context: GateUpHookContext) -> None:
            if hook_context.provider_layout != "gate_then_up":
                raise NotImplementedError("DeepGEMM requires [gate, up] output")
            pair_delta = torch.empty(
                num_pairs,
                gate_b_weight.shape[2],
                dtype=torch.bfloat16,
                device=device,
            )
            slice_width = gate_b_weight.shape[2] // 2
            stock_grouped_lora_b(
                gate_a_output,
                _flatten_factors(gate_b_weight),
                pair_delta,
                gate_b_routing,
                destination_offsets=(0, slice_width),
                config=self.launch_config.lora_b,
            )
            scatter_add_pair_rows(
                pair_delta,
                hook_context.provider_gate_up,
                hook_context.pair_to_provider_row,
                gate_b_routing,
            )

        def build_down_pair_contribution(
            hook_context: DownHookContext,
        ) -> PairDomainLoRAContribution:
            provider_rows = hook_context.provider_activation.view(
                -1, hook_context.provider_activation.shape[-1]
            )
            down_a_output = torch.empty(
                num_pairs,
                down_a_weight.shape[2],
                dtype=torch.bfloat16,
                device=provider_rows.device,
            )
            grouped_lora_a(
                provider_rows,
                _flatten_factors(down_a_weight),
                down_a_output,
                down_a_routing,
                config=self.launch_config.lora_a,
                input_row_map=hook_context.pair_to_provider_row,
            )
            pair_delta = torch.empty(
                num_pairs,
                down_b_weight.shape[2],
                dtype=torch.bfloat16,
                device=provider_rows.device,
            )
            stock_grouped_lora_b(
                down_a_output,
                _flatten_factors(down_b_weight),
                pair_delta,
                down_b_routing,
                destination_offsets=(0,),
                config=self.launch_config.lora_b,
            )
            return PairDomainLoRAContribution(
                values=pair_delta.view(
                    topk_ids.shape[0], topk_ids.shape[1], pair_delta.shape[1]
                ),
            )

        return SglMoeLoRAHooks(
            inject_gate_up=inject_gate_up,
            build_down_pair_contribution=build_down_pair_contribution,
        )


__all__ = [
    "Bf16MoeLaunchConfig",
    "DeepGemmBf16LoRAHookBuilder",
    "scatter_add_pair_rows",
]
