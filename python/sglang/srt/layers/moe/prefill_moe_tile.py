"""Bound local NPU prefill MoE activations after routing has been computed."""

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatcher
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import (
    get_moe_a2a_backend,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)
from sglang.srt.runtime_context import get_disagg


def can_tile_prefill_moe(layer, inputs, routes, tile_tokens: int) -> bool:
    experts = layer.experts
    return (
        tile_tokens > 0
        and inputs.shape[0] > tile_tokens
        and inputs.device.type == "npu"
        and get_disagg().disaggregation_mode == "prefill"
        and isinstance(routes, StandardTopKOutput)
        and isinstance(experts.dispatcher, StandardDispatcher)
        and get_moe_a2a_backend().is_none()
        and not should_use_flashinfer_cutlass_moe_fp4_allgather()
        and experts.moe_ep_size == 1
        and not experts.reduce_results
        and not experts._dwdp_bound
        and not layer._fuse_shared_experts_inside_sbo
        and (
            routes.router_logits is None
            or routes.router_logits.shape[0] == inputs.shape[0]
        )
    )


def tiled_experts(experts, inputs, routes, tile_tokens: int):
    if tile_tokens <= 0:
        raise ValueError("MoE tile size must be positive")
    output = torch.empty_like(inputs)
    for start in range(0, inputs.shape[0], tile_tokens):
        rows = slice(start, start + tile_tokens)
        tile_routes = routes._replace(
            topk_weights=routes.topk_weights[rows],
            topk_ids=routes.topk_ids[rows],
            router_logits=(
                None if routes.router_logits is None else routes.router_logits[rows]
            ),
        )
        result = experts(inputs[rows], tile_routes)
        if result.shape != output[rows].shape or result.dtype != output.dtype:
            raise RuntimeError(
                "Local prefill MoE must preserve activation shape and dtype"
            )
        output[rows].copy_(result)
    return output


def run_prefill_experts(layer, inputs, routes):
    tile_tokens = envs.SGLANG_NPU_PREFILL_MOE_TILE_TOKENS.get()
    if tile_tokens < 0:
        raise ValueError("SGLANG_NPU_PREFILL_MOE_TILE_TOKENS must be nonnegative")
    if can_tile_prefill_moe(layer, inputs, routes, tile_tokens):
        return tiled_experts(layer.experts, inputs, routes, tile_tokens)
    return layer.experts(inputs, routes)
