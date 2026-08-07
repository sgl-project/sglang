"""SharedEP-owned route and compute kernel entry points."""

from __future__ import annotations

import msgspec
import torch
import triton
import triton.language as tl

from sglang.srt.layers.moe.shared_ep.layout import SharedEpInputViews
from sglang.srt.layers.moe.shared_ep.profiles import SharedEpProfile


class RoutePreparation(msgspec.Struct, kw_only=True):
    local_ids: torch.Tensor
    local_weights: torch.Tensor
    sorted_token_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens_post_padded: torch.Tensor


@triton.jit
def _quantize_pack_input_kernel(
    source,
    source_ids,
    source_weights,
    target_q,
    target_scales,
    target_ids,
    target_weights,
    source_q_row_stride: tl.constexpr,
    source_id_row_stride: tl.constexpr,
    source_weight_row_stride: tl.constexpr,
    target_q_row_stride: tl.constexpr,
    target_scale_row_stride: tl.constexpr,
    target_id_row_stride: tl.constexpr,
    target_weight_row_stride: tl.constexpr,
    quant_programs: tl.constexpr,
    num_tokens: tl.constexpr,
    num_groups: tl.constexpr,
    top_k: tl.constexpr,
    max_tokens: tl.constexpr,
    group_size: tl.constexpr,
    fp8_max: tl.constexpr,
    eps: tl.constexpr,
):
    program = tl.program_id(0)
    quant_valid = program < quant_programs
    token = program // num_groups
    group = program % num_groups
    columns = group * group_size + tl.arange(0, group_size)
    value = tl.load(
        source + token * source_q_row_stride + columns,
        mask=quant_valid,
        other=0.0,
    ).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(value), axis=0), eps)
    scale = absmax / fp8_max
    quantized = tl.clamp(value / scale, -fp8_max, fp8_max).to(tl.float8e4nv)
    tl.store(
        target_q + token * target_q_row_stride + columns,
        quantized,
        mask=quant_valid,
    )
    tl.store(
        target_scales + token * target_scale_row_stride + group,
        scale,
        mask=quant_valid,
    )

    metadata_row = program
    metadata_columns = tl.arange(0, group_size)
    metadata_mask = (metadata_row < max_tokens) & (metadata_columns < top_k)
    source_mask = metadata_mask & (metadata_row < num_tokens)
    route_id = tl.load(
        source_ids + metadata_row * source_id_row_stride + metadata_columns,
        mask=source_mask,
        other=-1,
    )
    route_weight = tl.load(
        source_weights + metadata_row * source_weight_row_stride + metadata_columns,
        mask=source_mask,
        other=0.0,
    )
    tl.store(
        target_ids + metadata_row * target_id_row_stride + metadata_columns,
        route_id.to(tl.int32),
        mask=metadata_mask,
    )
    tl.store(
        target_weights + metadata_row * target_weight_row_stride + metadata_columns,
        route_weight,
        mask=metadata_mask,
    )


def quantize_pack_input(
    target: SharedEpInputViews,
    *,
    source: torch.Tensor,
    source_ids: torch.Tensor,
    source_weights: torch.Tensor,
    group_size: int,
) -> None:
    """Quantize activations and routes directly into the shared input object."""

    if source.ndim != 2:
        raise ValueError("SharedEP activations must be two-dimensional")
    if source_ids.ndim != 2 or source_weights.ndim != 2:
        raise ValueError("SharedEP routes must be two-dimensional")
    num_tokens, hidden_size = source.shape
    max_tokens = target.activations.shape[0]
    if not 0 <= num_tokens <= max_tokens:
        raise ValueError(
            f"SharedEP input has {num_tokens} tokens, capacity is {max_tokens}"
        )
    if group_size != 128 or hidden_size % group_size != 0:
        raise ValueError(
            "SharedEP direct input quantization requires FP8 groups of 128"
        )
    if source_ids.shape != source_weights.shape:
        raise ValueError("SharedEP route id and weight shapes must match")
    if source_ids.shape != (num_tokens, target.topk_ids.shape[1]):
        raise ValueError("SharedEP route shape does not match storage")
    if hidden_size != target.activations.shape[1]:
        raise ValueError("SharedEP activation hidden size does not match storage")
    if target.scales.shape[1] != hidden_size // group_size:
        raise ValueError("SharedEP activation scale shape does not match storage")
    if source.dtype != torch.bfloat16:
        raise TypeError("SharedEP activations must use bfloat16")
    if not source.is_contiguous():
        raise ValueError("SharedEP activations must be contiguous")
    if source_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("SharedEP route ids must use int32 or int64")
    if source_weights.dtype != torch.float32:
        raise TypeError("SharedEP route weights must use float32")

    num_groups = hidden_size // group_size
    quant_programs = num_tokens * num_groups
    grid = max(quant_programs, max_tokens)
    _quantize_pack_input_kernel[(grid,)](
        source,
        source_ids,
        source_weights,
        target.activations,
        target.scales,
        target.topk_ids,
        target.topk_weights,
        source_q_row_stride=source.stride(0),
        source_id_row_stride=source_ids.stride(0),
        source_weight_row_stride=source_weights.stride(0),
        target_q_row_stride=target.activations.stride(0),
        target_scale_row_stride=target.scales.stride(0),
        target_id_row_stride=target.topk_ids.stride(0),
        target_weight_row_stride=target.topk_weights.stride(0),
        quant_programs=quant_programs,
        num_tokens=num_tokens,
        num_groups=num_groups,
        top_k=source_ids.shape[1],
        max_tokens=max_tokens,
        group_size=group_size,
        fp8_max=448.0,
        eps=1e-10,
        num_warps=4,
        num_stages=1,
    )


def prepare_routes(
    profile: SharedEpProfile,
    global_ids: torch.Tensor,
    global_weights: torch.Tensor,
    ready_signals: torch.Tensor,
    ready_epoch: torch.Tensor,
    *,
    local_expert_start: int,
) -> RoutePreparation:
    _validate_routes(profile, global_ids, global_weights)
    from sglang.kernels.ops.moe.shared_ep_route_prep import prepare_routes_cuda

    result = prepare_routes_cuda(
        global_ids,
        global_weights,
        ready_signals,
        ready_epoch,
        local_expert_start=local_expert_start,
        num_local_experts=profile.num_local_experts,
        block_size_m=profile.block_size_m,
        num_threads=profile.route_kernel_config["num_threads"],
    )
    return RoutePreparation(
        local_ids=result[0],
        local_weights=result[1],
        sorted_token_ids=result[2],
        expert_ids=result[3],
        num_tokens_post_padded=result[4],
    )


def _validate_routes(
    profile: SharedEpProfile,
    global_ids: torch.Tensor,
    global_weights: torch.Tensor,
) -> None:
    if (
        global_ids.ndim != 3
        or not 0 < global_ids.shape[0] <= profile.ep_size
        or not 0 < global_ids.shape[1] <= profile.max_tokens_per_rank
        or global_ids.shape[2] != profile.top_k
    ):
        raise ValueError(
            "SharedEP route ids require [owners<=EP, tokens<=capacity, Top-K] "
            f"within {(profile.ep_size, profile.max_tokens_per_rank, profile.top_k)}, "
            f"got {tuple(global_ids.shape)}"
        )
    if global_weights.shape != global_ids.shape:
        raise ValueError("SharedEP route id and weight shapes must match")
    if global_ids.dtype != torch.int32:
        raise TypeError("SharedEP route ids must use int32")
    if global_weights.dtype != torch.float32:
        raise TypeError("SharedEP route weights must use float32")
