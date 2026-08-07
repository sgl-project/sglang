"""Pull-cache shared-object consumer for SharedEP prefill."""

from __future__ import annotations

import msgspec
import torch
import triton
import triton.language as tl

from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
    invoke_fused_moe_kernel,
)


class PullCachePrefillPlan(msgspec.Struct, frozen=True, kw_only=True):
    source_rows: int
    source_route_capacity: int
    cache_rows: int
    hidden_size: int
    scale_groups: int
    top_k: int
    num_local_experts: int
    expert_alignment: int

    @property
    def activation_bytes(self) -> int:
        return self.cache_rows * self.hidden_size

    @property
    def scale_bytes(self) -> int:
        return self.cache_rows * self.scale_groups * 4

    @property
    def total_cache_bytes(self) -> int:
        return self.activation_bytes + self.scale_bytes


class PullCache(msgspec.Struct, kw_only=True):
    plan: PullCachePrefillPlan
    active_rows: int
    activations: torch.Tensor
    scales: torch.Tensor
    row_ids: torch.Tensor
    row_weights: torch.Tensor
    scale_backing: torch.Tensor


def make_pull_cache_prefill_plan(
    *,
    owners: int,
    source_tokens_per_owner: int,
    hidden_size: int,
    top_k: int,
    num_local_experts: int,
    expert_alignment: int,
) -> PullCachePrefillPlan:
    dimensions = {
        "owners": owners,
        "source_tokens_per_owner": source_tokens_per_owner,
        "hidden_size": hidden_size,
        "top_k": top_k,
        "num_local_experts": num_local_experts,
        "expert_alignment": expert_alignment,
    }
    for name, value in dimensions.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if hidden_size % 128 != 0:
        raise ValueError(f"hidden_size must be divisible by 128, got {hidden_size}")
    if expert_alignment & (expert_alignment - 1):
        raise ValueError(
            f"expert_alignment must be a power of two, got {expert_alignment}"
        )
    source_rows = owners * source_tokens_per_owner
    source_route_capacity = source_rows * top_k
    cache_rows = source_route_capacity + num_local_experts * (expert_alignment - 1)
    return PullCachePrefillPlan(
        source_rows=source_rows,
        source_route_capacity=source_route_capacity,
        cache_rows=cache_rows,
        hidden_size=hidden_size,
        scale_groups=hidden_size // 128,
        top_k=top_k,
        num_local_experts=num_local_experts,
        expert_alignment=expert_alignment,
    )


def allocate_pull_cache(
    plan: PullCachePrefillPlan,
    *,
    active_rows: int,
    device: torch.device,
) -> PullCache:
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"pull cache requires a CUDA device, got {device}")
    if not 0 < active_rows <= plan.cache_rows:
        raise ValueError(
            f"active_rows must be in [1, {plan.cache_rows}], got {active_rows}"
        )

    activations = torch.empty(
        (active_rows, plan.hidden_size),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    padded_rows = triton.cdiv(active_rows, 4) * 4
    scale_backing = torch.empty(
        (plan.scale_groups, padded_rows),
        dtype=torch.float32,
        device=device,
    )
    scales = scale_backing.t()[:active_rows]
    row_ids = torch.arange(active_rows, dtype=torch.int32, device=device)
    row_weights = torch.ones((active_rows, 1), dtype=torch.float32, device=device)
    return PullCache(
        plan=plan,
        active_rows=active_rows,
        activations=activations,
        scales=scales,
        row_ids=row_ids,
        row_weights=row_weights,
        scale_backing=scale_backing,
    )


@triton.jit
def _pull_cache_rows_kernel(
    source_activations,
    source_scales,
    sorted_token_ids,
    num_tokens_post_padded,
    cache_activations,
    cache_scales,
    source_activation_row_stride: tl.constexpr,
    source_activation_column_stride: tl.constexpr,
    source_scale_row_stride: tl.constexpr,
    source_scale_column_stride: tl.constexpr,
    cache_activation_row_stride: tl.constexpr,
    cache_activation_column_stride: tl.constexpr,
    cache_scale_row_stride: tl.constexpr,
    cache_scale_column_stride: tl.constexpr,
    hidden_size: tl.constexpr,
    scale_groups: tl.constexpr,
    activation_block: tl.constexpr,
    scale_block: tl.constexpr,
    top_k: tl.constexpr,
    source_route_capacity: tl.constexpr,
):
    row = tl.program_id(0)
    padded_rows = tl.load(num_tokens_post_padded)
    valid_row = row < padded_rows
    if valid_row:
        route_id = tl.load(sorted_token_ids + row)
        valid_route = (route_id >= 0) & (route_id < source_route_capacity)
        source_row = route_id // top_k

        for column_start in tl.static_range(0, hidden_size, activation_block):
            columns = column_start + tl.arange(0, activation_block)
            column_mask = columns < hidden_size
            values = tl.load(
                source_activations
                + source_row * source_activation_row_stride
                + columns * source_activation_column_stride,
                mask=valid_route & column_mask,
                other=0.0,
            )
            tl.store(
                cache_activations
                + row * cache_activation_row_stride
                + columns * cache_activation_column_stride,
                values,
                mask=column_mask,
            )

        scale_columns = tl.arange(0, scale_block)
        scale_mask = scale_columns < scale_groups
        scale_values = tl.load(
            source_scales
            + source_row * source_scale_row_stride
            + scale_columns * source_scale_column_stride,
            mask=valid_route & scale_mask,
            other=0.0,
        )
        tl.store(
            cache_scales
            + row * cache_scale_row_stride
            + scale_columns * cache_scale_column_stride,
            scale_values,
            mask=scale_mask,
        )


def pull_cache_rows(
    *,
    source_activations: torch.Tensor,
    source_scales: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    cache: PullCache,
    top_k: int,
    source_route_capacity: int,
) -> None:
    _validate_pull_inputs(
        source_activations=source_activations,
        source_scales=source_scales,
        sorted_token_ids=sorted_token_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        cache=cache,
        top_k=top_k,
        source_route_capacity=source_route_capacity,
    )
    _pull_cache_rows_kernel[(cache.active_rows,)](
        source_activations,
        source_scales,
        sorted_token_ids,
        num_tokens_post_padded,
        cache.activations,
        cache.scales,
        source_activation_row_stride=source_activations.stride(0),
        source_activation_column_stride=source_activations.stride(1),
        source_scale_row_stride=source_scales.stride(0),
        source_scale_column_stride=source_scales.stride(1),
        cache_activation_row_stride=cache.activations.stride(0),
        cache_activation_column_stride=cache.activations.stride(1),
        cache_scale_row_stride=cache.scales.stride(0),
        cache_scale_column_stride=cache.scales.stride(1),
        hidden_size=cache.plan.hidden_size,
        scale_groups=cache.plan.scale_groups,
        activation_block=256,
        scale_block=triton.next_power_of_2(cache.plan.scale_groups),
        top_k=top_k,
        source_route_capacity=source_route_capacity,
        num_warps=4,
        num_stages=2,
    )


def invoke_pull_cache_w13(
    *,
    cache: PullCache,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    config: dict[str, int],
    block_shape: tuple[int, int],
) -> None:
    if output.shape[0] != cache.active_rows:
        raise ValueError("W13 output rows must match the pull-cache capacity")
    invoke_fused_moe_kernel(
        A=cache.activations,
        B=weight,
        bias=None,
        C=output,
        A_scale=cache.scales,
        B_scale=weight_scale,
        B_zp=None,
        topk_weights=cache.row_weights,
        topk_ids=cache.row_ids.view(-1, 1),
        sorted_token_ids=cache.row_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=False,
        top_k=1,
        config=config,
        compute_type=tl.bfloat16,
        use_fp8_w8a8=True,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        block_shape=list(block_shape),
        filter_expert=True,
        c_sorted=True,
        a_is_prequantized=True,
    )


def _validate_pull_inputs(
    *,
    source_activations: torch.Tensor,
    source_scales: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    cache: PullCache,
    top_k: int,
    source_route_capacity: int,
) -> None:
    tensors = (
        source_activations,
        source_scales,
        sorted_token_ids,
        num_tokens_post_padded,
        cache.activations,
        cache.scales,
        cache.row_ids,
        cache.row_weights,
    )
    if any(tensor.device != cache.activations.device for tensor in tensors):
        raise ValueError("pull-cache tensors must be on the same CUDA device")
    if source_activations.dtype != torch.float8_e4m3fn:
        raise TypeError("source activations must use float8_e4m3fn")
    if source_activations.ndim != 2:
        raise ValueError("source activations must be two-dimensional")
    if source_activations.shape[1] != cache.plan.hidden_size:
        raise ValueError("source activation hidden size does not match the cache plan")
    if source_scales.dtype != torch.float32:
        raise TypeError("source scales must use float32")
    if source_scales.shape != (
        source_activations.shape[0],
        cache.plan.scale_groups,
    ):
        raise ValueError("source scale shape does not match source activations")
    for name, tensor in (
        ("sorted_token_ids", sorted_token_ids),
        ("num_tokens_post_padded", num_tokens_post_padded),
    ):
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must use int32")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if sorted_token_ids.ndim != 1 or sorted_token_ids.numel() < cache.active_rows:
        raise ValueError("sorted_token_ids must cover every cache row")
    if num_tokens_post_padded.numel() != 1:
        raise ValueError("num_tokens_post_padded must contain one device scalar")
    if top_k != cache.plan.top_k:
        raise ValueError(f"top_k must match cache plan value {cache.plan.top_k}")
    max_route_capacity = source_activations.shape[0] * top_k
    if not 0 < source_route_capacity <= max_route_capacity:
        raise ValueError(
            "source_route_capacity must be positive and cannot exceed source rows "
            f"times Top-K ({max_route_capacity})"
        )
    if cache.activations.shape != (
        cache.active_rows,
        cache.plan.hidden_size,
    ):
        raise ValueError("cache activation shape does not match the cache plan")
    if not cache.activations.is_contiguous() or cache.activations.stride(1) != 1:
        raise ValueError("cache activations must be contiguous")
    if cache.scales.shape != (cache.active_rows, cache.plan.scale_groups):
        raise ValueError("cache scale shape does not match the cache plan")
    if cache.scales.stride(0) != 1:
        raise ValueError("cache scales must use the TMA-aligned column-major layout")
    if (
        cache.row_ids.dtype != torch.int32
        or cache.row_ids.shape != (cache.active_rows,)
        or not cache.row_ids.is_contiguous()
    ):
        raise ValueError("cache row_ids must be contiguous int32 cache indices")
    if (
        cache.row_weights.dtype != torch.float32
        or cache.row_weights.shape != (cache.active_rows, 1)
        or not cache.row_weights.is_contiguous()
    ):
        raise ValueError("cache row_weights must be contiguous float32 column values")
