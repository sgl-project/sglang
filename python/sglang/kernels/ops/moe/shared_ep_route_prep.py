"""Shape-specialized CUDA route preparation for SharedEP."""

from __future__ import annotations

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

_INT32_MAX = 2**31 - 1


def _route_specialization_key(
    shape: tuple[int, ...],
    *,
    num_local_experts: int,
    block_size_m: int,
    num_threads: int,
) -> tuple[int, int, int, int, int, int]:
    """Validate and return every compile-time specialization dimension."""
    if len(shape) != 3 or any(
        type(dimension) is not int or dimension <= 0 for dimension in shape
    ):
        raise ValueError("SharedEP CUDA route ids require a non-empty 3D tensor")
    if type(num_local_experts) is not int or type(block_size_m) is not int:
        raise ValueError("SharedEP route specialization values must be integers")
    if type(num_threads) is not int:
        raise ValueError("SharedEP route specialization values must be integers")
    if not 0 < num_local_experts <= num_threads:
        raise ValueError("num_local_experts must be within the CUDA thread count")
    if block_size_m <= 0:
        raise ValueError("block_size_m must be positive")
    if num_threads <= 0 or num_threads > 1024 or num_threads % 32 != 0:
        raise ValueError("num_threads must be a positive warp multiple up to 1024")
    owners, max_tokens, top_k = shape
    route_capacity = owners * max_tokens * top_k
    max_sorted = route_capacity + num_local_experts * (block_size_m - 1)
    if route_capacity > _INT32_MAX or max_sorted > _INT32_MAX:
        raise ValueError("SharedEP route specialization exceeds int32 indexing")
    return (
        owners,
        max_tokens,
        top_k,
        num_local_experts,
        block_size_m,
        num_threads,
    )


@cache_once
def _jit_shared_ep_route_prep_module(
    owners: int,
    max_tokens: int,
    top_k: int,
    num_local_experts: int,
    block_size_m: int,
    num_threads: int,
):
    suffix = (
        f"o{owners}_m{max_tokens}_k{top_k}_e{num_local_experts}"
        f"_b{block_size_m}_t{num_threads}"
    )
    return load_jit(
        "shared_ep_route_prep",
        suffix,
        cuda_files=["moe/shared_ep_route_prep.cu"],
        cuda_wrappers=[
            ("shared_ep_route_prep", "SharedEpRoutePrepKernel::run"),
        ],
        extra_cuda_cflags=[
            f"-DSHARED_EP_OWNERS={owners}",
            f"-DSHARED_EP_MAX_TOKENS={max_tokens}",
            f"-DSHARED_EP_TOP_K={top_k}",
            f"-DSHARED_EP_LOCAL_EXPERTS={num_local_experts}",
            f"-DSHARED_EP_BLOCK_M={block_size_m}",
            f"-DSHARED_EP_THREADS={num_threads}",
        ],
    )


def prepare_routes_cuda(
    global_ids: torch.Tensor,
    global_weights: torch.Tensor,
    ready_signals: torch.Tensor,
    ready_epoch: torch.Tensor,
    *,
    local_expert_start: int,
    num_local_experts: int,
    block_size_m: int,
    num_threads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare local routes with a cached specialization of one CUDA template."""
    if type(local_expert_start) is not int or not (
        -(2**31) <= local_expert_start <= _INT32_MAX
    ):
        raise ValueError("local_expert_start must fit int32")
    specialization = _route_specialization_key(
        tuple(global_ids.shape),
        num_local_experts=num_local_experts,
        block_size_m=block_size_m,
        num_threads=num_threads,
    )
    if global_weights.shape != global_ids.shape:
        raise ValueError("SharedEP route id and weight shapes must match")
    if not global_ids.is_cuda or not global_weights.is_cuda:
        raise ValueError("SharedEP route ids and weights must be CUDA tensors")
    if not ready_signals.is_cuda or not ready_epoch.is_cuda:
        raise ValueError("SharedEP ready signals and epoch must be CUDA tensors")
    if global_weights.device != global_ids.device:
        raise ValueError("SharedEP route ids and weights must use the same device")
    if (
        ready_signals.device != global_ids.device
        or ready_epoch.device != global_ids.device
    ):
        raise ValueError("SharedEP route data and readiness must use the same device")
    if global_ids.dtype != torch.int32:
        raise TypeError("SharedEP route ids must use int32")
    if global_weights.dtype != torch.float32:
        raise TypeError("SharedEP route weights must use float32")
    if ready_signals.dtype != torch.uint8:
        raise TypeError("SharedEP ready signals must use uint8 storage")
    if ready_epoch.dtype != torch.int32 or ready_epoch.numel() != 1:
        raise TypeError("SharedEP ready epoch must be one int32 value")
    if ready_signals.numel() < global_ids.shape[0] * 4:
        raise ValueError("SharedEP ready signal storage is too small")
    if global_ids.stride(2) != 1 or global_weights.stride(2) != 1:
        raise ValueError("SharedEP route columns must be contiguous")

    local_ids = torch.empty_like(
        global_ids,
        memory_format=torch.contiguous_format,
    )
    local_weights = torch.empty_like(
        global_weights,
        memory_format=torch.contiguous_format,
    )
    route_capacity = global_ids.numel()
    max_sorted = route_capacity + num_local_experts * (block_size_m - 1)
    sorted_token_ids = torch.empty(
        max_sorted,
        dtype=torch.int32,
        device=global_ids.device,
    )
    expert_ids = torch.empty(
        (max_sorted + block_size_m - 1) // block_size_m,
        dtype=torch.int32,
        device=global_ids.device,
    )
    num_tokens_post_padded = torch.empty(
        1,
        dtype=torch.int32,
        device=global_ids.device,
    )
    _jit_shared_ep_route_prep_module(*specialization).shared_ep_route_prep(
        global_ids,
        global_weights,
        ready_signals,
        ready_epoch,
        local_ids,
        local_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        local_expert_start,
    )
    return (
        local_ids,
        local_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    )
