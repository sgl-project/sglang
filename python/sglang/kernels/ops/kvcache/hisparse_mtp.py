from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_GATHER_BLOCK_SIZE = 64


class HiSparseMTPSwapState(NamedTuple):
    """Persistent cache state and reusable miss workspace for MTP swap.

    ``cache_index`` stores the two int64 hash banks as
    ``[num_requests, 2, hash_size]``. ``cache_policy`` uses a control-plane row
    for the packed CLOCK states followed by one reference-epoch row per
    request: ``[1 + num_requests, hot_buffer_size]``.

    ``scratch_locs`` and ``scratch_state`` hold reusable miss locations,
    counters, and metadata shared by all layers.
    """

    cache_index: torch.Tensor
    cache_policy: torch.Tensor
    scratch_locs: torch.Tensor
    scratch_state: torch.Tensor


@cache_once
def _jit_mtp_swap_module(
    item_size_bytes: int,
    block_size: int,
    num_top_k: int,
    hot_buffer_size: int,
    num_steps: int,
    record_miss_plan: bool,
) -> Module:
    template_args = make_cpp_args(
        block_size,
        num_top_k,
        hot_buffer_size,
        item_size_bytes,
        num_steps,
        record_miss_plan,
        is_arch_support_pdl(),
    )
    return load_jit(
        "hisparse_mtp_swap",
        *template_args,
        cuda_files=["hisparse_mtp_swap.cuh"],
        cuda_wrappers=[
            (
                "load_cache_to_device_buffer_mtp",
                f"load_cache_to_device_buffer_mtp<{template_args}>",
            )
        ],
    )


def load_cache_to_device_buffer_mtp_mla(
    *,
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    state: HiSparseMTPSwapState,
    num_real_reqs: torch.Tensor,
    miss_src: torch.Tensor | None = None,
    miss_dst: torch.Tensor | None = None,
    miss_count: torch.Tensor | None = None,
) -> None:
    """Resolve all speculative steps and swap unique misses in one launch pair.

    Optional miss-plan outputs use the same protocol as the non-MTP HiSparse
    kernel, so shared-index layers can replay only the Host-to-GPU copies with
    ``copy_cache_planned_mla``.
    """
    _, num_steps, num_top_k = top_k_tokens.shape
    if not 2 <= num_steps <= 4:
        raise ValueError(f"HiSparse MTP swap requires 2-4 steps, got {num_steps}.")
    hot_buffer_size = state.cache_policy.size(1)
    page_size = device_buffer_tokens.size(1) - hot_buffer_size
    item_size_bytes = host_cache.stride(0) * host_cache.element_size()
    record_miss_plan = miss_src is not None
    if record_miss_plan:
        if miss_dst is None or miss_count is None:
            raise ValueError(
                "miss_src, miss_dst, and miss_count must be provided together."
            )
        if miss_src.dtype != torch.int64 or miss_dst.dtype != torch.int32:
            raise ValueError("miss_src must be int64 and miss_dst must be int32.")
        if miss_count.dtype != torch.int32:
            raise ValueError("miss_count must be int32.")
        plan_capacity = num_steps * num_top_k
        batch_size = top_k_tokens.size(0)
        if (
            miss_src.ndim != 2
            or miss_dst.ndim != 2
            or miss_src.size(0) < batch_size
            or miss_dst.size(0) < batch_size
            or miss_src.size(1) < plan_capacity
            or miss_dst.size(1) < plan_capacity
        ):
            raise ValueError(
                "MTP miss_src/miss_dst must have shape "
                f"[batch, >= steps * top_k] (capacity {plan_capacity})."
            )
        if miss_count.ndim != 1 or miss_count.numel() < batch_size:
            raise ValueError("MTP miss_count must have shape [batch].")
        if miss_src.stride(0) != miss_dst.stride(0):
            raise ValueError("miss_src/miss_dst row strides must match.")
    else:
        if miss_dst is not None or miss_count is not None:
            raise ValueError(
                "miss_src, miss_dst, and miss_count must be provided together."
            )
        empty = torch.empty(0)
        miss_src = miss_dst = miss_count = empty

    module = _jit_mtp_swap_module(
        item_size_bytes,
        _GATHER_BLOCK_SIZE,
        num_top_k,
        hot_buffer_size,
        num_steps,
        record_miss_plan,
    )

    module.load_cache_to_device_buffer_mtp(
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        host_cache,
        device_buffer,
        top_k_device_locs,
        req_pool_indices,
        seq_lens,
        state.cache_index,
        state.cache_policy,
        state.scratch_locs,
        state.scratch_state,
        num_real_reqs,
        page_size,
        miss_src,
        miss_dst,
        miss_count,
    )
