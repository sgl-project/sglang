from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_MIN_TOP_K = 1024
_MAX_STEPS = 4
_MAX_OCCURRENCES = 8192
_GATHER_BLOCK_SIZE = 64


class HiSparseMTPSwapState(NamedTuple):
    """Persistent cache state and reusable miss workspace for MTP swap.

    ``cache_index`` stores the two int64 hash banks as
    ``[num_requests, 2, hash_size]``. ``cache_policy`` uses a control-plane row
    for the packed CLOCK states followed by one reference-epoch row per
    request: ``[1 + num_requests, hot_buffer_size]``.

    ``scratch_locs`` stores the physical locations assigned to unique misses.
    ``scratch_state`` likewise uses its first row for four contiguous counter
    banks and one metadata row per request. Its shape is
    ``[1 + num_requests, max(4 * num_requests, 5 * max_occurrences)]``.
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
) -> Module:
    template_args = make_cpp_args(
        block_size,
        num_top_k,
        hot_buffer_size,
        item_size_bytes,
        num_steps,
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
) -> None:
    """Resolve all speculative steps and swap unique misses in one launch pair."""
    if top_k_tokens.ndim != 3:
        raise ValueError("top_k_tokens must have shape [batch, steps, top_k].")

    batch_size, num_steps, num_top_k = top_k_tokens.shape
    hot_buffer_size = state.cache_policy.size(1)
    page_size = device_buffer_tokens.size(1) - hot_buffer_size
    total_occurrences = num_steps * num_top_k
    if not (
        num_top_k <= hot_buffer_size
        and 1 < num_steps <= _MAX_STEPS
        and num_top_k >= _MIN_TOP_K
        and total_occurrences <= _MAX_OCCURRENCES
    ):
        raise ValueError(
            "HiSparse MTP swap requires hot_buffer_size >= top_k >= 1024, "
            "2-4 steps, and at most 8192 total occurrences."
        )
    if page_size <= 0:
        raise ValueError("device_buffer_tokens must include an extra page.")
    if top_k_device_locs.shape != top_k_tokens.shape:
        raise ValueError("top_k_device_locs must match top_k_tokens.")
    if seq_lens.numel() < batch_size * num_steps:
        raise ValueError("seq_lens must provide one length per request and MTP step.")

    item_size_bytes = host_cache.stride(0) * host_cache.element_size()
    module = _jit_mtp_swap_module(
        item_size_bytes,
        _GATHER_BLOCK_SIZE,
        num_top_k,
        hot_buffer_size,
        num_steps,
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
    )
