"""Exact-LRU kernel for independent logical shards."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@functools.cache
def _jit_sharded_module(
    block_size: int,
    num_top_k: int,
    hot_buffer_size: int,
    num_ctas: int,
    min_blocks_per_sm: int,
) -> Module:
    metadata_template_args = make_cpp_args(
        block_size,
        num_top_k,
        hot_buffer_size,
        num_ctas,
        min_blocks_per_sm,
    )
    return load_jit(
        "hisparse_sharded",
        block_size,
        num_top_k,
        hot_buffer_size,
        num_ctas,
        min_blocks_per_sm,
        cuda_files=["hisparse_sharded.cuh"],
        cuda_wrappers=[
            (
                "load_cache_to_device_buffer_mla_sharded",
                f"load_cache_to_device_buffer_mla_sharded<{metadata_template_args}>",
            ),
        ],
    )


def logical_shards_for_hot_buffer(
    hot_buffer_size: int, device: torch.device | str | int
) -> int:
    warp_size = torch.cuda.get_device_properties(device).warp_size
    if hot_buffer_size % warp_size:
        raise ValueError("hot_buffer_size must be divisible by the device warp size")
    return hot_buffer_size // warp_size


def load_cache_to_device_buffer_mla_sharded(
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
    lru_slots: torch.Tensor,
    num_real_reqs: torch.Tensor,
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
    num_ctas: int,
    block_size: int,
    min_blocks_per_sm: int = 1,
) -> None:
    logical_shards = logical_shards_for_hot_buffer(hot_buffer_size, top_k_tokens.device)
    if num_ctas <= 0 or num_ctas > logical_shards:
        raise ValueError("num_ctas must not exceed logical_shards")
    if num_ctas & (num_ctas - 1) or logical_shards % num_ctas:
        raise ValueError("num_ctas must be a power-of-two divisor of shards")
    if block_size not in (32, 64, 128, 256, 512, 768, 1024):
        raise ValueError("block_size must be one of 32, 64, 128, 256, 512, 768, 1024")
    if min_blocks_per_sm not in (1, 2, 3, 4, 5, 6, 8):
        raise ValueError("unsupported launch-bounds min_blocks_per_sm")
    if req_pool_indices.dtype != torch.int64 or seq_lens.dtype != torch.int32:
        raise ValueError("req_pool_indices must be int64 and seq_lens int32")
    if num_real_reqs.dtype != torch.int32 or num_real_reqs.numel() != 1:
        raise ValueError("num_real_reqs must be a single-element int32 tensor")
    if lru_slots.dtype != torch.uint8:
        raise ValueError("sharded exact LRU uses uint8 local way indices")
    contiguous = (
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        host_cache,
        device_buffer,
        top_k_device_locs,
        req_pool_indices,
        seq_lens,
        lru_slots,
        num_real_reqs,
    )
    if not all(tensor.is_contiguous() for tensor in contiguous):
        raise ValueError("sharded kernel requires contiguous tensors")

    module = _jit_sharded_module(
        block_size,
        num_top_k,
        hot_buffer_size,
        num_ctas,
        min_blocks_per_sm,
    )
    module.load_cache_to_device_buffer_mla_sharded(
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        host_cache,
        device_buffer,
        top_k_device_locs,
        req_pool_indices,
        seq_lens,
        lru_slots,
        num_real_reqs,
        item_size_bytes,
    )
