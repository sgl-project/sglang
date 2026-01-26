from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@functools.cache
def _jit_sparse_module(
    block_size: int,
    num_top_k: int,
    hot_buffer_size: int,
    is_mla: bool = False,
) -> Module:
    args = make_cpp_args(block_size, num_top_k, hot_buffer_size, is_mla)
    return load_jit(
        "sparse_cache",
        *args,
        cuda_files=["sparse.cuh"],
        cuda_wrappers=[
            ("load_cache_to_device_buffer", f"load_cache_to_device_buffer<{args}>")
        ],
    )


def load_cache_to_device_buffer_mla(
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    page_table: torch.Tensor,
    diff_map: torch.Tensor,
    req_pool_indices: torch.Tensor,
    sparse_mask: torch.Tensor,
    seq_lens: torch.Tensor,
    page_size: int,
    layer_id: int,
    item_size_bytes: int,
    *,
    block_size: int = 256,
    num_top_k: int = 0,
    hot_buffer_size: int = 0,
) -> None:
    """
    Load cache from host to device buffer for MLA architecture (K cache only).

    Args:
        top_k_tokens: Top-k token indices, shape (bs, num_top_k)
        device_buffer_tokens: Token indices in device buffer
        host_cache_locs: Cache locations in host memory
        device_buffer_locs: Cache locations in device buffer
        host_cache: Host K cache data
        device_buffer: Device K buffer data
        top_k_device_locs: Output device locations for top-k tokens, shape (bs, num_top_k)
        page_table: Page table for memory management
        req_pool_indices: Request pool indices
        sparse_mask: Sparse mask to enable/disable sparse attention per request
        seq_lens: Sequence lengths for each request
        page_size: Page size for memory management
        layer_id: Current layer ID
        item_size_bytes: Size of each cache item in bytes
        block_size: CUDA block size (default: 256)
        num_top_k: Number of top-k tokens (default: inferred from top_k_tokens)
        hot_buffer_size: Size of hot buffer (default: inferred from device_buffer_tokens)
    """
    # Infer parameters if not provided
    if num_top_k <= 0:
        num_top_k = top_k_tokens.size(-1)
    if hot_buffer_size <= 0:
        hot_buffer_size = device_buffer_tokens.size(-1)

    # Validate that HOT_BUFFER_SIZE >= NUM_TOP_K
    assert (
        hot_buffer_size >= num_top_k
    ), f"hot_buffer_size ({hot_buffer_size}) must be >= num_top_k ({num_top_k})"

    module = _jit_sparse_module(block_size, num_top_k, hot_buffer_size, is_mla=True)
    
    # Create empty tensors for V cache (not used in MLA)
    empty = torch.empty(0)
    
    module.load_cache_to_device_buffer(
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        host_cache,
        empty,
        device_buffer,
        empty,
        top_k_device_locs,
        page_table,
        diff_map,
        req_pool_indices,
        sparse_mask,
        seq_lens,
        page_size,
        layer_id,
        item_size_bytes,
    )


def load_cache_to_device_buffer(
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache_k: torch.Tensor,
    host_cache_v: torch.Tensor,
    device_buffer_k: torch.Tensor,
    device_buffer_v: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    page_table: torch.Tensor,
    diff_map: torch.Tensor,
    req_pool_indices: torch.Tensor,
    sparse_mask: torch.Tensor,
    seq_lens: torch.Tensor,
    page_size: int,
    layer_id: int,
    item_size_bytes: int,
    *,
    block_size: int = 256,
    num_top_k: int = 0,
    hot_buffer_size: int = 0,
) -> None:
    """
    Load cache from host to device buffer using sparse attention pattern (with K/V caches).

    Args:
        top_k_tokens: Top-k token indices, shape (bs, num_top_k)
        device_buffer_tokens: Token indices in device buffer
        host_cache_locs: Cache locations in host memory
        device_buffer_locs: Cache locations in device buffer
        host_cache_k: Host K cache data
        host_cache_v: Host V cache data
        device_buffer_k: Device K buffer data
        device_buffer_v: Device V buffer data
        top_k_device_locs: Output device locations for top-k tokens, shape (bs, num_top_k)
        page_table: Page table for memory management
        req_pool_indices: Request pool indices
        sparse_mask: Sparse mask to enable/disable sparse attention per request
        seq_lens: Sequence lengths for each request
        page_size: Page size for memory management
        layer_id: Current layer ID
        item_size_bytes: Size of each cache item in bytes
        block_size: CUDA block size (default: 256)
        num_top_k: Number of top-k tokens (default: inferred from top_k_tokens)
        hot_buffer_size: Size of hot buffer (default: inferred from device_buffer_tokens)
    """
    # Infer parameters if not provided
    if num_top_k <= 0:
        num_top_k = top_k_tokens.size(-1)
    if hot_buffer_size <= 0:
        hot_buffer_size = device_buffer_tokens.size(-1)

    # Validate that HOT_BUFFER_SIZE >= NUM_TOP_K
    assert (
        hot_buffer_size >= num_top_k
    ), f"hot_buffer_size ({hot_buffer_size}) must be >= num_top_k ({num_top_k})"

    module = _jit_sparse_module(block_size, num_top_k, hot_buffer_size, is_mla=False)
    
    module.load_cache_to_device_buffer(
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        host_cache_k,
        host_cache_v,
        device_buffer_k,
        device_buffer_v,
        top_k_device_locs,
        page_table,
        diff_map,
        req_pool_indices,
        sparse_mask,
        seq_lens,
        page_size,
        layer_id,
        item_size_bytes,
    )

