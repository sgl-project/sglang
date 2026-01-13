from typing import List

import torch


def is_hip() -> bool:
    return torch.version.hip is not None


_is_hip = is_hip()


def transfer_kv_per_layer(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_per_layer.default(
        src_k,
        dst_k,
        src_v,
        dst_v,
        src_indices,
        dst_indices,
        item_size,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_per_layer_pf_lf(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    item_size: int,
    src_layout_dim: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_per_layer_pf_lf.default(
        src_k,
        dst_k,
        src_v,
        dst_v,
        src_indices,
        dst_indices,
        layer_id,
        item_size,
        src_layout_dim,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_per_layer_ph_lf(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    item_size: int,
    src_layout_dim: int,
    page_size: int,
    head_num: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_per_layer_ph_lf.default(
        src_k,
        dst_k,
        src_v,
        dst_v,
        src_indices,
        dst_indices,
        layer_id,
        item_size,
        src_layout_dim,
        page_size,
        head_num,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_all_layer(
    src_k_layers: torch.Tensor,
    dst_k_layers: torch.Tensor,
    src_v_layers: torch.Tensor,
    dst_v_layers: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    num_layers: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_all_layer.default(
        src_k_layers,
        dst_k_layers,
        src_v_layers,
        dst_v_layers,
        src_indices,
        dst_indices,
        item_size,
        num_layers,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_all_layer_lf_pf(
    src_k_layers: torch.Tensor,
    dst_k: torch.Tensor,
    src_v_layers: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    dst_layout_dim: int,
    num_layers: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_all_layer_lf_pf.default(
        src_k_layers,
        dst_k,
        src_v_layers,
        dst_v,
        src_indices,
        dst_indices,
        item_size,
        dst_layout_dim,
        num_layers,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_all_layer_lf_ph(
    src_k_layers: torch.Tensor,
    dst_k: torch.Tensor,
    src_v_layers: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    dst_layout_dim: int,
    num_layers: int,
    page_size: int,
    head_num: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_all_layer_lf_ph.default(
        src_k_layers,
        dst_k,
        src_v_layers,
        dst_v,
        src_indices,
        dst_indices,
        item_size,
        dst_layout_dim,
        num_layers,
        page_size,
        head_num,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_direct(
    src_layers: List[torch.Tensor],
    dst_layers: List[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    page_size: int,
):
    torch.ops.sgl_kernel.transfer_kv_direct.default(
        src_layers, dst_layers, src_indices, dst_indices, page_size
    )


def transfer_kv_per_layer_direct_pf_lf(
    src_ptrs: List[torch.Tensor],
    dst_ptrs: List[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    page_size: int,
):
    torch.ops.sgl_kernel.transfer_kv_per_layer_direct_pf_lf.default(
        src_ptrs, dst_ptrs, src_indices, dst_indices, layer_id, page_size
    )


def transfer_kv_all_layer_direct_lf_pf(
    src_ptrs: List[torch.Tensor],
    dst_ptrs: List[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    page_size: int,
):
    torch.ops.sgl_kernel.transfer_kv_all_layer_direct_lf_pf.default(
        src_ptrs, dst_ptrs, src_indices, dst_indices, page_size
    )


def transfer_kv_per_layer_mla(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_per_layer_mla.default(
        src,
        dst,
        src_indices,
        dst_indices,
        item_size,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_per_layer_mla_pf_lf(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    item_size: int,
    src_layout_dim: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_per_layer_mla_pf_lf.default(
        src,
        dst,
        src_indices,
        dst_indices,
        layer_id,
        item_size,
        src_layout_dim,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_all_layer_mla(
    src_layers: torch.Tensor,
    dst_layers: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    num_layers: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_all_layer_mla.default(
        src_layers,
        dst_layers,
        src_indices,
        dst_indices,
        item_size,
        num_layers,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_all_layer_mla_lf_pf(
    src_layers: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    dst_layout_dim: int,
    num_layers: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_all_layer_mla_lf_pf.default(
        src_layers,
        dst,
        src_indices,
        dst_indices,
        item_size,
        dst_layout_dim,
        num_layers,
        block_quota,
        num_warps_per_block,
    )


def sparse_cache_manager_fused(
    top_k_indices: torch.Tensor,
    hot_buffer_token_indices: torch.Tensor,
    hot_buffer_device_locations: torch.Tensor,
    cache_cpu_locations: torch.Tensor,
    top_k_device_locations: torch.Tensor,
    copy_src_cpu_locations: torch.Tensor,
    copy_dst_gpu_locations: torch.Tensor,
    copy_count: torch.Tensor,
):
    """
    Fused sparse cache manager kernel for small top_k sizes.

    This kernel identifies cache hits and misses, assigns eviction slots to misses,
    and prepares copy operations for CPU→GPU transfers.

    Args:
        top_k_indices: [K] Token indices that need to be in GPU for this iteration
        hot_buffer_token_indices: [H] Token indices currently in GPU hot buffer (modified in-place)
        hot_buffer_device_locations: [H] GPU memory locations for hot buffer tokens
        cache_cpu_locations: [N] CPU memory location for each token
        top_k_device_locations: [K] Output: GPU locations for top_k tokens
        copy_src_cpu_locations: [M] Output: CPU locations to copy from
        copy_dst_gpu_locations: [M] Output: GPU locations to copy to
        copy_count: [1] Output: Number of copies needed

    Note:
        - Supports top_k <= 256 and hot_buffer <= 1024
        - All tensors must be int64 type on CUDA
        - hot_buffer_token_indices is modified in-place to reflect evictions
    """
    torch.ops.sgl_kernel.sparse_cache_manager_fused.default(
        top_k_indices,
        hot_buffer_token_indices,
        hot_buffer_device_locations,
        cache_cpu_locations,
        top_k_device_locations,
        copy_src_cpu_locations,
        copy_dst_gpu_locations,
        copy_count,
    )


def sparse_cache_manager(
    top_k_indices: torch.Tensor,
    hot_buffer_token_indices: torch.Tensor,
    hot_buffer_device_locations: torch.Tensor,
    cache_cpu_locations: torch.Tensor,
    top_k_device_locations: torch.Tensor,
    copy_src_cpu_locations: torch.Tensor,
    copy_dst_gpu_locations: torch.Tensor,
    copy_count: torch.Tensor,
):
    """
    Multi-phase sparse cache manager kernel for larger sizes.

    Similar to sparse_cache_manager_fused but uses multiple kernel launches
    for better scalability with larger top_k and hot_buffer sizes.

    Args:
        Same as sparse_cache_manager_fused

    Note:
        - All tensors must be int64 type on CUDA
        - hot_buffer_token_indices is modified in-place to reflect evictions
    """
    torch.ops.sgl_kernel.sparse_cache_manager.default(
        top_k_indices,
        hot_buffer_token_indices,
        hot_buffer_device_locations,
        cache_cpu_locations,
        top_k_device_locations,
        copy_src_cpu_locations,
        copy_dst_gpu_locations,
        copy_count,
    )


def sparse_cache_copy(
    cpu_cache: torch.Tensor,
    gpu_cache: torch.Tensor,
    copy_src_cpu_locations: torch.Tensor,
    copy_dst_gpu_locations: torch.Tensor,
    copy_count: int,
    item_size_bytes: int,
):
    """
    Perform CPU→GPU copies for sparse cache misses.

    Each warp handles one copy operation for efficient parallel data transfer.

    Args:
        cpu_cache: CPU pinned memory buffer containing KV cache
        gpu_cache: GPU memory buffer for KV cache
        copy_src_cpu_locations: [M] CPU locations to copy from
        copy_dst_gpu_locations: [M] GPU locations to copy to
        copy_count: Number of copies to perform
        item_size_bytes: Size of each cache item in bytes (must be divisible by 8)
    """
    torch.ops.sgl_kernel.sparse_cache_copy.default(
        cpu_cache,
        gpu_cache,
        copy_src_cpu_locations,
        copy_dst_gpu_locations,
        copy_count,
        item_size_bytes,
    )


def sparse_cache_manager_batch(
    top_k_indices: torch.Tensor,
    hot_buffer_token_indices: torch.Tensor,
    hot_buffer_device_locations: torch.Tensor,
    cache_cpu_locations: torch.Tensor,
    top_k_device_locations: torch.Tensor,
    copy_src_cpu_locations: torch.Tensor,
    copy_dst_gpu_locations: torch.Tensor,
    copy_counts: torch.Tensor,
    top_k_sizes: torch.Tensor,
    hot_buffer_sizes: torch.Tensor,
    token_pool_sizes: torch.Tensor,
    batch_size: int,
    max_top_k: int,
    max_hot_buffer: int,
    max_copies: int,
):
    """
    Batched sparse cache manager for multiple requests.

    Processes multiple requests in parallel, where each CUDA block handles
    one request independently.

    Args:
        top_k_indices: [B, max_K] Token indices for each request
        hot_buffer_token_indices: [B, max_H] Token indices in GPU hot buffer per request
        hot_buffer_device_locations: [B, max_H] GPU locations per request
        cache_cpu_locations: [sum(N_i)] Flattened CPU locations for all tokens
        top_k_device_locations: [B, max_K] Output: GPU locations per request
        copy_src_cpu_locations: [B, max_M] Output: CPU locations to copy
        copy_dst_gpu_locations: [B, max_M] Output: GPU locations to copy
        copy_counts: [B] Output: Number of copies per request
        top_k_sizes: [B] Actual top_k size per request
        hot_buffer_sizes: [B] Actual hot buffer size per request
        token_pool_sizes: [B] Total tokens per request
        batch_size: Number of requests
        max_top_k: Maximum top_k size (padded dimension)
        max_hot_buffer: Maximum hot buffer size (padded dimension)
        max_copies: Maximum copies per request (padded dimension)
    """
    torch.ops.sgl_kernel.sparse_cache_manager_batch.default(
        top_k_indices,
        hot_buffer_token_indices,
        hot_buffer_device_locations,
        cache_cpu_locations,
        top_k_device_locations,
        copy_src_cpu_locations,
        copy_dst_gpu_locations,
        copy_counts,
        top_k_sizes,
        hot_buffer_sizes,
        token_pool_sizes,
        batch_size,
        max_top_k,
        max_hot_buffer,
        max_copies,
    )
