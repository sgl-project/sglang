from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.kernels.jit.utils import (
    cache_once,
    is_hip_runtime,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    import torch
    from tvm_ffi.module import Module

_is_hip = is_hip_runtime()

# ROCm needs a wider grid to saturate mapped-host transfers; CUDA keeps the legacy quota.
DEFAULT_BLOCK_QUOTA = 32 if _is_hip else 2


@cache_once
def _jit_hicache_module(*, element_size: int, unroll: int, block_quota: int) -> Module:
    args = make_cpp_args(
        element_size,
        unroll,
        block_quota,
        1024,  # num_threads, can be tuned for performance
    )
    return load_jit(
        "hicache",
        *args,
        cuda_files=[
            "kvcacheio/hicache.cuh",
        ],
        cuda_wrappers=[
            ("launch_one", f"&HiCacheKernel<{args}>::run_one"),
            ("launch_all", f"&HiCacheKernel<{args}>::run_all"),
            ("launch_one_mla", f"&HiCacheKernel<{args}>::run_one_mla"),
            ("launch_all_mla", f"&HiCacheKernel<{args}>::run_all_mla"),
        ],
    )


@cache_once
def _jit_hicache_staged_module(
    *, element_size: int, unroll: int, block_quota: int
) -> Module:
    args = make_cpp_args(
        element_size,
        unroll,
        block_quota,
        1024,  # num_threads, kept for template compatibility
    )
    return load_jit(
        "hicache_staged",
        *args,
        cuda_files=[
            "kvcacheio/staged_write_back.cuh",
        ],
        cuda_wrappers=[
            (
                "launch_all_lf_pf_staged",
                f"&HiCacheStagedWriteBackKernel<{args}>::run_all_lf_pf_staged",
            ),
            (
                "launch_all_mla_lf_pf_staged",
                f"&HiCacheStagedWriteBackKernel<{args}>::run_all_mla_lf_pf_staged",
            ),
        ],
    )


def can_use_hicache_jit_kernel(
    *,
    element_size: int,
    unroll: int | None = None,  # can be tuned for performance
    block_quota: int | None = None,  # can be tuned for less interference
) -> bool:
    logger = logging.getLogger(__name__)
    if element_size % 128 != 0:
        logger.warning(f"Unsupported {element_size = } for JIT HiCache kernel")
        return False
    try:
        unroll = unroll or _default_unroll(element_size)
        block_quota = block_quota or DEFAULT_BLOCK_QUOTA
        _jit_hicache_module(
            element_size=element_size,
            unroll=unroll,
            block_quota=block_quota,
        )
        return True
    except Exception as e:
        logger.warning(f"Failed to load JIT HiCache kernel: {e}")
        return False


def can_use_write_back_jit_kernel(
    *,
    element_size: int,
    unroll: int | None = None,  # can be tuned for performance
    block_quota: int | None = None,  # can be tuned for less interference
) -> bool:
    logger = logging.getLogger(__name__)
    if element_size % 16 != 0:
        logger.warning(f"Unsupported {element_size = } for staged JIT HiCache kernel")
        return False
    try:
        unroll = unroll or _default_unroll(element_size)
        block_quota = block_quota or DEFAULT_BLOCK_QUOTA
        _jit_hicache_staged_module(
            element_size=element_size,
            unroll=unroll,
            block_quota=block_quota,
        )
        return True
    except Exception as e:
        logger.warning(f"Failed to load staged JIT HiCache kernel: {e}")
        return False


def _default_unroll(element_size: int) -> int:
    if element_size <= 512:
        return 4

    if element_size <= 1024:
        return 2

    # fallback: no unroll
    return 1


@debug_kernel_api
def transfer_hicache_one_layer(
    k_cache_dst: torch.Tensor,
    v_cache_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    k_cache_src: torch.Tensor,
    v_cache_src: torch.Tensor,
    indices_src: torch.Tensor,
    *,
    element_dim: int | None = None,
    unroll: int | None = None,  # can be tuned for performance
    block_quota: int | None = None,  # can be tuned for less interference
) -> None:
    element_dim = element_dim or k_cache_dst.size(-1)
    k_cache_src = k_cache_src.view(-1, element_dim)
    v_cache_src = v_cache_src.view(-1, element_dim)
    k_cache_dst = k_cache_dst.view(-1, element_dim)
    v_cache_dst = v_cache_dst.view(-1, element_dim)
    element_size = element_dim * k_cache_dst.element_size()
    block_quota = block_quota or DEFAULT_BLOCK_QUOTA
    unroll = unroll or _default_unroll(element_size)
    module = _jit_hicache_module(
        element_size=element_size,
        unroll=unroll,
        block_quota=block_quota,
    )
    module.launch_one(
        k_cache_dst,
        v_cache_dst,
        indices_dst,
        k_cache_src,
        v_cache_src,
        indices_src,
    )


@debug_kernel_api
def transfer_hicache_all_layer(
    k_ptr_dst: torch.Tensor,
    v_ptr_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    k_ptr_src: torch.Tensor,
    v_ptr_src: torch.Tensor,
    indices_src: torch.Tensor,
    *,
    kv_cache_src_stride_bytes: int,
    kv_cache_dst_stride_bytes: int,
    element_size: int | None = None,
    unroll: int | None = None,  # can be tuned for performance
    block_quota: int | None = None,  # can be tuned for less interference
) -> None:
    if element_size is None:  # assume both contiguous
        assert kv_cache_dst_stride_bytes == kv_cache_src_stride_bytes
        element_size = kv_cache_dst_stride_bytes

    block_quota = block_quota or DEFAULT_BLOCK_QUOTA
    unroll = unroll or _default_unroll(element_size)
    module = _jit_hicache_module(
        element_size=element_size,
        unroll=unroll,
        block_quota=block_quota,
    )
    module.launch_all(
        k_ptr_dst,
        v_ptr_dst,
        indices_dst,
        k_ptr_src,
        v_ptr_src,
        indices_src,
        kv_cache_src_stride_bytes,
        kv_cache_dst_stride_bytes,
    )


def transfer_hicache_one_layer_mla(
    cache_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    cache_src: torch.Tensor,
    indices_src: torch.Tensor,
    *,
    element_dim: int | None = None,
    unroll: int | None = None,
    block_quota: int | None = None,
) -> None:
    element_dim = element_dim or cache_dst.size(-1)
    cache_src = cache_src.view(-1, element_dim)
    cache_dst = cache_dst.view(-1, element_dim)
    element_size = element_dim * cache_dst.element_size()
    block_quota = block_quota or DEFAULT_BLOCK_QUOTA
    unroll = unroll or _default_unroll(element_size)
    module = _jit_hicache_module(
        element_size=element_size,
        unroll=unroll,
        block_quota=block_quota,
    )
    module.launch_one_mla(
        cache_dst,
        indices_dst,
        cache_src,
        indices_src,
    )


def transfer_hicache_all_layer_mla(
    ptr_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    ptr_src: torch.Tensor,
    indices_src: torch.Tensor,
    *,
    cache_src_stride_bytes: int,
    cache_dst_stride_bytes: int,
    element_size: int | None = None,
    unroll: int | None = None,
    block_quota: int | None = None,
) -> None:
    if element_size is None:
        assert cache_dst_stride_bytes == cache_src_stride_bytes
        element_size = cache_dst_stride_bytes

    block_quota = block_quota or DEFAULT_BLOCK_QUOTA
    unroll = unroll or _default_unroll(element_size)
    module = _jit_hicache_module(
        element_size=element_size,
        unroll=unroll,
        block_quota=block_quota,
    )
    module.launch_all_mla(
        ptr_dst,
        indices_dst,
        ptr_src,
        indices_src,
        cache_src_stride_bytes,
        cache_dst_stride_bytes,
    )


@debug_kernel_api
def transfer_hicache_all_layer_staged_lf_pf(
    k_ptr_src: torch.Tensor,
    v_ptr_src: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    staging_k: torch.Tensor,
    staging_v: torch.Tensor,
    dst_k: torch.Tensor,
    dst_v: torch.Tensor,
    *,
    page_size: int,
    element_size: int | None = None,
    unroll: int | None = None,
    block_quota: int | None = None,
) -> None:
    element_dim = staging_k[0, 0].numel()
    element_size = element_size or (element_dim * staging_k.element_size())
    block_quota = block_quota or DEFAULT_BLOCK_QUOTA
    unroll = unroll or _default_unroll(element_size)
    src_page_indices = src_indices[::page_size].contiguous()
    module = _jit_hicache_staged_module(
        element_size=element_size,
        unroll=unroll,
        block_quota=block_quota,
    )
    staging_page_capacity = staging_k.shape[0] // page_size
    staging_k = staging_k.view(staging_k.shape[0], staging_k.shape[1], -1)
    staging_v = staging_v.view(staging_v.shape[0], staging_v.shape[1], -1)
    dst_k = dst_k.view(dst_k.shape[0], dst_k.shape[1], -1)
    dst_v = dst_v.view(dst_v.shape[0], dst_v.shape[1], -1)
    for page_begin in range(0, src_page_indices.numel(), staging_page_capacity):
        chunk_pages = min(staging_page_capacity, src_page_indices.numel() - page_begin)
        chunk_tokens = chunk_pages * page_size
        module.launch_all_lf_pf_staged(
            dst_k,
            dst_v,
            dst_indices[
                page_begin * page_size : (page_begin + chunk_pages) * page_size
            ],
            staging_k[:chunk_tokens],
            staging_v[:chunk_tokens],
            src_page_indices[page_begin : page_begin + chunk_pages],
            k_ptr_src,
            v_ptr_src,
            page_size,
        )


@debug_kernel_api
def transfer_hicache_all_layer_mla_staged_lf_pf(
    ptr_src: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    staging: torch.Tensor,
    dst: torch.Tensor,
    *,
    page_size: int,
    element_size: int | None = None,
    unroll: int | None = None,
    block_quota: int | None = None,
) -> None:
    element_dim = staging[0, 0].numel()
    element_size = element_size or (element_dim * staging.element_size())
    block_quota = block_quota or DEFAULT_BLOCK_QUOTA
    unroll = unroll or _default_unroll(element_size)
    src_page_indices = src_indices[::page_size].contiguous()
    module = _jit_hicache_staged_module(
        element_size=element_size,
        unroll=unroll,
        block_quota=block_quota,
    )
    staging_page_capacity = staging.shape[0] // page_size
    staging = staging.view(staging.shape[0], staging.shape[1], -1)
    dst = dst.view(dst.shape[0], dst.shape[1], -1)
    for page_begin in range(0, src_page_indices.numel(), staging_page_capacity):
        chunk_pages = min(staging_page_capacity, src_page_indices.numel() - page_begin)
        chunk_tokens = chunk_pages * page_size
        module.launch_all_mla_lf_pf_staged(
            dst,
            dst_indices[
                page_begin * page_size : (page_begin + chunk_pages) * page_size
            ],
            staging[:chunk_tokens],
            src_page_indices[page_begin : page_begin + chunk_pages],
            ptr_src,
            page_size,
        )


# A separate module from `_jit_hicache_module`: this one instantiates
# HiCacheKernel with `kElementSize` set to one head group's token row rather than
# the whole row, which is often not the 128-byte multiple `run_one` / `run_all`
# need. Only the exported entry points are instantiated, so those never compile
# here.
@cache_once
def _jit_hicache_page_unified_module(*, element_size: int, block_quota: int) -> Module:
    args = make_cpp_args(
        element_size,
        1,  # kUnroll: unread by this entry point, which always moves 16B per thread
        block_quota,
        1024,  # num_threads, can be tuned for performance
    )
    return load_jit(
        "hicache_page_unified",
        *args,
        cuda_files=[
            "kvcacheio/hicache.cuh",
        ],
        cuda_wrappers=[
            (
                "launch_one_page_unified",
                f"&HiCacheKernel<{args}>::run_one_page_unified<false>",
            ),
            (
                "launch_one_mla_page_unified",
                f"&HiCacheKernel<{args}>::run_one_page_unified<true>",
            ),
        ],
    )


def can_use_page_unified_load_back_jit_kernel(
    *,
    group_bytes: int,
    block_quota: int | None = None,  # can be tuned for less interference
) -> bool:
    logger = logging.getLogger(__name__)
    if group_bytes <= 0 or group_bytes % 16 != 0:
        logger.warning(f"Unsupported {group_bytes = } for page-unified JIT load-back")
        return False
    try:
        _jit_hicache_page_unified_module(
            element_size=group_bytes,
            block_quota=block_quota or DEFAULT_BLOCK_QUOTA,
        )
        return True
    except Exception as e:
        logger.warning(f"Failed to load page-unified JIT load-back kernel: {e}")
        return False


@debug_kernel_api
def transfer_hicache_one_layer_page_unified_lf(
    k_cache_dst: torch.Tensor,
    v_cache_dst: torch.Tensor,
    src: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    *,
    block_quota: int | None = None,  # can be tuned for less interference
) -> None:
    """Load one layer of page_unified host pages into the device KV pool.

    ``src`` is a contiguous host tensor with layout
    (page, head_group, layer, 2, page_size, head_in_group, dim); K=0, V=1 --
    the order the page-unified write-back produces. It must be pinned or
    CUDA-registered, because the kernel reads it in place over its device
    mapping; no staging buffer is allocated.

    ``k_cache_dst`` / ``v_cache_dst`` are one layer of the device pool, shaped
    (token, head, dim) with the pool's full head count. ``src_indices`` holds
    host-pool TOKEN indices (the page is ``index // page_size``) and
    ``dst_indices`` the device-pool token indices they land on, in matching
    order; both are CUDA int32/int64 vectors on the destination device.
    One head group's token row must be a positive multiple of 16 bytes.

    Runs on the current CUDA stream. Keep ``src`` alive and unwritten until
    that stream completes.
    """
    if src.ndim != 7:
        raise ValueError(
            "Expected (page, head_group, layer, 2, page_size, head_in_group, dim)"
        )
    if src.shape[3] != 2 or any(d <= 0 for d in src.shape[1:]):
        raise ValueError(
            "Page dimensions must be positive and the K/V dimension must be 2"
        )
    _transfer_hicache_one_layer_page_unified(
        k_cache_dst,
        v_cache_dst,
        src,
        src_indices,
        dst_indices,
        layer_id=layer_id,
        group_bytes=src.shape[5] * src.shape[6] * src.element_size(),
        num_layers=src.shape[2],
        num_groups=src.shape[1],
        page_size=src.shape[4],
        is_mla=False,
        block_quota=block_quota,
    )


@debug_kernel_api
def transfer_hicache_one_layer_mla_page_unified_lf(
    cache_dst: torch.Tensor,
    src: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    *,
    block_quota: int | None = None,
) -> None:
    """Load one layer of compressed MLA pages in (page, layer, page_size, dim) order.

    MLA has one latent cache per layer, with no head-group or separate K/V
    axes; ``dim`` covers all stored latent and positional components.
    ``cache_dst`` is one layer of the device pool, shaped (token, 1, dim).
    Host memory, index and stream requirements match the MHA entry point.
    """
    if src.ndim != 4:
        raise ValueError("Expected MLA (page, layer, page_size, dim)")
    if any(d <= 0 for d in src.shape[1:]):
        raise ValueError("MLA page dimensions must be positive")
    _transfer_hicache_one_layer_page_unified(
        cache_dst,
        cache_dst,
        src,
        src_indices,
        dst_indices,
        layer_id=layer_id,
        group_bytes=src.shape[3] * src.element_size(),
        num_layers=src.shape[1],
        num_groups=1,
        page_size=src.shape[2],
        is_mla=True,
        block_quota=block_quota,
    )


def _transfer_hicache_one_layer_page_unified(
    k_cache_dst: torch.Tensor,
    v_cache_dst: torch.Tensor,
    src: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    *,
    layer_id: int,
    group_bytes: int,
    num_layers: int,
    num_groups: int,
    page_size: int,
    is_mla: bool,
    block_quota: int | None,
) -> None:
    if not src.is_contiguous():
        raise ValueError("The host pool must be contiguous")
    if src.device.type != "cpu" and not src.is_cuda:
        raise ValueError("Expected a CPU or CUDA-registered host pool")
    if group_bytes % 16:
        raise ValueError("Each copied token row must be 16-byte aligned")
    module = _jit_hicache_page_unified_module(
        element_size=group_bytes,
        block_quota=block_quota or DEFAULT_BLOCK_QUOTA,
    )
    launch = (
        module.launch_one_mla_page_unified if is_mla else module.launch_one_page_unified
    )
    launch(
        k_cache_dst.view(k_cache_dst.shape[0], -1),
        v_cache_dst.view(v_cache_dst.shape[0], -1),
        src.view(src.shape[0], -1),
        src_indices,
        dst_indices,
        layer_id,
        num_layers,
        num_groups,
        page_size,
    )
