from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

# Mirrors device::kWarpThreads in include/sgl_kernel/utils.cuh (32 on CUDA and HIP).
_WARP_THREADS = 32


@cache_once
def _jit_kvcache_module(k_row_bytes: int, v_row_bytes: int, num_threads: int) -> Module:
    if num_threads == 0:
        num_threads = 32
        # rare case. just don't optimize it
        if k_row_bytes % num_threads != 0 or v_row_bytes % num_threads != 0:
            return _jit_kvcache_module(k_row_bytes, v_row_bytes, num_threads)
        k_bytes = k_row_bytes / num_threads
        v_bytes = v_row_bytes / num_threads
        # increase threads if row is too large
        while k_bytes % 8 == 0 and v_bytes % 8 == 0 and (k_bytes + v_bytes) >= 64:
            num_threads *= 2
            k_bytes /= 2
            v_bytes /= 2
        logger.debug(f"Heuristic {num_threads = } for {k_row_bytes = }, {v_row_bytes}")
        return _jit_kvcache_module(k_row_bytes, v_row_bytes, num_threads)

    args = make_cpp_args(k_row_bytes, v_row_bytes, num_threads, is_arch_support_pdl())
    return load_jit(
        "kvcache",
        *args,
        cuda_files=["elementwise/kvcache.cuh"],
        cuda_wrappers=[("store_cache", f"StoreKVCacheKernel<{args}>::run")],
    )


@cache_once
def can_use_store_cache(
    k_row_bytes: int, v_row_bytes: int = 0, num_threads: int = 0
) -> bool:
    """Whether the JIT store_cache kernel can serve these row widths.
    v_row_bytes=0 means symmetric, i.e. it defaults to k_row_bytes."""
    v_row_bytes = v_row_bytes or k_row_bytes
    try:
        _jit_kvcache_module(k_row_bytes, v_row_bytes, num_threads)
        return True
    except Exception as e:
        logger.warning(
            f"Failed to load JIT KV-Cache kernel with "
            f"k_row_bytes={k_row_bytes} v_row_bytes={v_row_bytes}: {e}"
        )
        return False


@register_custom_op(mutates_args=["k_cache", "v_cache"])
def store_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    row_bytes: int = 0,
    v_row_bytes: int = 0,
    num_split: int = 0,
    size_limit: int = 0,
    reserved_skip_index: int = 0,
) -> None:
    """Store key and value tensors into KV cache at specified indices.

    Args:
        k (torch.Tensor): Key tensor of shape (batch_size, H * D).
        v (torch.Tensor): Value tensor of shape (batch_size, H * Dv).
        k_cache (torch.Tensor): Key cache tensor of shape (num_pages, H * D).
        v_cache (torch.Tensor): Value cache tensor of shape (num_pages, H * Dv).
        indices (torch.Tensor): Indices tensor of shape (batch_size,).
        row_bytes (int): Key row width in bytes. Inferred from k when 0.
        v_row_bytes (int): Value row width in bytes; differs from row_bytes for
            asymmetric KV (head_dim != v_head_dim). Inferred from v when 0.
        num_split (int): Warps cooperating on one row. A heuristic picks it
            when 0; it is the only knob here that exists purely for tuning.
        size_limit (int): Valid slot bound (cache row count = real slots + the
            reserved padding slot); an index outside [0, size_limit) fails fast
            (device assert) instead of an illegal memory access. Defaults to the
            cache row count when 0.
        reserved_skip_index (int): If nonnegative, writes targeting this index
            are skipped. Defaults to the reserved CUDA-graph padding slot 0;
            pass -1 to disable skipping.
    """
    row_bytes = row_bytes or k.shape[-1] * k.element_size()
    v_row_bytes = v_row_bytes or v.shape[-1] * v.element_size()
    # One warp per split. The knob stays the split count it has always been:
    # renaming it changes the registered op schema, and a warm inductor cache
    # does not notice that -- it replays generated code carrying the old name.
    module = _jit_kvcache_module(row_bytes, v_row_bytes, num_split * _WARP_THREADS)
    if size_limit <= 0:
        size_limit = k_cache.shape[0]
    module.store_cache(
        k,
        v,
        k_cache,
        v_cache,
        indices,
        size_limit,
        reserved_skip_index,
    )
