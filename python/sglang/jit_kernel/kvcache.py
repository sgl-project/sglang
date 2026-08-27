from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_kvcache_module(row_bytes: int) -> Module:
    args = make_cpp_args(row_bytes, is_arch_support_pdl())
    return load_jit(
        "kvcache",
        *args,
        cuda_files=["elementwise/kvcache.cuh"],
        cuda_wrappers=[("store_cache", f"StoreKVCacheKernel<{args}>::run")],
    )


@cache_once
def can_use_store_cache(size: int) -> bool:
    logger = logging.getLogger(__name__)
    if size % 4 != 0:
        logger.warning(
            f"Unsupported row_bytes={size} for JIT KV-Cache kernel:"
            " must be multiple of 4"
        )
        return False
    try:
        _jit_kvcache_module(size)
        return True
    except Exception as e:
        logger.warning(
            f"Failed to load JIT KV-Cache kernel " f"with row_bytes={size}: {e}"
        )
        return False


def store_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    row_bytes: int = 0,
    num_split: int = 0,  # can be tuned for performance
) -> None:
    """Store key and value tensors into KV cache at specified indices.

    Args:
        k (torch.Tensor): Key tensor of shape (batch_size, H * D).
        v (torch.Tensor): Value tensor of shape (batch_size, H * D).
        k_cache (torch.Tensor): Key cache tensor of shape (num_pages, H * D).
        v_cache (torch.Tensor): Value cache tensor of shape (num_pages, H * D).
        indices (torch.Tensor): Indices tensor of shape (batch_size,).
    """
    row_bytes = row_bytes or k.shape[-1] * k.element_size()
    module = _jit_kvcache_module(row_bytes)
    if num_split <= 0:
        if row_bytes % 2048 == 0:
            num_split = 4
        elif row_bytes % 1024 == 0:
            num_split = 2
        else:
            num_split = 1
    module.store_cache(
        k,
        v,
        k_cache,
        v_cache,
        indices,
        num_split,
    )
