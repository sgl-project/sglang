from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# The fused quant-store kernel relies on the __nv_fp8 conversion operators;
# on ROCm fp8_e4m3_t is a plain byte type, and only e4m3 has a clip constant.
QUANT_SRC_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
QUANT_DST_DTYPES = (torch.float8_e4m3fn,)


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


@register_custom_op(mutates_args=["k_cache", "v_cache"])
def store_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    row_bytes: int = 0,
    num_split: int = 0,  # can be tuned for performance
    size_limit: int = 0,
) -> None:
    """Store key and value tensors into KV cache at specified indices.

    Args:
        k (torch.Tensor): Key tensor of shape (batch_size, H * D).
        v (torch.Tensor): Value tensor of shape (batch_size, H * D).
        k_cache (torch.Tensor): Key cache tensor of shape (num_pages, H * D).
        v_cache (torch.Tensor): Value cache tensor of shape (num_pages, H * D).
        indices (torch.Tensor): Indices tensor of shape (batch_size,).
        size_limit (int): Valid slot bound (cache row count = real slots + the
            reserved padding slot); an index outside [0, size_limit) fails fast
            (device assert) instead of an illegal memory access. Defaults to the
            cache row count when 0.
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
    if size_limit <= 0:
        size_limit = k_cache.shape[0]
    module.store_cache(
        k,
        v,
        k_cache,
        v_cache,
        indices,
        num_split,
        size_limit,
    )


@cache_once
def _jit_kvcache_quant_module(
    row_elems: int, src_dtype: torch.dtype, dst_dtype: torch.dtype
) -> Module:
    args = make_cpp_args(row_elems, src_dtype, dst_dtype, is_arch_support_pdl())
    return load_jit(
        "kvcache_quant",
        *args,
        cuda_files=["elementwise/kvcache.cuh"],
        cuda_wrappers=[("store_cache_quant", f"StoreKVCacheQuantKernel<{args}>::run")],
    )


@cache_once
def can_use_store_cache_quant(
    row_elems: int, src_dtype: torch.dtype, dst_dtype: torch.dtype
) -> bool:
    logger = logging.getLogger(__name__)
    if src_dtype not in QUANT_SRC_DTYPES or dst_dtype not in QUANT_DST_DTYPES:
        return False
    vec_elems = 16 // src_dtype.itemsize
    if row_elems % vec_elems != 0:
        logger.warning(
            f"Unsupported row_elems={row_elems} for JIT quant KV-Cache kernel:"
            f" must be multiple of {vec_elems} for {src_dtype}"
        )
        return False
    try:
        _jit_kvcache_quant_module(row_elems, src_dtype, dst_dtype)
        return True
    except Exception as e:
        logger.warning(
            f"Failed to load JIT quant KV-Cache kernel "
            f"with row_elems={row_elems}: {e}"
        )
        return False


@register_custom_op(mutates_args=["k_cache", "v_cache"])
def store_cache_quant(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    *,
    k_inv_scale: float = 1.0,
    v_inv_scale: float = 1.0,
    size_limit: int = 0,
) -> None:
    """Quantize key and value tensors to FP8 and store them into the KV cache
    at specified indices in a single fused kernel. Unlike the unfused eager
    path (in-place div + dtype cast + byte store), the inputs are not mutated.

    Per-tensor scales come in one of two forms: a 1-element float32 GPU tensor
    (``k_scale`` / ``v_scale``, read on device — no host sync) or a
    host-precomputed reciprocal (``k_inv_scale`` / ``v_inv_scale``, used when
    the tensor form is None). Values are divided by the scale, clipped to the
    finite FP8 range, and round-to-nearest converted.

    Args:
        k (torch.Tensor): Key tensor of shape (batch_size, H * D), bf16/fp16/fp32.
        v (torch.Tensor): Value tensor of shape (batch_size, H * D).
        k_cache (torch.Tensor): Key cache tensor of shape (num_pages, H * D), fp8.
        v_cache (torch.Tensor): Value cache tensor of shape (num_pages, H * D), fp8.
        indices (torch.Tensor): Indices tensor of shape (batch_size,).
        size_limit (int): Valid slot bound (cache row count = real slots + the
            reserved padding slot); an index outside [0, size_limit) fails fast
            (device assert) instead of an illegal memory access. Defaults to the
            cache row count when 0.
    """
    module = _jit_kvcache_quant_module(k.shape[-1], k.dtype, k_cache.dtype)
    if size_limit <= 0:
        size_limit = k_cache.shape[0]
    module.store_cache_quant(
        k,
        v,
        k_cache,
        v_cache,
        indices,
        k_scale,
        v_scale,
        k_inv_scale,
        v_inv_scale,
        size_limit,
    )
