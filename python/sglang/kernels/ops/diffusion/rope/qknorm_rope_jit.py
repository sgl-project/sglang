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

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
_SUPPORTED_CACHE_DTYPES = (*_SUPPORTED_DTYPES, torch.float32)


@cache_once
def _jit_qknorm_rope_module(
    head_dim: int,
    rope_dim: int,
    is_neox: bool,
    dtype: torch.dtype,
    cache_dtype: torch.dtype,
    round_norm_before_rope: bool,
    pack_kv: bool = False,
    cache_has_full_width: bool = False,
) -> Module:
    args = make_cpp_args(
        head_dim,
        rope_dim,
        is_neox,
        is_arch_support_pdl(),
        dtype,
        cache_dtype,
        round_norm_before_rope,
        cache_has_full_width,
    )
    op_name = "qknorm_rope_pack_kv" if pack_kv else "qknorm_rope"
    kernel_name = "QKNormRopePackKVKernel" if pack_kv else "QKNormRopeKernel"
    return load_jit(
        op_name,
        *args,
        cuda_files=["diffusion/qknorm_rope.cuh"],
        cuda_wrappers=[(op_name, f"{kernel_name}<{args}>::run")],
    )


def _can_use_fused_qknorm_rope(
    head_dim: int,
    rope_dim: int,
    is_neox: bool,
    dtype: torch.dtype,
    cache_dtype: torch.dtype,
    round_norm_before_rope: bool,
    pack_kv: bool,
    cache_has_full_width: bool,
) -> bool:
    if dtype not in _SUPPORTED_DTYPES or cache_dtype not in _SUPPORTED_CACHE_DTYPES:
        logger.warning(
            "Unsupported dtype pair (%s, %s) for JIT fused QKNorm+RoPE",
            dtype,
            cache_dtype,
        )
        return False
    if head_dim not in (64, 128, 256):
        logger.warning(f"Unsupported head_dim={head_dim} for JIT fused QKNorm+RoPE")
        return False
    if rope_dim <= 0 or rope_dim > head_dim:
        logger.warning(
            f"Unsupported rope_dim={rope_dim} for head_dim={head_dim} in fused QKNorm+RoPE"
        )
        return False
    elems_per_thread = head_dim // 32
    if rope_dim % elems_per_thread != 0:
        logger.warning(
            "rope_dim=%s must be divisible by per-thread width=%s for fused QKNorm+RoPE",
            rope_dim,
            elems_per_thread,
        )
        return False
    if is_neox:
        rotary_lanes = rope_dim // elems_per_thread
        if rotary_lanes < 2 or rotary_lanes % 2:
            logger.warning(
                "rope_dim=%s yields invalid rotary_lanes=%s for neox fused QKNorm+RoPE; rotary lane count must be even",
                rope_dim,
                rotary_lanes,
            )
            return False
    elif cache_has_full_width:
        logger.warning("Full-width cos/sin caches are only supported for NeoX RoPE")
        return False
    if pack_kv and cache_has_full_width:
        logger.warning("KV packing does not support full-width cos/sin caches")
        return False
    if round_norm_before_rope and cache_dtype != dtype:
        logger.warning(
            "Exact fused QKNorm+RoPE requires cache dtype %s to match activation dtype %s",
            cache_dtype,
            dtype,
        )
        return False
    try:
        _jit_qknorm_rope_module(
            head_dim,
            rope_dim,
            is_neox,
            dtype,
            cache_dtype,
            round_norm_before_rope,
            pack_kv,
            cache_has_full_width,
        )
        return True
    except Exception as e:
        suffix = "+KV pack" if pack_kv else ""
        logger.warning(f"Failed to load JIT fused QKNorm+RoPE{suffix} kernel: {e}")
        return False


@torch.compiler.assume_constant_result
@cache_once
def can_use_fused_inplace_qknorm_rope(
    head_dim: int,
    rope_dim: int,
    is_neox: bool,
    dtype: torch.dtype,
    cache_dtype: torch.dtype = torch.float32,
    round_norm_before_rope: bool = False,
    pack_kv: bool = False,
    cache_has_full_width: bool = False,
) -> bool:
    return _can_use_fused_qknorm_rope(
        head_dim,
        rope_dim,
        is_neox,
        dtype,
        cache_dtype,
        round_norm_before_rope,
        pack_kv,
        cache_has_full_width,
    )


@register_custom_op(mutates_args=["q", "k"])
def fused_inplace_qknorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    is_neox: bool,
    eps: float = 1e-6,
    head_dim: int = 0,
    rope_dim: int = 0,
    round_norm_before_rope: bool = False,
    cache_has_full_width: bool = False,
) -> None:
    head_dim = head_dim or q.size(-1)
    if not rope_dim:
        cache_width = cos_sin_cache.size(-1)
        rope_dim = cache_width // 2 if cache_has_full_width else cache_width
    module = _jit_qknorm_rope_module(
        head_dim,
        rope_dim,
        is_neox,
        q.dtype,
        cos_sin_cache.dtype,
        round_norm_before_rope,
        False,
        cache_has_full_width,
    )
    module.qknorm_rope(q, k, q_weight, k_weight, cos_sin_cache, positions, eps)


@register_custom_op(mutates_args=["q", "packed_kv"])
def fused_qknorm_rope_pack_kv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_prefix: torch.Tensor,
    v_prefix: torch.Tensor,
    packed_kv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    is_neox: bool,
    eps: float = 1e-6,
    head_dim: int = 0,
    rope_dim: int = 0,
    round_norm_before_rope: bool = False,
) -> None:
    head_dim = head_dim or q.size(-1)
    rope_dim = rope_dim or cos_sin_cache.size(-1)
    batch_size, suffix_tokens = q.shape[:2]
    prefix_tokens = k_prefix.shape[1]
    module = _jit_qknorm_rope_module(
        head_dim,
        rope_dim,
        is_neox,
        q.dtype,
        cos_sin_cache.dtype,
        round_norm_before_rope,
        True,
        False,
    )
    module.qknorm_rope_pack_kv(
        q.view(-1, q.shape[-2], head_dim),
        k.view(-1, k.shape[-2], head_dim),
        v.view(-1, v.shape[-2], head_dim),
        k_prefix.view(-1, k_prefix.shape[-2], head_dim),
        v_prefix.view(-1, v_prefix.shape[-2], head_dim),
        packed_kv[0],
        packed_kv[1],
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        batch_size,
        prefix_tokens,
        suffix_tokens,
        eps,
    )
