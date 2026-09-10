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
_CPU_RELEASE_HEAD_DIM = 128
_CPU_RELEASE_ROPE_DIM = 96


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
    if pack_kv and cache_has_full_width:
        logger.warning("KV packing does not support full-width cos/sin caches")
        return False
    if round_norm_before_rope and cache_dtype not in (dtype, torch.float32):
        logger.warning(
            "Exact fused QKNorm+RoPE requires cache dtype %s to match activation "
            "dtype %s or use float32",
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
def can_use_fused_inplace_qknorm_rope_cpu(
    head_dim: int,
    rope_dim: int,
    is_neox: bool,
    dtype: torch.dtype,
    cache_dtype: torch.dtype = torch.float32,
    round_norm_before_rope: bool = False,
) -> bool:
    if dtype != torch.bfloat16 or cache_dtype != torch.bfloat16:
        return False
    if head_dim != _CPU_RELEASE_HEAD_DIM or rope_dim != _CPU_RELEASE_ROPE_DIM:
        return False
    if not is_neox or not round_norm_before_rope:
        return False
    try:
        import sgl_kernel  # noqa: F401

        return hasattr(torch.ops.sgl_kernel, "fused_inplace_qknorm_rope_cpu")
    except Exception as exc:
        logger.warning("Failed to load CPU fused QK norm + RoPE kernel: %s", exc)
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


@register_custom_op(op_name="fused_inplace_qknorm_rope", mutates_args=["q", "k"])
def _fused_inplace_qknorm_rope_cuda(
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
    if q.device.type == "cpu":
        import sgl_kernel  # noqa: F401

        torch.ops.sgl_kernel.fused_inplace_qknorm_rope_cpu(
            q, k, q_weight, k_weight, cos_sin_cache, positions, eps,
            head_dim, rope_dim, is_neox, round_norm_before_rope
        )
        return
    _fused_inplace_qknorm_rope_cuda(
        q, k, q_weight, k_weight, cos_sin_cache, positions,
        is_neox=is_neox, eps=eps, head_dim=head_dim, rope_dim=rope_dim,
        round_norm_before_rope=round_norm_before_rope,
        cache_has_full_width=cache_has_full_width,
    )


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
