from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils import is_xpu, print_warning_once
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# For XPU, try to import JIT infrastructure from sgl_kernel
try:
    from sgl_kernel.jit import apply_rope_inplace as _xpu_apply_rope_inplace
    from sgl_kernel.jit import (
        apply_rope_inplace_with_kvcache as _xpu_apply_rope_inplace_with_kvcache,
    )

    _HAS_SGL_KERNEL_JIT = True
except ImportError:
    _HAS_SGL_KERNEL_JIT = False


logger = logging.getLogger(__name__)


def _native_rope_rotate(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, is_neox: bool
) -> None:
    """Rotate the first ``rope_dim`` channels of ``x`` in place (PyTorch fallback).

    ``x``: [num_tokens, num_heads, head_dim]; ``cos``/``sin``: [num_tokens, rope_dim // 2].
    NeoX splits the rotary block into halves; non-NeoX (GPT-J) uses interleaved
    even/odd pairs. Channels beyond ``rope_dim`` are left untouched.
    """
    rope_dim = cos.shape[-1] * 2
    xf = x[..., :rope_dim].float()
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    if is_neox:
        x1 = xf[..., : rope_dim // 2]
        x2 = xf[..., rope_dim // 2 :]
        rotated = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    else:
        x1 = xf[..., 0::2]
        x2 = xf[..., 1::2]
        rotated = torch.stack(
            (x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1
        ).flatten(-2)
    x[..., :rope_dim] = rotated.to(x.dtype)


def _native_apply_rope_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    is_neox: bool,
    rope_dim: int,
) -> None:
    """Generic PyTorch RoPE applied in place to q/k (XPU fallback)."""
    half = rope_dim // 2
    gathered = cos_sin_cache[positions.long()]
    cos, sin = gathered[..., :half], gathered[..., half:rope_dim]
    _native_rope_rotate(q, cos, sin, is_neox)
    _native_rope_rotate(k, cos, sin, is_neox)


@cache_once
def _jit_rotary_embedding_module() -> Module:
    return load_jit(
        "rotary_embedding",
        cuda_files=["elementwise/pos_enc.cuh"],
        cuda_wrappers=[("rotary_embedding", "RotaryEmbeddingKernel::run")],
    )


@cache_once
def _jit_fused_rope_module(is_neox: bool, rope_dim: int, dtype: torch.dtype) -> Module:
    args = make_cpp_args(is_neox, rope_dim, is_arch_support_pdl(), dtype)
    return load_jit(
        "fused_rope",
        *args,
        cuda_files=["elementwise/rope.cuh"],
        cuda_wrappers=[
            ("run_rope", f"FusedRopeKernel<{args}>::run"),
            ("run_rope_store", f"FusedRopeKernel<{args}>::run_fused"),
        ],
    )


@register_custom_op(
    op_name="rotary_embedding_with_key",
    mutates_args=["query", "key"],
)
def rotary_embedding_with_key(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
) -> None:
    module = _jit_rotary_embedding_module()
    module.rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox)


@register_custom_op(
    op_name="rotary_embedding_without_key",
    mutates_args=["query"],
)
def rotary_embedding_without_key(
    positions: torch.Tensor,
    query: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
) -> None:
    module = _jit_rotary_embedding_module()
    module.rotary_embedding(positions, query, None, head_size, cos_sin_cache, is_neox)


def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: Optional[torch.Tensor],
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
):
    if key is None:
        rotary_embedding_without_key(
            positions, query, head_size, cos_sin_cache, is_neox
        )
    else:
        rotary_embedding_with_key(
            positions, query, key, head_size, cos_sin_cache, is_neox
        )
    return query, key


@dataclass
class FusedSetKVBufferArg:
    """
    value : Optional[torch.Tensor]
        Value tensor, shape: ``(nnz, num_v_heads * head_size)``.
    k_buffer : Optional[torch.Tensor]
        Buffer for keys, shape: ``(nnz, num_k_heads * head_size)``.
    v_buffer : Optional[torch.Tensor]
        Buffer for values, shape: ``(nnz, num_v_heads * head_size)``.
    cache_loc : Optional[torch.Tensor]
        Cache location tensor, used for indexing kv cache.
    """

    value: torch.Tensor
    k_buffer: torch.Tensor
    v_buffer: torch.Tensor
    cache_loc: torch.Tensor


@register_custom_op(mutates_args=["q", "k"])
def apply_rope_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    is_neox: bool,
    rope_dim: int = 0,
) -> None:
    """
    Fused inplace rotary position embedding for query and key tensors.

    Args:
        q: Query tensor of shape [num_tokens, num_qo_heads, rope_dim].
        k: Key tensor of shape [num_tokens, num_kv_heads, rope_dim].
        cos_sin_cache: Cosine/sine cache of shape [max_position, rope_dim],
            where the first half along dim=-1 is cos and the second half is sin.
            Must be float32.
        positions: Position indices of shape [num_tokens], int32 or int64.
        is_neox: Whether to use GPT-NeoX style (True) or GPT-J interleaved style (False).
        rope_dim: Rotary embedding dimension. Defaults to cos_sin_cache.size(-1).
    """
    rope_dim = rope_dim or cos_sin_cache.size(-1)

    # Dispatch to XPU or CUDA based on device type
    if is_xpu() and q.device.type == "xpu":
        if _HAS_SGL_KERNEL_JIT:
            try:
                _xpu_apply_rope_inplace(
                    q, k, cos_sin_cache, positions, is_neox=is_neox, rope_dim=rope_dim
                )
                return
            except (ValueError, RuntimeError) as e:
                print_warning_once(
                    f"XPU JIT rope kernel failed ({e}), "
                    "falling back to native implementation"
                )
        else:
            logger.debug("sgl-kernel-xpu not installed, using native implementation")

        # Generic PyTorch fallback for XPU
        _native_apply_rope_inplace(q, k, cos_sin_cache, positions, is_neox, rope_dim)
        return

    module = _jit_fused_rope_module(is_neox, rope_dim, q.dtype)
    module.run_rope(q, k, cos_sin_cache, positions)


@register_custom_op(mutates_args=["q", "k", "k_cache", "v_cache"])
def apply_rope_inplace_with_kvcache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    out_loc: torch.Tensor,
    *,
    is_neox: bool,
    rope_dim: int = 0,
) -> None:
    """
    Fused inplace RoPE + KV cache store.

    Applies rotary position embedding to q and k inplace. The rotated k is also
    stored in k_cache. The original v is also stored in v_cache.

    Args:
        q: Query tensor of shape [num_tokens, num_qo_heads, head_dim].
        k: Key tensor of shape [num_tokens, num_kv_heads, head_dim].
        v: Value tensor of shape [num_tokens, num_kv_heads, head_dim].
        k_cache: Key cache of shape [cache_size, num_kv_heads * head_dim].
        v_cache: Value cache of shape [cache_size, num_kv_heads * head_dim].
        cos_sin_cache: Cosine/sine cache of shape [max_position, rope_dim], float32.
        positions: Position indices of shape [num_tokens], int32 or int64.
        out_loc: Cache write locations of shape [num_tokens], same dtype as positions.
        is_neox: Whether to use GPT-NeoX style (True) or GPT-J interleaved (False).
        rope_dim: Rotary embedding dimension. Defaults to cos_sin_cache.size(-1).
    """
    rope_dim = rope_dim or cos_sin_cache.size(-1)
    v = v.view_as(k)

    # Dispatch to XPU or CUDA based on device type
    if is_xpu() and q.device.type == "xpu":
        if _HAS_SGL_KERNEL_JIT:
            try:
                _xpu_apply_rope_inplace_with_kvcache(
                    q,
                    k,
                    v,
                    k_cache,
                    v_cache,
                    cos_sin_cache,
                    positions,
                    out_loc,
                    is_neox=is_neox,
                    rope_dim=rope_dim,
                )
                return
            except (ValueError, RuntimeError) as e:
                print_warning_once(
                    f"XPU JIT rope+kvcache kernel failed ({e}), "
                    "falling back to native implementation"
                )
        else:
            logger.debug("sgl-kernel-xpu not installed, using native implementation")

        # Generic PyTorch fallback for XPU: RoPE in place, then store k/v to cache.
        _native_apply_rope_inplace(q, k, cos_sin_cache, positions, is_neox, rope_dim)
        loc = out_loc.long()
        k_cache[loc] = k.reshape(k.shape[0], -1).to(k_cache.dtype)
        v_cache[loc] = v.reshape(v.shape[0], -1).to(v_cache.dtype)
        return

    module = _jit_fused_rope_module(is_neox, rope_dim, q.dtype)
    module.run_rope_store(q, k, v, k_cache, v_cache, cos_sin_cache, positions, out_loc)


# NOTE: this name is intentionally set as the old kernel in `sgl_kernel`
def apply_rope_with_cos_sin_cache_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    is_neox: bool,
    rope_dim: int = 0,
    fused_args: Optional[FusedSetKVBufferArg] = None,
) -> None:
    """
    Apply RoPE to q and k inplace, with optional fused kv cache store.

    If `fused_args` is provided, it will perform fused RoPE and KV cache store.
    Otherwise, it will only apply RoPE inplace.

    Args:
        q: Query tensor of shape [num_tokens, num_qo_heads, head_dim].
        k: Key tensor of shape [num_tokens, num_kv_heads, head_dim].
        cos_sin_cache: Cosine/sine cache of shape [max_position, rope_dim], float32.
        positions: Position indices of shape [num_tokens], int32 or int64.
        is_neox: Whether to use GPT-NeoX style (True) or GPT-J interleaved (False).
        rope_dim: Rotary embedding dimension. Defaults to cos_sin_cache.size(-1).
        fused_args: Optional arguments for fused RoPE + KV cache store. If None,
            only RoPE will be applied inplace without touching kv cache.
    """
    if fused_args is not None:
        apply_rope_inplace_with_kvcache(
            q,
            k,
            fused_args.value,
            fused_args.k_buffer,
            fused_args.v_buffer,
            cos_sin_cache,
            positions,
            fused_args.cache_loc,
            is_neox=is_neox,
            rope_dim=rope_dim,
        )
    else:
        apply_rope_inplace(
            q, k, cos_sin_cache, positions, is_neox=is_neox, rope_dim=rope_dim
        )
