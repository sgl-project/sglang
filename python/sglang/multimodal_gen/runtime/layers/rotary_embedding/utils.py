"""Primitive RoPE ops: rotate helpers and apply_rotary_emb utilities."""

import logging
from typing import Optional, Tuple

import torch

from sglang.jit_kernel.diffusion.triton.rotary import apply_rotary_embedding
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.common import get_bool_env_var

logger = logging.getLogger(__name__)

_is_hip = current_platform.is_hip()
_use_aiter_rope = get_bool_env_var("SGLANG_USE_AITER_ROPE") and _is_hip

_aiter_rope_cached_2c_fwd = None
if _use_aiter_rope:
    try:
        from aiter.ops.rope import rope_cached_2c_fwd

        _aiter_rope_cached_2c_fwd = rope_cached_2c_fwd
        logger.info("AITer RoPE enabled (SGLANG_USE_AITER_ROPE=1)")
    except ImportError:
        _use_aiter_rope = False
        logger.warning("SGLANG_USE_AITER_ROPE=1 but aiter.ops.rope not available")


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
    interleaved: bool = False,
) -> torch.Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size] or [num_tokens, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    # cos = cos.unsqueeze(-2).to(x.dtype)
    # sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
        if is_neox_style:
            x1, x2 = torch.chunk(x, 2, dim=-1)
        else:
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
        o1 = (x1.float() * cos - x2.float() * sin).type_as(x)
        o2 = (x2.float() * cos + x1.float() * sin).type_as(x)
        return torch.cat((o1, o2), dim=-1)
    else:
        return apply_rotary_embedding(x, cos, sin, interleaved)


def apply_flashinfer_rope_qk_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    head_size: Optional[int] = None,
    is_neox: bool = False,
    positions: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if q.dim() != 4 or k.dim() != 4:
        raise ValueError(
            f"Expected q/k to be 4D [bsz, seqlen, nheads, head_size], "
            f"got q:{tuple(q.shape)} k:{tuple(k.shape)}"
        )
    if q.shape != k.shape:
        raise ValueError(
            f"q and k must have the same shape, got {q.shape} vs {k.shape}"
        )

    if not (isinstance(cos_sin_cache, torch.Tensor) and cos_sin_cache.dim() == 2):
        raise ValueError("cos_sin_cache must be a 2D torch.Tensor")

    bsz, seqlen, nheads, d = q.shape
    if head_size is None:
        head_size = d
    if head_size != d:
        raise ValueError(f"head_size mismatch: inferred {d}, but head_size={head_size}")

    try:
        from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace
    except ImportError:
        # Triton fallback for AMD/ROCm where FlashInfer is not available
        import warnings

        warnings.warn(
            "FlashInfer not available, using Triton fallback for RoPE",
            stacklevel=2,
        )
        half_size = cos_sin_cache.shape[-1] // 2
        if positions is None:
            cos = cos_sin_cache[:seqlen, :half_size].to(q.dtype)
            sin = cos_sin_cache[:seqlen, half_size:].to(q.dtype)
            cos = cos.unsqueeze(0).expand(bsz, -1, -1).reshape(bsz * seqlen, -1)
            sin = sin.unsqueeze(0).expand(bsz, -1, -1).reshape(bsz * seqlen, -1)
        else:
            positions = positions.to(cos_sin_cache.device).view(-1)
            cos = cos_sin_cache[positions, :half_size].to(q.dtype)
            sin = cos_sin_cache[positions, half_size:].to(q.dtype)
        q_flat = q.reshape(bsz * seqlen, nheads, d)
        k_flat = k.reshape(bsz * seqlen, nheads, d)
        q_rot = apply_rotary_embedding(q_flat, cos, sin, interleaved=not is_neox)
        k_rot = apply_rotary_embedding(k_flat, cos, sin, interleaved=not is_neox)
        return q_rot.view(bsz, seqlen, nheads, d), k_rot.view(bsz, seqlen, nheads, d)

    if positions is None:
        pos_1d = torch.arange(seqlen, device=q.device, dtype=torch.long)
        positions = pos_1d if bsz == 1 else pos_1d.repeat(bsz)
    else:
        if not (
            isinstance(positions, torch.Tensor)
            and positions.dtype == torch.long
            and positions.dim() == 1
        ):
            raise ValueError("positions must be a 1D torch.long Tensor")
        if positions.numel() != bsz * seqlen:
            raise ValueError(
                f"positions length must be bsz*seqlen={bsz*seqlen}, got {positions.numel()}"
            )

    q_flat = q.reshape(bsz * seqlen, nheads * d).contiguous()
    k_flat = k.reshape(bsz * seqlen, nheads * d).contiguous()
    apply_rope_with_cos_sin_cache_inplace(
        positions=positions,
        query=q_flat,
        key=k_flat,
        head_size=d,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
    )
    return q_flat.view(bsz, seqlen, nheads, d), k_flat.view(bsz, seqlen, nheads, d)


def apply_aiter_rope_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to Q and K simultaneously using AITer rope_cached_2c_fwd.

    Processes both Q and K in a single kernel launch instead of two separate
    Triton kernel calls. Requires SGLANG_USE_AITER_ROPE=1 on ROCm.

    Args:
        q: [batch, seq_len, num_heads, head_dim] or [seq_len, num_heads, head_dim]
        k: same shape as q
        cos: [seq_len, head_dim // 2]
        sin: [seq_len, head_dim // 2]
        is_neox_style: True for NEOX (split-half), False for GPT-J (interleaved).
    """
    assert (
        _aiter_rope_cached_2c_fwd is not None
    ), "AITer RoPE not available. Set SGLANG_USE_AITER_ROPE=1 on ROCm with aiter installed."

    is_3d = q.dim() == 3
    if is_3d:
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)

    bsz, seqlen, nheads_q, head_dim = q.shape
    nheads_k = k.shape[2]

    # AITer expects sbhd; transpose bshd → sbhd.
    # For batch=1 (common in diffusion inference) this is a zero-cost view.
    q_sbhd = q.transpose(0, 1).contiguous()
    k_sbhd = k.transpose(0, 1).contiguous()

    # AITer expects cos/sin in [s, 1, 1, d//2] with float32.
    cos_4d = cos[:seqlen].to(torch.float32).view(seqlen, 1, 1, -1).contiguous()
    sin_4d = sin[:seqlen].to(torch.float32).view(seqlen, 1, 1, -1).contiguous()

    # rotate_style: 0 = NEOX (split-half), 1 = GPT-J (interleaved pairs)
    rotate_style = 0 if is_neox_style else 1

    q_out, k_out = _aiter_rope_cached_2c_fwd(
        q_sbhd,
        k_sbhd,
        cos_4d,
        sin_4d,
        rotate_style=rotate_style,
        reuse_freqs_front_part=True,
        nope_first=False,
        transpose_output=True,
    )

    # transpose_output=True → output memory is bshd, shape is sbhd.
    # .transpose(0,1) restores bshd shape with contiguous memory.
    q_out = q_out.transpose(0, 1)
    k_out = k_out.transpose(0, 1)

    if is_3d:
        q_out = q_out.squeeze(0)
        k_out = k_out.squeeze(0)

    return q_out, k_out
