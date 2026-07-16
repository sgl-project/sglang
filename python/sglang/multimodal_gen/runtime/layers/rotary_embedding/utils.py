"""Primitive RoPE ops: rotate helpers and apply_rotary_emb utilities."""

from typing import Optional, Tuple

import torch

from sglang.jit_kernel.diffusion.triton.rotary import apply_rotary_embedding
from sglang.kernel_api_logging import debug_kernel_api
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.utils.custom_op import register_custom_op_from_extern

logger = init_logger(__name__)

_is_cuda = current_platform.is_cuda()
if _is_cuda:
    try:
        from flashinfer.rope import (
            apply_rope_with_cos_sin_cache_inplace as _flashinfer_apply_rope_inplace,
        )
    except Exception:
        _flashinfer_apply_rope_inplace = None
else:
    _flashinfer_apply_rope_inplace = None

if _flashinfer_apply_rope_inplace is not None:
    flashinfer_apply_rope_inplace = register_custom_op_from_extern(
        _flashinfer_apply_rope_inplace,
        op_name="flashinfer_apply_rope_with_cos_sin_cache_inplace",
        mutates_args=["query", "key"],
    )
else:
    flashinfer_apply_rope_inplace = None


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


def _apply_rotary_emb_complex(
    x: torch.Tensor, # [b, s, h, d]
    freqs: torch.Tensor, # [s, 1, d // 2]
) -> torch.Tensor: # [b, s, h, d]
    """
    Apply complex rotary positional embeddings designed for interleaved=True, neox_style=False.
    Works by mathematically mapping the complex multiplication 
    (a + ib) * (cos + isin) to the interleaved layout.

    Args:
        x: Input activation tensor in bf16/fp16.
            Shape: [batch, num_tokens, num_heads, head_size]
        freqs: Complex-valued frequency tensor in complex64 format.
            Shape: [num_tokens, 1, head_size // 2]

    Returns:
        torch.Tensor: The same shape and dtype as x.
    """
    b, s, h, d = x.shape
    dtype_c = torch.float32

    x_complex = torch.view_as_complex(
        x.to(dtype_c).reshape(b, s, h, d // 2, 2)
    )
    x_out = torch.view_as_real(x_complex * freqs)
    x_out = x_out.view(b, s, h, d)
    return x_out.to(x.dtype)
    

@debug_kernel_api
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
    if q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]:
        raise ValueError(
            f"q and k must share batch, sequence, and head size, got {q.shape} vs {k.shape}"
        )

    if not (isinstance(cos_sin_cache, torch.Tensor) and cos_sin_cache.dim() == 2):
        raise ValueError("cos_sin_cache must be a 2D torch.Tensor")

    bsz, seqlen, q_heads, d = q.shape
    k_heads = k.shape[2]
    rope_dim = cos_sin_cache.shape[-1]
    if k.device != q.device or cos_sin_cache.device != q.device:
        raise ValueError(
            "q, k, and cos_sin_cache must be on the same device, "
            f"got q={q.device}, k={k.device}, cos_sin_cache={cos_sin_cache.device}"
        )
    if rope_dim % 2 != 0 or rope_dim > d:
        raise ValueError(
            f"cos_sin_cache width must be even and <= head_size, got {rope_dim} vs {d}"
        )
    if head_size is None:
        head_size = d
    if head_size != d:
        raise ValueError(f"head_size mismatch: inferred {d}, but head_size={head_size}")

    use_flashinfer = (
        flashinfer_apply_rope_inplace is not None
        and q.is_cuda
        and k.is_cuda
        and cos_sin_cache.is_cuda
        and q_heads == k_heads
    )

    if not use_flashinfer:
        if flashinfer_apply_rope_inplace is None:
            _warn_about_missing_flashinfer()

        half_size = rope_dim // 2
        if positions is None:
            cos = cos_sin_cache[:seqlen, :half_size].to(q.dtype)
            sin = cos_sin_cache[:seqlen, half_size:].to(q.dtype)
            cos = cos.unsqueeze(0).expand(bsz, -1, -1).reshape(bsz * seqlen, -1)
            sin = sin.unsqueeze(0).expand(bsz, -1, -1).reshape(bsz * seqlen, -1)
        else:
            positions = positions.to(device=q.device, dtype=torch.long).view(-1)
            cos = cos_sin_cache[positions, :half_size].to(q.dtype)
            sin = cos_sin_cache[positions, half_size:].to(q.dtype)

        if current_platform.is_npu():
            q_flat = q.reshape(bsz * seqlen, q_heads, d)
            k_flat = k.reshape(bsz * seqlen, k_heads, d)
            q_rot = apply_rotary_embedding(q_flat, cos, sin, interleaved=not is_neox)
            k_rot = apply_rotary_embedding(k_flat, cos, sin, interleaved=not is_neox)
            return q_rot.view(bsz, seqlen, q_heads, d), k_rot.view(
                bsz, seqlen, k_heads, d
            )

        def apply_rope_prefix(x: torch.Tensor, num_heads: int) -> torch.Tensor:
            x_flat = x.reshape(bsz * seqlen, num_heads, d)
            x_rot = x_flat[..., :rope_dim]
            out_rot = torch.empty_like(x_rot)
            cos_b = cos.unsqueeze(-2)
            sin_b = sin.unsqueeze(-2)
            if is_neox:
                x1, x2 = torch.chunk(x_rot, 2, dim=-1)
                out_rot[..., :half_size] = x1 * cos_b - x2 * sin_b
                out_rot[..., half_size:] = x2 * cos_b + x1 * sin_b
            else:
                x1 = x_rot[..., ::2]
                x2 = x_rot[..., 1::2]
                out_rot[..., ::2] = x1 * cos_b - x2 * sin_b
                out_rot[..., 1::2] = x2 * cos_b + x1 * sin_b
            if rope_dim == d:
                return out_rot.view(bsz, seqlen, num_heads, d)
            out = x_flat.clone()
            out[..., :rope_dim] = out_rot
            return out.view(bsz, seqlen, num_heads, d)

        return apply_rope_prefix(q, q_heads), apply_rope_prefix(k, k_heads)

    if positions is None:
        pos_1d = torch.arange(seqlen, device=q.device, dtype=torch.long)
        positions = pos_1d if bsz == 1 else pos_1d.repeat(bsz)
    else:
        if not (isinstance(positions, torch.Tensor) and positions.dim() == 1):
            raise ValueError("positions must be a 1D Tensor")
        if positions.numel() != bsz * seqlen:
            raise ValueError(
                f"positions length must be bsz*seqlen={bsz*seqlen}, got {positions.numel()}"
            )
        positions = positions.to(device=q.device, dtype=torch.long)

    q_flat = q.reshape(bsz * seqlen, q_heads * d).contiguous()
    k_flat = k.reshape(bsz * seqlen, k_heads * d).contiguous()
    flashinfer_apply_rope_inplace(
        positions=positions,
        query=q_flat,
        key=k_flat,
        head_size=d,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
    )
    return q_flat.view(bsz, seqlen, q_heads, d), k_flat.view(bsz, seqlen, k_heads, d)


@torch.compiler.assume_constant_result
def _warn_about_missing_flashinfer():
    """
    Function to warn about the missing FlashInfer.
    Exists to not cause a graph break during the compilation.
    """
    logger.warning_once(
        "FlashInfer not available, using Triton fallback for RoPE",
    )
