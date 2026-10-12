"""SM100/SM103 FlashMLA Q RoPE + sparse attention + inverse RoPE + MXFP8.

Q is BF16 in 16-element chunks interleaved across padded heads. The output
uses 32-element chunks interleaved across eight heads per WO-A group, with
packed UE8M0 scales in the layout consumed directly by DeepGEMM.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import triton
import triton.language as tl

from .mega_layout import HEAD_DIM, HEADS_PER_GROUP, Q_CHUNK


class MegaAttentionOutput(NamedTuple):
    data: torch.Tensor
    scale: torch.Tensor


def require_mega_attention() -> None:
    try:
        from sgl_kernel.flash_mla_mega import is_flash_mla_mega_available
    except ImportError as exc:
        raise ImportError(
            "flashmla_mega requires sgl-kernel built with CUDA >= 12.9 and "
            "the FlashMLA MegaAttention extension"
        ) from exc
    if not is_flash_mla_mega_available():
        raise ImportError("sgl-kernel was built without the FlashMLA MegaAttention ops")


@triton.jit
def _pack_query_kernel(
    Q,
    OUT,
    stride_token,
    stride_head,
    HEADS: tl.constexpr,
    PADDED_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    h = (i // CHUNK) % PADDED_HEADS
    d = i // (PADDED_HEADS * CHUNK) * CHUNK + i % CHUNK
    value = tl.load(
        Q + token * stride_token + h * stride_head + d,
        (h < HEADS) & (i < PADDED_HEADS * HEAD_DIM),
        other=0,
    )
    tl.store(
        OUT + token * PADDED_HEADS * HEAD_DIM + i,
        value,
        i < PADDED_HEADS * HEAD_DIM,
    )


def pack_mega_query(q: torch.Tensor, padded_heads: int) -> torch.Tensor:
    """Pack unrotated queries and zero padded heads in one launch.

    WQ-B keeps its checkpoint layout and its existing quantization backend.
    This replaces the split path's Q-RoPE/pad launch; Q RoPE runs in attention.
    """
    if (
        q.ndim != 3
        or not q.is_cuda
        or q.shape[-1] != HEAD_DIM
        or q.dtype != torch.bfloat16
        or q.stride(-1) != 1
        or padded_heads not in (64, 128)
        or not 0 < q.shape[1] <= padded_heads
    ):
        raise ValueError("MegaAttention needs BF16 [tokens, heads<=64/128, 512] Q")
    out = q.new_empty(q.shape[0], padded_heads, HEAD_DIM)
    if q.shape[0]:
        _pack_query_kernel[(q.shape[0], triton.cdiv(padded_heads * HEAD_DIM, 512))](
            q,
            out,
            q.stride(0),
            q.stride(1),
            HEADS=q.shape[1],
            PADDED_HEADS=padded_heads,
            HEAD_DIM=HEAD_DIM,
            CHUNK=Q_CHUNK,
            BLOCK=512,
        )
    return out


def _empty_output(q: torch.Tensor) -> MegaAttentionOutput:
    groups = q.shape[1] // HEADS_PER_GROUP
    return MegaAttentionOutput(
        torch.zeros(
            (q.shape[0], groups, HEADS_PER_GROUP * HEAD_DIM),
            device=q.device,
            dtype=torch.float8_e4m3fn,
        ),
        torch.zeros(
            (groups, HEADS_PER_GROUP * HEAD_DIM // 128, (q.shape[0] + 3) // 4 * 4),
            device=q.device,
            dtype=torch.int32,
        ).permute(2, 0, 1)[: q.shape[0]],
    )


def flashmla_mega_decode(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    softmax_scale: float,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_kv: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
) -> MegaAttentionOutput:
    """Paged FP8 SWA plus FP8/NVFP4 compressed KV -> quantized WO-A input."""
    from sgl_kernel.flash_mla_mega import flash_mla_mega_decode

    if not q.shape[0]:
        return _empty_output(q)
    data, scale, _ = flash_mla_mega_decode(
        q,
        kv,
        indices,
        positions.to(torch.int32).contiguous(),
        cos_sin_cache,
        softmax_scale,
        attn_sink=attn_sink,
        topk_length=topk_length,
        extra_kv=extra_kv,
        extra_indices=extra_indices,
        extra_topk_length=extra_topk_length,
    )
    return MegaAttentionOutput(data, scale)


def flashmla_mega_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    softmax_scale: float,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> MegaAttentionOutput:
    """Gathered BF16 sparse KV -> the same quantized output as fused decode."""
    from sgl_kernel.flash_mla_mega import flash_mla_mega_prefill

    if not q.shape[0]:
        return _empty_output(q)
    data, scale, _, _ = flash_mla_mega_prefill(
        q,
        kv,
        indices,
        positions.to(torch.int32).contiguous(),
        cos_sin_cache,
        softmax_scale,
        attn_sink=attn_sink,
        topk_length=topk_length,
    )
    return MegaAttentionOutput(data, scale)
