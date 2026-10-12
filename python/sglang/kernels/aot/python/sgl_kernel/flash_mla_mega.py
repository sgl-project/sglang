"""FlashMLA MegaAttention wheel APIs for DeepSeek-V4.1 on SM100/SM103.

Q must be unrotated BF16 with 16-element chunks interleaved across 64/128
heads. Outputs have 32-element chunks interleaved across eight heads per WO-A
group, and packed per-32 UE8M0 scales with TMA-aligned token-major strides.
The existing FlashMLA APIs and cache-format defaults are unchanged.
"""

from typing import Optional

import torch
from sgl_kernel.flash_mla import _flashmla_import_error


def is_flash_mla_mega_available() -> bool:
    return _flashmla_import_error is None and hasattr(
        torch.ops.sgl_kernel, "flash_mla_mega_decode"
    )


def _require_mega() -> None:
    if not is_flash_mla_mega_available():
        raise ImportError(
            "FlashMLA MegaAttention requires sgl-kernel built with CUDA >= 12.9"
        )


def flash_mla_mega_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    softmax_scale: float,
    *,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gathered BF16 KV -> (FP8 output, packed scales, max logits, LSE)."""
    _require_mega()
    return tuple(
        torch.ops.sgl_kernel.flash_mla_mega_prefill.default(
            q,
            kv,
            indices,
            positions,
            cos_sin_cache,
            softmax_scale,
            q.shape[1] // 8,
            attn_sink,
            topk_length,
        )
    )


def flash_mla_mega_decode(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    softmax_scale: float,
    *,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_kv: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Paged KV -> (FP8 output, packed scales, LSE).

    Main KV accepts V4 (584 B) or V4.1 FP8 (528 B). Extra KV can use the
    matching format or V4.1 NVFP4 (288 B) beside V4.1 FP8 main KV.
    Positions are contiguous int32, and cosine/sine tables contiguous float32.
    """
    _require_mega()
    return tuple(
        torch.ops.sgl_kernel.flash_mla_mega_decode.default(
            q,
            kv,
            indices,
            positions,
            cos_sin_cache,
            softmax_scale,
            q.shape[1] // 8,
            attn_sink,
            topk_length,
            extra_kv,
            extra_indices,
            extra_topk_length,
        )
    )


def permute_mega_q_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Permute FP8 WQ-B and packed per-row UE8M0 scales into fused Q order."""
    _require_mega()
    return tuple(
        torch.ops.sgl_kernel.flash_mla_mega_permute_q.default(weight, scale, num_heads)
    )


def permute_mega_wo_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Permute grouped FP8 WO-A and per-32 scales into fused output order."""
    _require_mega()
    return tuple(torch.ops.sgl_kernel.flash_mla_mega_permute_wo.default(weight, scale))
