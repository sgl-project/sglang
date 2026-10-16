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
    _require_mega()
    return tuple(
        torch.ops.sgl_kernel.flash_mla_mega_permute_q.default(weight, scale, num_heads)
    )


def permute_mega_wo_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_mega()
    return tuple(torch.ops.sgl_kernel.flash_mla_mega_permute_wo.default(weight, scale))
