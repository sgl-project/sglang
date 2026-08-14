"""Shared contracts for Ascend DSA/MLA FP8 attention paths."""

from __future__ import annotations

from typing import Optional

import torch

DSA_KV_QUANT_TILE_SIZE = 128


def get_dsa_fp8_packed_cache_dim(
    *,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    tile_size: int = DSA_KV_QUANT_TILE_SIZE,
) -> int:
    """Return the byte-addressed packed DSA KV width.

    The custom Ascend kernels store one FP8 byte for every latent element,
    BF16 bytes for the RoPE part, and one FP32 scale per latent tile.
    """

    if kv_lora_rank <= 0 or qk_rope_head_dim <= 0:
        raise ValueError(
            "DSA FP8 cache dimensions must be positive, "
            f"got kv_lora_rank={kv_lora_rank}, "
            f"qk_rope_head_dim={qk_rope_head_dim}."
        )
    if tile_size <= 0 or kv_lora_rank % tile_size != 0:
        raise ValueError(
            f"kv_lora_rank {kv_lora_rank} must be divisible by tile_size "
            f"{tile_size}."
        )

    latent_fp8_bytes = kv_lora_rank
    rope_bf16_bytes = qk_rope_head_dim * 2
    scale_fp32_bytes = kv_lora_rank // tile_size * 4
    return latent_fp8_bytes + rope_bf16_bytes + scale_fp32_bytes


def normalize_required_fp8_scale(
    scale: Optional[torch.Tensor],
    *,
    name: str,
    device: torch.device,
) -> torch.Tensor:
    """Normalize a checkpoint-derived runtime scale or fail explicitly."""

    if scale is None or scale.numel() == 0:
        raise RuntimeError(
            f"{name} is required for the Ascend FP8 KV attention path. "
            "Refusing to use an implicit unit scale because it changes model "
            "outputs silently."
        )
    normalized = scale.reshape(-1).to(device=device, dtype=torch.float32)
    if not torch.isfinite(normalized).all():
        raise RuntimeError(
            f"{name} contains a non-finite value. This normally means the "
            "checkpoint scale was not loaded or post-load processing did not "
            "run."
        )
    if (normalized <= 0).any():
        raise ValueError(f"{name} must contain only positive FP8 scales.")
    return normalized
