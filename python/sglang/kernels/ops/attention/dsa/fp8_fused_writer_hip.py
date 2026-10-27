# SPDX-License-Identifier: Apache-2.0
"""AITER fused FP8 DSA indexer writer (RoPE, K LayerNorm, quant, cache write)."""

from __future__ import annotations

from typing import Optional

import torch

SUPPORTED_HEAD_DIM = 128
SUPPORTED_ROPE_DIM = 64
SUPPORTED_QUANT_BLOCK_SIZE = 128


def fused_fp8_writer_geometry_supported(
    head_dim: int, rope_dim: int, quant_block_size: int
) -> bool:
    return (
        head_dim == SUPPORTED_HEAD_DIM
        and rope_dim == SUPPORTED_ROPE_DIM
        and quant_block_size == SUPPORTED_QUANT_BLOCK_SIZE
    )


def aiter_fused_fp8_writer_available() -> bool:
    from sglang.srt.utils import is_gfx95_supported

    # gfx950-only kernel. gfx942 may still import the Python symbol.
    if not is_gfx95_supported():
        return False
    try:
        from aiter.ops.cache import indexer_qk_rope_quant_and_cache  # noqa: F401
    except (ImportError, AttributeError):
        return False
    return True


def prepare_aiter_rope_caches(
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    # AITER RotaryEmbedding stores CPU [max_pos, 1, 1, rotary_dim/2]; the fused
    # kernel wants device [max_pos, rotary_dim/2].
    def _prepare(table: torch.Tensor) -> torch.Tensor:
        prepared = table.squeeze()
        if prepared.dim() != 2:
            raise ValueError(
                "fused FP8 writer expects RoPE table [max_pos, rotary_dim/2], "
                f"got {tuple(table.shape)}"
            )
        return prepared.to(device=device, dtype=dtype).contiguous()

    return _prepare(cos_cache), _prepare(sin_cache)


def aiter_fused_fp8_qk_write(
    q: torch.Tensor,
    q_out: torch.Tensor,
    weights: torch.Tensor,
    weights_out: torch.Tensor,
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    positions: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    epsilon: float,
    quant_block_size: int,
    scale_fmt: Optional[str],
    weights_scale: float,
    *,
    preshuffle: bool,
    is_neox: bool,
    compute_all_q_rope: bool,
) -> None:
    if not fused_fp8_writer_geometry_supported(
        k.shape[-1], cos_cache.shape[-1] * 2, quant_block_size
    ):
        raise ValueError(
            "fused FP8 writer requires "
            f"head_dim={SUPPORTED_HEAD_DIM}, rope_dim={SUPPORTED_ROPE_DIM}, "
            f"quant_block_size={SUPPORTED_QUANT_BLOCK_SIZE}; got "
            f"head_dim={k.shape[-1]}, rope_dim={cos_cache.shape[-1] * 2}, "
            f"quant_block_size={quant_block_size}"
        )
    if norm_weight.dtype != torch.float32 or norm_bias.dtype != torch.float32:
        raise TypeError(
            "fused FP8 writer reads LayerNorm parameters as fp32; got "
            f"{norm_weight.dtype} and {norm_bias.dtype}"
        )

    from aiter.ops.cache import indexer_qk_rope_quant_and_cache

    indexer_qk_rope_quant_and_cache(
        q,
        q_out,
        weights,
        weights_out,
        k,
        kv_cache,
        slot_mapping,
        norm_weight,
        norm_bias,
        positions,
        cos_cache,
        sin_cache,
        epsilon,
        quant_block_size,
        scale_fmt,
        weights_scale,
        preshuffle=preshuffle,
        is_neox=is_neox,
        compute_all_q_rope=compute_all_q_rope,
    )
