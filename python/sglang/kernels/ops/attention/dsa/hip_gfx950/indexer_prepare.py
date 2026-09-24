"""gfx950 fused DSA indexer preparation."""

from __future__ import annotations

import logging
from functools import lru_cache
from importlib import import_module
from typing import Optional, Tuple

import torch
from packaging.version import Version

logger = logging.getLogger(__name__)

MAX_FULL_INDEXER_PREPARE_ROWS = 128


def is_full_indexer_prepare_geometry_supported(
    hidden_size: int,
    heads: int,
    q_lora_rank: int,
) -> bool:
    return (
        hidden_size >= 1536
        and hidden_size % 1536 == 0
        and heads >= 16
        and heads % 16 == 0
        and q_lora_rank >= 1024
        and q_lora_rank % 512 == 0
    )


def is_full_indexer_prepare_layout_supported(
    head_dim: int,
    rope_dim: int,
    block_size: int,
    scale_fmt: Optional[str],
) -> bool:
    return (
        head_dim == 128
        and 0 < rope_dim <= head_dim
        and rope_dim % 2 == 0
        and block_size == 128
        and scale_fmt == "ue8m0"
    )


@lru_cache(maxsize=1)
def is_full_indexer_prepare_available() -> bool:
    """Return whether this runtime can JIT the gfx950 Gluon kernels."""
    try:
        triton = import_module("triton")
        if Version(Version(triton.__version__).base_version) < Version("3.5.0"):
            return False

        gl = import_module("triton.experimental.gluon.language")
        cdna4 = gl.amd.cdna4
        for obj, name in (
            (cdna4, "buffer_load"),
            (cdna4, "mfma"),
            (cdna4.async_copy, "buffer_load_to_shared"),
            (cdna4.async_copy, "commit_group"),
            (cdna4.async_copy, "wait_group"),
        ):
            getattr(obj, name)

        package = __package__
        import_module(f"{package}.indexer_prepare_m4")
        import_module(f"{package}.indexer_prepare_m128")
    except Exception as exc:
        logger.info("ROCm full indexer prepare JIT is unavailable: %s", exc)
        return False
    return True


def full_indexer_prepare(
    x: torch.Tensor,
    q_lora: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wgate: torch.Tensor,
    k_gamma: torch.Tensor,
    k_beta: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    slots: torch.Tensor,
    cache: torch.Tensor,
    *,
    eps: float,
    rope_dim: int = 64,
    is_neox_style: bool = False,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Run the low-latency decode kernels, or request native fallback."""
    rows = x.shape[0]
    hidden_size = x.shape[1]
    heads = wgate.shape[0]
    q_lora_rank = q_lora.shape[1]
    if (
        not is_full_indexer_prepare_geometry_supported(hidden_size, heads, q_lora_rank)
        or cache.shape[1] % 16 != 0
        or not is_full_indexer_prepare_layout_supported(128, rope_dim, 128, "ue8m0")
        or cos_sin.ndim != 2
        or cos_sin.shape[1] != rope_dim
    ):
        return None
    if heads == 32 and hidden_size % 1024 == 0 and rows in (64, 96, 128):
        # The large-M schedule is specialized for 32 heads and these row counts.
        from .indexer_prepare_m128 import indexer_prepare
    elif 1 <= rows <= MAX_FULL_INDEXER_PREPARE_ROWS:
        from .indexer_prepare_m4 import indexer_prepare
    else:
        return None

    return indexer_prepare(
        x,
        q_lora,
        wq,
        wk,
        wgate,
        k_gamma,
        k_beta,
        cos_sin,
        positions,
        slots,
        cache,
        eps=eps,
        rope_dim=rope_dim,
        is_neox_style=is_neox_style,
    )
