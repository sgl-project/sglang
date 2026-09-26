"""gfx950 full indexer preparation for compatible DSA shapes.

The decode kernels combine the three indexer projections with LayerNorm, RoPE,
FP8 quantization, the head-gate scale, and the paged index-K cache write.
Unsupported shapes stay on the native path.

Like the AITER writer used as the fallback, these kernels intentionally omit
Hadamard so every token in a request uses one index-cache representation even
when different row counts select different prepare paths.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from importlib import import_module
from typing import Optional, Tuple

import torch
from packaging.version import Version

logger = logging.getLogger(__name__)

MAX_FULL_INDEXER_PREPARE_ROWS = 128


@lru_cache(maxsize=1)
def is_full_indexer_prepare_available() -> bool:
    """Return whether this runtime can JIT the gfx950 Gluon kernels."""
    try:
        triton = import_module("triton")
        if Version(Version(triton.__version__).base_version) < Version("3.5.0"):
            return False

        gl = import_module("triton.experimental.gluon.language")
        cdna4 = gl.amd.cdna4
        # This check is intentionally separate from AITER's paged-MQA
        # capability: that path may be supplied by an AOT bundle even when
        # this JIT-only path is unavailable.
        for obj, name in (
            (cdna4, "buffer_load"),
            (cdna4, "mfma"),
            (cdna4.async_copy, "buffer_load_to_shared"),
            (cdna4.async_copy, "commit_group"),
            (cdna4.async_copy, "wait_group"),
        ):
            getattr(obj, name)

        package = __package__
        import_module(f"{package}.indexer_prepare_m1")
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
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Run the low-latency decode kernels, or request native fallback."""
    rows = x.shape[0]
    heads = wgate.shape[0]
    if rows == 1:
        from .indexer_prepare_m1 import indexer_prepare
    elif heads == 32 and rows in (64, 96, 128):
        # The large-M schedule uses async shared-memory projection tiles. Keep
        # this exact allowlist: other row counts can violate its layout
        # constraints, while these graph sizes are both correct and materially
        # faster than the small-M schedule. Its projection grid specializes
        # for 32 heads, so other supported head counts stay on the generic
        # schedule below.
        from .indexer_prepare_m128 import indexer_prepare
    elif 2 <= rows <= MAX_FULL_INDEXER_PREPARE_ROWS:
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
    )
