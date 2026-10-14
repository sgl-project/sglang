"""Split-KV helpers shared by the K3 MLA target-verify kernels: the per-request
KV chunking and the default split count."""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

try:
    from triton.experimental import gluon  # noqa: F401

    _HAS_GLUON = True
except ImportError:  # pragma: no cover
    _HAS_GLUON = False


@functools.lru_cache(maxsize=8)
def _ones(device):
    return torch.ones(1, dtype=torch.float32, device=device)


@triton.jit
def _split_chunk(
    L, NSPLIT: tl.constexpr, BLOCK_N: tl.constexpr, MIN_CHUNK: tl.constexpr
):
    per = tl.cdiv(L, NSPLIT)
    per = tl.cdiv(per, BLOCK_N) * BLOCK_N
    return tl.maximum(per, MIN_CHUNK)


# Tuning constants (gfx950, 256 CUs).
_TARGET_CTAS = 256  # bs * NSPLIT ~ one CTA per CU (1 CTA/CU: ~480 VGPRs, 131 KB LDS)
_MAX_SPLITS = 256
_MIN_CHUNK = 256  # min KV tokens per split


def default_num_splits(bs: int) -> int:
    """Power-of-two splits per request so that bs * NSPLIT ~ _TARGET_CTAS."""
    n = max(1, -(-_TARGET_CTAS // bs))
    p = 1
    while p < n:
        p *= 2
    return min(p, _MAX_SPLITS)
