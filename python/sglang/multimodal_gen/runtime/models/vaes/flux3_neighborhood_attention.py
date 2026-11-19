# SPDX-License-Identifier: Apache-2.0
"""Neighborhood attention of the FLUX 3 video VAE without NATTEN.

NATTEN is an optional dependency; when it is missing, the VAE runs the same
windows through a compiled FlexAttention block mask. The windows follow
NATTEN's ``na2d`` / ``na3d`` (stride 1, no dilation): on a non-causal axis the
window of query ``i`` is ``[start, start + k)`` with
``start = clamp(i - k // 2, 0, L - k)`` (shifted inward at the borders); on a
causal axis it is ``[max(0, i - k + 1), i]``.
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# One block mask per (grid, kernel, causal, device); a server sees a handful.
_BLOCK_MASK_CACHE_MAX = 16
_block_masks: dict[tuple, object] = {}


@lru_cache(maxsize=1)
def natten_available() -> bool:
    available = importlib.util.find_spec("natten") is not None
    if not available:
        logger.warning(
            "NATTEN is not installed; the FLUX 3 video VAE uses the FlexAttention "
            "fallback, compiled on the first request. Install a wheel matching "
            "your torch/CUDA build from https://whl.natten.org/ to use NATTEN."
        )
    return available


@lru_cache(maxsize=1)
def _compiled_flex_attention():
    from torch.nn.attention.flex_attention import flex_attention

    # Uncompiled, flex_attention materializes the dense score matrix.
    return torch.compile(flex_attention, dynamic=False)


def _neighborhood_block_mask(
    grid: tuple[int, ...],
    kernel: tuple[int, ...],
    causal: tuple[bool, ...],
    device: torch.device,
):
    from torch.nn.attention.flex_attention import create_block_mask

    key = (grid, kernel, causal, str(device))
    mask = _block_masks.get(key)
    if mask is not None:
        return mask
    kernel = tuple(min(k, n) for k, n in zip(kernel, grid))
    strides = [1] * len(grid)
    for axis in range(len(grid) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * grid[axis + 1]

    def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
        inside = None
        for n, k, is_causal, stride in zip(grid, kernel, causal, strides):
            q_pos = (q_idx // stride) % n
            k_pos = (kv_idx // stride) % n
            if is_causal:
                axis_ok = (k_pos <= q_pos) & (k_pos > q_pos - k)
            else:
                start = torch.clamp(q_pos - k // 2, 0, n - k)
                axis_ok = (k_pos >= start) & (k_pos < start + k)
            inside = axis_ok if inside is None else inside & axis_ok
        return inside

    seq_len = 1
    for n in grid:
        seq_len *= n
    # _compile=True keeps the mask construction sparse (eager is O(S^2) memory).
    mask = create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
        _compile=True,
    )
    if len(_block_masks) >= _BLOCK_MASK_CACHE_MAX:
        _block_masks.pop(next(iter(_block_masks)))
    _block_masks[key] = mask
    return mask


def neighborhood_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    kernel_size: list[int],
    is_causal: list[bool] | None = None,
) -> torch.Tensor:
    """NATTEN-layout ``(B, *grid, heads, head_dim)`` in and out, like ``na2d`` / ``na3d``."""
    batch, *grid, heads, head_dim = q.shape
    causal = tuple(is_causal or [False] * len(grid))
    mask = _neighborhood_block_mask(tuple(grid), tuple(kernel_size), causal, q.device)

    def to_flex(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(batch, -1, heads, head_dim).transpose(1, 2)

    out = _compiled_flex_attention()(
        to_flex(q), to_flex(k), to_flex(v), block_mask=mask
    )
    return out.transpose(1, 2).reshape(q.shape)
