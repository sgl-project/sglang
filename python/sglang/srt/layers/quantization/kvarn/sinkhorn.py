# SPDX-License-Identifier: Apache-2.0
"""Log-domain iterative variance-normalization for KVarN (pure PyTorch).

No framework dependencies so it can be unit-tested
in isolation.
"""

from __future__ import annotations

import torch

_DEFAULT_ITERATIONS = 16
_CLIP_STD_MIN = 1e-3
_CLIP_STD_MAX = 1e3
_LOG_S_MIN = -0.3
_LOG_S_MAX = 10.0


def _imbalance(tile: torch.Tensor) -> torch.Tensor:
    """Sum of column-std spread and row-std spread.

    Lower is better; a perfectly balanced tile has ``imbalance == 2``.
    """
    sc = tile.std(dim=-2)  # along rows ⇒ per-column std
    sr = tile.std(dim=-1)  # along cols ⇒ per-row std
    return sc.amax(dim=-1) / sc.amin(dim=-1).clamp_min(1e-8) + sr.amax(
        dim=-1
    ) / sr.amin(dim=-1).clamp_min(1e-8)


def variance_normalize(
    tile: torch.Tensor,
    iterations: int = _DEFAULT_ITERATIONS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-tile log-domain Sinkhorn balancing.

    Args:
        tile: [R, C] fp32 (or any real dtype; cast to fp32 internally).
        iterations: number of alternating col/row passes (default 16).

    Returns:
        balanced [R, C] fp32
        s_col    [1, C] fp32
        s_row    [R, 1] fp32
    """
    m = tile.float()
    R, C = m.shape
    dev = m.device

    log_s_col = torch.zeros(1, C, device=dev)
    log_s_row = torch.zeros(R, 1, device=dev)

    cur = m / log_s_col.exp() / log_s_row.exp()
    imb_best = _imbalance(cur)
    sc_best = log_s_col.exp().clone()
    sr_best = log_s_row.exp().clone()

    for _ in range(iterations):
        col_std = cur.std(dim=0, keepdim=True).clamp(_CLIP_STD_MIN, _CLIP_STD_MAX)
        log_s_col = (log_s_col + col_std.log()).clip(_LOG_S_MIN, _LOG_S_MAX)
        cur = m / log_s_col.exp() / log_s_row.exp()

        row_std = cur.std(dim=1, keepdim=True).clamp(_CLIP_STD_MIN, _CLIP_STD_MAX)
        log_s_row = (log_s_row + row_std.log()).clip(_LOG_S_MIN, _LOG_S_MAX)
        cur = m / log_s_col.exp() / log_s_row.exp()

        imb = _imbalance(cur)
        if imb <= imb_best:
            imb_best = imb
            sc_best = log_s_col.exp().clone()
            sr_best = log_s_row.exp().clone()

    balanced = m / sc_best / sr_best
    return balanced, sc_best, sr_best


def variance_normalize_batched(
    tiles: torch.Tensor,
    iterations: int = _DEFAULT_ITERATIONS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched version: tiles is [N, R, C].

    Returns:
        balanced [N, R, C] fp32
        s_col    [N, 1, C] fp32
        s_row    [N, R, 1] fp32
    """
    m = tiles.float()
    N, R, C = m.shape
    dev = m.device

    log_s_col = torch.zeros(N, 1, C, device=dev)
    log_s_row = torch.zeros(N, R, 1, device=dev)

    cur = m / log_s_col.exp() / log_s_row.exp()
    imb_best = _imbalance(cur)  # [N]
    sc_best = log_s_col.exp().clone()
    sr_best = log_s_row.exp().clone()

    for _ in range(iterations):
        col_std = cur.std(dim=1, keepdim=True).clamp(_CLIP_STD_MIN, _CLIP_STD_MAX)
        log_s_col = (log_s_col + col_std.log()).clip(_LOG_S_MIN, _LOG_S_MAX)
        cur = m / log_s_col.exp() / log_s_row.exp()

        row_std = cur.std(dim=2, keepdim=True).clamp(_CLIP_STD_MIN, _CLIP_STD_MAX)
        log_s_row = (log_s_row + row_std.log()).clip(_LOG_S_MIN, _LOG_S_MAX)
        cur = m / log_s_col.exp() / log_s_row.exp()

        imb = _imbalance(cur)  # [N]
        better = imb <= imb_best
        if better.any():
            mask = better.view(N, 1, 1).to(log_s_col.dtype)
            sc_best = mask * log_s_col.exp() + (1 - mask) * sc_best
            sr_best = mask * log_s_row.exp() + (1 - mask) * sr_best
            imb_best = torch.where(better, imb, imb_best)

    balanced = m / sc_best / sr_best
    return balanced, sc_best, sr_best
