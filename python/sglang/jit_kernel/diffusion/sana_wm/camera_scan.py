# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional

import torch

from sglang.jit_kernel.diffusion.sana_wm.fused_gdn_chunkwise import (
    cam_scan_bidi_chunkwise,
)


def sana_wm_cam_scan_bidi_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    dot_precision: Optional[int] = None,
) -> torch.Tensor:
    """SGLang wrapper for Sana's full phase-A/B/C camera scan port."""
    return cam_scan_bidi_chunkwise(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        beta.contiguous(),
        decay.contiguous(),
        dot_precision=dot_precision,
    )


def can_use_sana_wm_cam_scan_bidi_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
) -> bool:
    if q.shape != k.shape or q.shape != v.shape:
        return False
    if q.dim() != 4 or not q.is_cuda:
        return False
    if q.dtype != torch.float32:
        return False
    B, H, D, N = q.shape
    if beta.ndim != 4 or decay.ndim != 3:
        return False
    T = beta.shape[2]
    if T <= 0 or N % T != 0:
        return False
    S = N // T
    return tuple(beta.shape) == (B, H, T, S) and tuple(decay.shape) == (B, H, T)
