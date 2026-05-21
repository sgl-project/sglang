from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


def _resolve_swa_lut(
    lut: Optional[torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, int, bool]:
    """Return the (tensor, length, has_lut) triple to launch the plan kernel with.

    Triton requires a valid tensor pointer at every kernel-arg slot even when ``HAS_SWA_LUT`` is False, so
    when the caller passes ``None`` we substitute a one-element sentinel tensor and set ``lut_len=0``;
    the kernel's constexpr branch guarantees no dereference happens. Dtype matches the production LUT
    (int64) so Triton ``tl.load`` element typing stays consistent.
    """
    if lut is not None:
        return lut, int(lut.shape[0]), True
    return torch.zeros(1, dtype=torch.int64, device=device), 0, False


@triton.jit
def _compute_window_start(prefix_lens, SWA_WINDOW: tl.constexpr):
    """Per-req window start: max(prefix_lens - SWA_WINDOW, 0) when SWA, else 0.
    Works for tile and scalar inputs (broadcasts via prefix_lens shape).
    """
    if SWA_WINDOW > 0:
        clipped = prefix_lens - SWA_WINDOW
        return tl.where(clipped > 0, clipped, 0)
    else:
        return prefix_lens - prefix_lens


@triton.jit
def _swa_translate_tile(raw, mask, lut_ptr, lut_len):
    """SWA-translate a tile of slot indices. Sentinels (raw < 0) are passed through unchanged.

    ``lut_len`` is the LUT's length (Python int from the host wrapper); when 0 the LUT is unused (the caller
    will only enter this branch when HAS_SWA_LUT is True, so lut_len is always > 0 in practice).
    """
    sentinel = raw < 0
    safe = tl.where(sentinel, 0, raw)
    if lut_len > 0:
        safe = tl.where(safe >= lut_len, lut_len - 1, safe)
    xlat = tl.load(lut_ptr + safe, mask=mask & (~sentinel), other=0)
    return tl.where(sentinel, raw, xlat)
