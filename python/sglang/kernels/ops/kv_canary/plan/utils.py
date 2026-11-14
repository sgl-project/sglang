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


def _require_dtype(tensor: torch.Tensor, name: str, dtype: torch.dtype) -> None:
    if tensor.dtype != dtype:
        raise ValueError(
            f"kv-canary: {name} must have dtype {dtype}, got {tensor.dtype}"
        )


def _require_1d(tensor: torch.Tensor, name: str) -> None:
    if tensor.ndim != 1:
        raise ValueError(
            f"kv-canary: {name} must be 1-D, got shape {tuple(tensor.shape)}"
        )


def _require_2d(tensor: torch.Tensor, name: str) -> None:
    if tensor.ndim != 2:
        raise ValueError(
            f"kv-canary: {name} must be 2-D, got shape {tuple(tensor.shape)}"
        )


def _require_len(tensor: torch.Tensor, name: str, expected: int) -> None:
    _require_1d(tensor=tensor, name=name)
    actual = int(tensor.shape[0])
    if actual != expected:
        raise ValueError(f"kv-canary: {name} length must be {expected}, got {actual}")


def _require_min_len(tensor: torch.Tensor, name: str, minimum: int) -> None:
    _require_1d(tensor=tensor, name=name)
    actual = int(tensor.shape[0])
    if actual < minimum:
        raise ValueError(f"kv-canary: {name} length must be >= {minimum}, got {actual}")


def _require_same_device(
    reference: torch.Tensor,
    reference_name: str,
    tensors: tuple[tuple[torch.Tensor, str], ...],
) -> None:
    for tensor, name in tensors:
        if tensor.device != reference.device:
            raise ValueError(
                f"kv-canary: {name} must be on {reference_name}'s device "
                f"{reference.device}, got {tensor.device}"
            )


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
