"""One-shot handoff of an already-quantized static per-tensor FP8 activation.

A producer that wrote both the bf16 activation and its static FP8 quantization (bit-identical to
``static_quant_fp8``) offers the FP8 tensor here; the next ``static_quant_fp8`` call with that exact
scale tensor takes it if it quantizes that exact bf16 tensor, and drops it otherwise. At most one
offer is pending: a new offer, or ``clear``, drops the previous one.
"""

from __future__ import annotations

from typing import Optional

import torch

_pending: Optional[tuple[int, int, torch.Size, torch.Tensor]] = None
# Host-side counts (offers made, offers taken), for engagement checks.
_counts = {"offered": 0, "taken": 0}


def stats() -> dict:
    return dict(_counts)


def offer(bf16: torch.Tensor, scale: torch.Tensor, fp8: torch.Tensor) -> None:
    global _pending
    _pending = (bf16.data_ptr(), scale.data_ptr(), bf16.shape, fp8)
    _counts["offered"] += 1


def clear() -> None:
    global _pending
    _pending = None


def take(x: torch.Tensor, scale: torch.Tensor) -> Optional[torch.Tensor]:
    global _pending
    if _pending is None or _pending[1] != scale.data_ptr():
        return None
    ptr, _, shape, fp8 = _pending
    _pending = None
    if ptr != x.data_ptr() or shape != x.shape:
        return None
    _counts["taken"] += 1
    return fp8
