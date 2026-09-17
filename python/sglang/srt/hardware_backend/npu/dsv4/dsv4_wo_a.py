"""BF16 wo_a linear path for a single NPU-local output group."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def apply_npu_wo_a_bf16(o: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Map [T, 1, D] to [T, 1, R] using the original [R, D] weight.

    With one local group (DSV4 Flash TP8), the grouped contraction is a linear
    projection. F.linear outperformed both einsum and a prepacked BMM in the
    910B4 BS1/BS8 microbenchmark. Keep the original Parameter storage so online
    copy_ updates and graph replay share the same weight without a layout cache.
    """
    return F.linear(o[:, 0, :], weight).unsqueeze(1)
