from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    is_arch_support_pdl,
)

if TYPE_CHECKING:
    pass


from sglang.kernels.ops.elementwise.row_scale import _jit_row_scale_module


def row_compact_bf16(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Contiguous copy of row-strided ``x`` ([rows, inner] bf16, inner
    contiguous, inner % 8 == 0, 16B-aligned rows) -- the tau-less flavor of
    ``row_scale_bf16``. Beats the TensorIterator strided copy that einsum's
    reshape would otherwise run on such inputs."""
    if out is None:
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    module = _jit_row_scale_module(is_arch_support_pdl())
    module.run_compact(x, out)
    return out
