# SPDX-License-Identifier: Apache-2.0
"""Small-M (at most 16 rows: decode, a speculative verify or draft block) linears on
gfx1151.

``small_m_linear`` is the one entry point the generic layers call. It returns None
wherever the gfx1151 kernels do not apply, so callers keep their existing path.
"""

from typing import Optional

import torch

from sglang.srt.utils import use_gfx1151_kernels


def small_m_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> Optional[torch.Tensor]:
    """``x @ weight.T (+ bias)`` through the gfx1151 bf16 small-M GEMM, or None when it
    does not take these operands."""
    if not use_gfx1151_kernels() or x.dim() != 2 or not 0 < x.shape[0] <= 16:
        return None
    from sglang.kernels.ops.gemm import bf16_small_m_gfx1151 as bf

    x = x.contiguous()
    if not bf.covered(x, weight):
        return None
    out = bf.bf16_small_m_gemm(x, weight)
    return out if bias is None else out + bias
