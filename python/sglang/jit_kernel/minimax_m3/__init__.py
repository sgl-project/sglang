# SPDX-License-Identifier: Apache-2.0
"""Fused Triton kernels for MiniMax-M3 on AMD ROCm (gfx94x / gfx95x).

Model-scoped JIT kernels (mirrors ``jit_kernel/dsv4``), split by op type:
  * ``rmsnorm`` -- fused fp32 Gemma RMSNorm (plain + fused-add-residual)
  * ``swiglu``  -- fused fp32 SwiGLU-OAI (split layout)
"""

from sglang.jit_kernel.minimax_m3.rmsnorm import (
    _num_warps,
    gemma_fused_add_rmsnorm,
    gemma_rmsnorm,
)
from sglang.jit_kernel.minimax_m3.swiglu import (
    swiglu_oai_mxfp8_quant,
    swiglu_oai_split,
)

__all__ = [
    "gemma_rmsnorm",
    "gemma_fused_add_rmsnorm",
    "swiglu_oai_split",
    "swiglu_oai_mxfp8_quant",
    "_num_warps",
]
