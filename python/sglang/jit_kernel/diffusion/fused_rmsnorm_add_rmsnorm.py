from __future__ import annotations

import functools
import os
from typing import TYPE_CHECKING, Optional, Tuple

import flashinfer
import torch

from sglang.jit_kernel.utils import load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@functools.cache
def _jit_rmsnorm_add_rmsnorm_module() -> Module:
    kernel_name = "fused_rmsnorm_add_rmsnorm"
    marker = kernel_name
    export_name = kernel_name
    # TODO: workaround, do not import cutlass from flashinfer
    cutlass_include = os.path.join(
        os.path.dirname(flashinfer.__file__), "data", "cutlass", "include"
    )

    return load_jit(
        marker,
        cuda_files=["diffusion/fused_norm_scale_shift.cuh"],
        cuda_wrappers=[(export_name, kernel_name)],
        extra_include_paths=[cutlass_include],
    )


def fused_rmsnorm_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma1: Optional[torch.Tensor],
    gamma2: Optional[torch.Tensor],
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused: RMSNorm(residual, gamma1) + x -> output1, then RMSNorm(output1, gamma2) -> output2.
    This fuses: x = x + self.attention_norm2(attn_out); self.ffn_norm1(x)

    Expects:
      - x: [M, N], contiguous on last dim
      - residual: [M, N], contiguous on last dim
      - gamma1: None or [N]
      - gamma2: None or [N]
      - eps: float, default: 1e-6

    Returns:
      - output1: [M, N], intermediate result (x + RMSNorm(residual))
      - output2: [M, N], final result (RMSNorm(output1))
    """
    output1 = torch.empty_like(x)
    output2 = torch.empty_like(x)
    module = _jit_rmsnorm_add_rmsnorm_module()
    kernel = getattr(module, "fused_rmsnorm_add_rmsnorm")
    kernel(output1, output2, x, residual, gamma1, gamma2, eps)
    return output1, output2
