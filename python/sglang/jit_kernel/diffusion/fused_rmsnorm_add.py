from __future__ import annotations

import functools
import os
from typing import TYPE_CHECKING, Optional

import flashinfer
import torch

from sglang.jit_kernel.utils import load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@functools.cache
def _jit_rmsnorm_add_module() -> Module:
    kernel_name = "fused_rmsnorm_add"
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


def fused_rmsnorm_add(
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: Optional[torch.Tensor],
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Fused: RMSNorm(residual, gamma) + x -> output.
    This fuses: x = x + self.ffn_norm2(ffn_out)

    Expects:
      - x: [M, N], contiguous on last dim
      - residual: [M, N], contiguous on last dim
      - gamma: None or [N]
      - eps: float, default: 1e-6

    Returns:
      - output: [M, N], result (x + RMSNorm(residual))
    """
    output = torch.empty_like(x)
    module = _jit_rmsnorm_add_module()
    kernel = getattr(module, "fused_rmsnorm_add")
    kernel(output, x, residual, gamma, eps)
    return output
