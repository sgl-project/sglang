"""CUDA-JIT vectorized per-row scale (the apply_log_scaling_tau contract):
``out[row, :] = bf16(fp32(x[row, :]) * tau[row])``. See
csrc/tml/inkling_row_scale.cuh; the scalar triton kernel remains the fallback
for non-bf16 / unaligned inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_row_scale_module(use_pdl: bool) -> Module:
    args = make_cpp_args(use_pdl)
    return load_jit(
        "inkling_row_scale",
        *args,
        cuda_files=["inkling/inkling_row_scale.cuh"],
        cuda_wrappers=[
            ("run", f"row_scale<{args}>"),
            ("run_compact", f"row_compact<{args}>"),
        ],
    )


def row_scale_bf16(
    x: torch.Tensor, tau: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """``x``: [rows, inner] bf16, possibly row-strided (inner contiguous,
    inner % 8 == 0, 16B-aligned rows); ``tau``: fp32 [rows]. Returns a fresh
    contiguous scaled tensor (bit-identical to the triton kernel's output)."""
    if out is None:
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    module = _jit_row_scale_module(is_arch_support_pdl())
    module.run(x, tau, out)
    return out


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
