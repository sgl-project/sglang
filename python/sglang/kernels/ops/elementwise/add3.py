"""CUDA JIT elementwise 3-way add: out = bf16(bf16(a + b) + c)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# The kernel vectorizes by device::kMaxVecBytes (16B pre-Blackwell, 32B on
# Blackwell+). Requiring divisibility by the widest case keeps covered()
# arch-independent and never looser than the compiled kernel's check.
_MAX_VEC_ELEMS: int = 16


@cache_once
def _jit_add3_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "add3_bf16",
        *args,
        cuda_files=["elementwise/add3.cuh"],
        cuda_wrappers=[("run", f"sglang::Add3Kernel<{args}>::launch")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )


def covered(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> bool:
    """Same-shape contiguous CUDA bf16 tensors, numel a multiple of the
    widest vector (16 elements)."""
    return (
        a.dtype == b.dtype == c.dtype == torch.bfloat16
        and a.shape == b.shape == c.shape
        and a.is_contiguous()
        and b.is_contiguous()
        and c.is_contiguous()
        and a.device.type == "cuda"
        and a.numel() > 0
        and a.numel() % _MAX_VEC_ELEMS == 0
    )


def add3(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    prefetch_bc: bool = False,
) -> torch.Tensor:
    """out = bf16(bf16(a + b) + c); double rounding matches the unfused add
    pair bit-for-bit. With prefetch_bc, b/c are loaded before the PDL wait —
    only safe when their producers are at least two kernels back."""
    if out is None:
        out = torch.empty_like(a)
    module = _jit_add3_module()
    module.run(a.view(-1), b.view(-1), c.view(-1), out.view(-1), prefetch_bc)
    return out
