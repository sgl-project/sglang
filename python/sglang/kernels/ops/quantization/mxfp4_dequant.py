"""MXFP4 (packed e2m1 + 1x32 E8M0 scale) to bf16 dequantization."""

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

MXFP4_BLOCK_SIZE = 32


@cache_once
def _jit_mxfp4_dequant_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "mxfp4_dequant",
        *args,
        cuda_files=["gemm/mxfp4_dequant.cuh"],
        cuda_wrappers=[("mxfp4_dequant", f"mxfp4_dequant<{args}>")],
    )


def mxfp4_dequant(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize MXFP4 weights to bf16 along the last dim.

    ``packed`` is ``[..., K // 2]`` uint8 (two e2m1 values per byte, low nibble
    first) and ``scale`` is ``[..., K // 32]`` uint8 E8M0; both contiguous.
    Returns ``[..., K]`` bf16, bit-identical to
    ``triton_kernels.numerics_details.mxfp.upcast_from_mxfp(packed, scale,
    torch.bfloat16, axis=-1)``.
    """
    *lead, packed_k = packed.shape
    k = packed_k * 2
    out = torch.empty((*lead, k), dtype=torch.bfloat16, device=packed.device)
    _jit_mxfp4_dequant_module().mxfp4_dequant(
        packed.view(-1, packed_k),
        scale.view(-1, k // MXFP4_BLOCK_SIZE),
        out.view(-1, k),
    )
    return out
