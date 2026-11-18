"""MXFP8 (OCP 1x32 block, E8M0 scale) quantization along the last dim."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

MXFP8_BLOCK_SIZE = 32


@cache_once
def _jit_mxfp8_group_quant_module(dtype: torch.dtype) -> Module:
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError(
            f"Unsupported dtype {dtype}. Supported: float16, bfloat16, float32"
        )
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        "mxfp8_group_quant",
        *args,
        cuda_files=["gemm/mxfp8_group_quant.cuh"],
        cuda_wrappers=[("mxfp8_group_quant", f"mxfp8_group_quant<{args}>")],
    )


def mxfp8_group_quant(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a contiguous 2-D ``[M, K]`` tensor (``K % 32 == 0``) to MXFP8.

    Returns ``(q, scale)``: ``q`` is ``[M, K]`` ``float8_e4m3fn`` and ``scale`` is
    ``[M, K // 32]`` ``uint8`` biased E8M0 exponents in row-major order. The scale
    is ``2^ceil(log2(amax / 448))`` per block, bit-identical to
    ``triton_kernels.numerics_details.mxfp.downcast_to_mxfp(x, float8_e4m3fn,
    axis=1)`` with its default ROUND_UP mode.
    """
    m, k = x.shape
    q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty((m, k // MXFP8_BLOCK_SIZE), dtype=torch.uint8, device=x.device)
    _jit_mxfp8_group_quant_module(x.dtype).mxfp8_group_quant(x, q, scale)
    return q, scale
