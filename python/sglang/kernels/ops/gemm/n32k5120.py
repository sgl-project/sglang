"""Decode-batch bf16 GEMM for ``x[m, 5120] @ w[32, 5120].T``.

Each CTA prefetches one weight row before the PDL wait. Row groups map to
blockIdx.y, keeping per-CTA work independent of batch size.
The thread-to-K mapping and shared dot_product reduction match tiny_gemm_bf16,
so results agree bitwise where both apply and are invariant to batch composition.
cuBLAS uses a different reduction order. See ``gemm/small/n32k5120.cuh``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    get_jit_cuda_arch,
    is_arch_support_pdl,
    is_hip_runtime,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

N: int = 32
K: int = 5120
# Decode batches only: above this the caller keeps the general GEMM. The kernel
# itself accepts any m (rows are split over blockIdx.y), this is a policy cap.
MAX_M: int = 32


@cache_once
def _jit_n32k5120_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "n32k5120_gemm",
        *args,
        cuda_files=["gemm/small/n32k5120.cuh"],
        cuda_wrappers=[("run", f"N32K5120Kernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


@register_custom_op(op_name="n32k5120_gemm_bf16", mutates_args=["out"])
def _n32k5120_gemm_custom_op(
    x: torch.Tensor, w: torch.Tensor, out: torch.Tensor
) -> None:
    _jit_n32k5120_module().run(x, w, out)


@cache_once
def _arch_supported() -> bool:
    return not is_hip_runtime() and get_jit_cuda_arch().major >= 9


def can_use_n32k5120_gemm(n: int, k: int, m: int, max_m: int = MAX_M) -> bool:
    """Whether :func:`n32k5120_gemm_bf16` serves ``[m, k] @ [n, k].T``.
    Callers fall back to a general GEMM when this is False."""
    return n == N and k == K and 1 <= m <= max_m and _arch_supported()


@debug_kernel_api
def n32k5120_gemm_bf16(
    x: torch.Tensor,
    w: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Equal to ``torch.nn.functional.linear(x, w)`` for ``w`` of shape [32, 5120].

    Call :func:`can_use_n32k5120_gemm` first: other shapes raise rather than fall
    back. ``x`` may be a row-sliced view (``x.stride(1) == 1``), but each row must
    start on a 32-byte boundary because rows are loaded as whole vectors.

    :param x: Shape [m, 5120], bf16.
    :param w: Shape [32, 5120], bf16, contiguous.
    :param out: Optional [m, 32] bf16 buffer to write into.
    """
    m = x.shape[0]
    if out is None:
        out = torch.empty((m, N), dtype=torch.bfloat16, device=x.device)
    if m == 0:
        return out
    _n32k5120_gemm_custom_op(x, w, out)
    return out
