"""Decode-batch bf16 GEMM for x[m, 512] @ w[128, 512].T.

Shares dot_product's reduction order with tiny_gemm_bf16 for bitwise parity;
a row's result does not depend on the batch composition.
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

N: int = 128
K: int = 512
# Decode batches only: above this the caller keeps the general GEMM. The kernel
# itself accepts any m (rows are split over blockIdx.y), this is a policy cap.
MAX_M: int = 32


@cache_once
def _jit_n128k512_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "n128k512_gemm",
        *args,
        cuda_files=["gemm/small/n128k512.cuh"],
        cuda_wrappers=[("run", f"N128K512Kernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


@register_custom_op(op_name="n128k512_gemm_bf16", mutates_args=["out"])
def _n128k512_gemm_custom_op(
    x: torch.Tensor, w: torch.Tensor, out: torch.Tensor
) -> None:
    _jit_n128k512_module().run(x, w, out)


@cache_once
def _arch_supported() -> bool:
    return not is_hip_runtime() and get_jit_cuda_arch().major >= 9


def can_use_n128k512_gemm(n: int, k: int, m: int, max_m: int = MAX_M) -> bool:
    """Whether :func:`n128k512_gemm_bf16` serves ``[m, k] @ [n, k].T``.
    Callers fall back to a general GEMM when this is False."""
    return n == N and k == K and 1 <= m <= max_m and _arch_supported()


@debug_kernel_api
def n128k512_gemm_bf16(
    x: torch.Tensor,
    w: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Equal to ``torch.nn.functional.linear(x, w)`` for ``w`` of shape [128, 512].

    Call :func:`can_use_n128k512_gemm` first: other shapes raise rather than fall
    back. ``x`` may be a row-sliced view (``x.stride(1) == 1``), but each row must
    start on a 32-byte boundary because rows are loaded as whole vectors.

    :param x: Shape [m, 512], bf16.
    :param w: Shape [128, 512], bf16, contiguous.
    :param out: Optional [m, 128] bf16 buffer to write into.
    """
    m = x.shape[0]
    if out is None:
        out = torch.empty((m, N), dtype=torch.bfloat16, device=x.device)
    if m == 0:
        return out
    _n128k512_gemm_custom_op(x, w, out)
    return out
