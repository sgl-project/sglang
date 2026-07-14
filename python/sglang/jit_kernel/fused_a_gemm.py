"""Unified entry point for the DeepSeek-V3 fused QKV-A GEMM.

Dispatches to one of three interchangeable implementations via ``backend``:

- ``"aot"``: prebuilt ``sgl_kernel.dsv3_fused_a_gemm`` (CUDA C++).
- ``"jit"``: runtime-compiled CUDA C++ (``sglang.jit_kernel.dsv3_fused_a_gemm``).
- ``"cutedsl"``: CuTe DSL (``sglang.jit_kernel.cutedsl_dsv3_fused_a_gemm``).
- ``"auto"``: CuTe DSL on SM120+, otherwise the JIT kernel.

All backends share the signature ``(mat_a, mat_b, output=None) -> Tensor`` with
``mat_a`` row-major ``[M, K]`` (M in [1, 16], bf16) and ``mat_b`` the column-major
weight ``[K, N]`` (``weight.T``).
"""

from enum import Enum

import torch

from sglang.srt.utils.common import is_sm120_supported


class FusedAGemmBackend(str, Enum):
    AUTO = "auto"
    AOT = "aot"
    JIT = "jit"
    CUTEDSL = "cutedsl"


_AUTO_BACKEND = (
    FusedAGemmBackend.CUTEDSL if is_sm120_supported() else FusedAGemmBackend.JIT
)


def dsv3_fused_a_gemm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    output: torch.Tensor | None = None,
    backend: FusedAGemmBackend | str = FusedAGemmBackend.AUTO,
    tile_m: int = 16,
) -> torch.Tensor:
    backend = FusedAGemmBackend(backend)
    if backend == FusedAGemmBackend.AUTO:
        backend = _AUTO_BACKEND

    if backend == FusedAGemmBackend.AOT:
        from sgl_kernel import dsv3_fused_a_gemm as impl

        return impl(mat_a, mat_b, output)
    elif backend == FusedAGemmBackend.JIT:
        from sglang.jit_kernel.dsv3_fused_a_gemm import dsv3_fused_a_gemm as impl

        return impl(mat_a, mat_b, output, tile_m=tile_m)
    else:
        from sglang.jit_kernel.cutedsl_dsv3_fused_a_gemm import (
            dsv3_fused_a_gemm as impl,
        )

        return impl(mat_a, mat_b, output)
