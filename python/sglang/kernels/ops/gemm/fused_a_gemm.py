"""Unified entry point for the DeepSeek-V3 fused QKV-A GEMM.

Dispatches to one of two interchangeable implementations via ``backend``:

- ``"jit"``: runtime-compiled CUDA C++ (``sglang.kernels.ops.gemm._jit_dsv3_fused_a_gemm``).
- ``"cutedsl"``: CuTe DSL (``sglang.kernels.ops.gemm.cutedsl_dsv3_fused_a_gemm``).
- ``"auto"``: CuTe DSL on SM120+, otherwise the JIT kernel.

All backends share the signature ``(mat_a, mat_b, output=None) -> Tensor`` with
``mat_a`` row-major ``[M, K]`` (M in [1, 16], bf16) and ``mat_b`` the column-major
weight ``[K, N]`` (``weight.T``).
"""

from enum import Enum

import torch

from sglang.srt.utils.common import get_device_sm, is_cuda, is_sm120_supported


class FusedAGemmBackend(str, Enum):
    AUTO = "auto"
    JIT = "jit"
    CUTEDSL = "cutedsl"


_AUTO_BACKEND = (
    FusedAGemmBackend.CUTEDSL if is_sm120_supported() else FusedAGemmBackend.JIT
)

_IS_CUDA = is_cuda()
_DEVICE_SM = get_device_sm()


def fused_a_gemm_weight_eligible(layer: torch.nn.Module) -> bool:
    return (
        layer.weight.dtype == torch.bfloat16
        and layer.weight.shape[0] % 16 == 0
        and layer.weight.shape[1] % 256 == 0
        and _IS_CUDA
        and _DEVICE_SM >= 90
    )


def linear_with_fused_a_gemm(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    *,
    backend: "FusedAGemmBackend | str" = FusedAGemmBackend.AUTO,
) -> torch.Tensor:
    # LoRA reads weight.T directly, bypassing the adapter, so fall back when active.
    if (
        not isinstance(hidden_states, tuple)
        and 1 <= hidden_states.shape[0] <= 16
        and not getattr(layer, "set_lora", False)
    ):
        return dsv3_fused_a_gemm(hidden_states, layer.weight.T, backend=backend)
    return layer(hidden_states)[0]


def dsv3_fused_a_gemm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    output: torch.Tensor | None = None,
    backend: FusedAGemmBackend | str = FusedAGemmBackend.AUTO,
) -> torch.Tensor:
    backend = FusedAGemmBackend(backend)
    if backend == FusedAGemmBackend.AUTO:
        backend = _AUTO_BACKEND

    if backend == FusedAGemmBackend.JIT:
        from sglang.kernels.ops.gemm._jit_dsv3_fused_a_gemm import (
            dsv3_fused_a_gemm as impl,
        )
    else:
        from sglang.kernels.ops.gemm.cutedsl_dsv3_fused_a_gemm import (
            dsv3_fused_a_gemm as impl,
        )
    return impl(mat_a, mat_b, output)
