"""Unified entry point for the DeepSeek-V3 fused QKV-A GEMM.

Dispatches to one of two interchangeable implementations via ``backend``:

- ``"jit"``: runtime-compiled CUDA C++ (``sglang.kernels.ops.gemm.dsv3_fused_a_gemm``).
- ``"cutedsl"``: CuTe DSL (``sglang.kernels.ops.gemm.cutedsl_dsv3_fused_a_gemm``).
- ``"auto"``: CuTe DSL on SM120+, otherwise the JIT kernel.

All backends share the signature ``(mat_a, mat_b, output=None) -> Tensor`` with
``mat_a`` row-major ``[M, K]`` (M in [1, 16], bf16) and ``mat_b`` the column-major
weight ``[K, N]`` (``weight.T``).

On SM103, the block-FP8 projections in ``_FP8_SMALL_M_SHAPES`` are served by the JIT
W8A16 kernel (row-major weights, BF16 activations): ``Fp8LinearMethod`` checks
``fp8_small_m_eligible`` once after weight loading and calls ``fp8_small_m_linear``
from ``apply``. Other layers keep their normal dispatch, and so does every layer when
``--fp8-gemm-backend`` is set explicitly or ``SGLANG_ENABLE_FP8_SMALL_M_W8A16=0``.
"""

from enum import Enum

import torch

from sglang.srt.environ import envs
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
# The kernel's limit: two 8-token MMA tiles.
_FP8_MAX_TOKENS = 16
# (N, K) of the block-FP8 self_attn projections served by the W8A16 kernel (GLM-5.2 TP4 shards, incl. indexer wq_b).
_FP8_SMALL_M_SHAPES = frozenset(
    {
        (2624, 6144),  # fused qkv_a
        (4096, 2048),  # q_b and indexer wq_b
        (6144, 4096),  # o_proj
    }
)


def fused_a_gemm_weight_eligible(layer: torch.nn.Module) -> bool:
    return (
        layer.weight.dtype == torch.bfloat16
        and layer.weight.shape[0] % 16 == 0
        and layer.weight.shape[1] % 256 == 0
        and _IS_CUDA
        and _DEVICE_SM >= 90
    )


def fp8_small_m_eligible(
    layer: torch.nn.Module, weight_block_size: list[int], block_fp8_as_mxfp8: bool
) -> bool:
    # SM103 (GB300) only, the one architecture the kernel is validated on; MXFP8-view layers keep their own kernel.
    if (
        not _IS_CUDA
        or _DEVICE_SM != 103
        or weight_block_size != [128, 128]
        or block_fp8_as_mxfp8
        or layer.bias is not None
    ):
        return False
    from sglang.srt.layers.quantization.fp8_utils import get_fp8_gemm_runner_backend
    from sglang.srt.runtime_context import get_exec

    # An explicit --fp8-gemm-backend or the kill switch keeps the selected W8A8 backend for every M.
    if (
        not envs.SGLANG_ENABLE_FP8_SMALL_M_W8A16.get()
        or not get_fp8_gemm_runner_backend().is_auto()
    ):
        return False

    weight, scale = layer.weight, layer.weight_scale_inv
    n, k = weight.shape
    if (
        (n, k) not in _FP8_SMALL_M_SHAPES
        or get_exec().deterministic.enable_deterministic_inference
        or weight.dtype != torch.float8_e4m3fn
        or not weight.is_contiguous()
    ):
        return False
    if scale.dtype == torch.float32:
        return tuple(scale.shape) == ((n + 127) // 128, k // 128)
    return scale.dtype == torch.int32 and tuple(scale.shape) == (n, (k + 511) // 512)


def fp8_small_m_linear(layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor | None:
    if (
        x.ndim != 2
        or not 1 <= x.shape[0] <= _FP8_MAX_TOKENS
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
    ):
        return None
    from sglang.kernels.ops.gemm.dsv3_fused_a_gemm import dsv3_fused_a_gemm_fp8

    return dsv3_fused_a_gemm_fp8(
        input=x, weight=layer.weight, weight_scale=layer.weight_scale_inv
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
        from sglang.kernels.ops.gemm.dsv3_fused_a_gemm import dsv3_fused_a_gemm as impl
    else:
        from sglang.kernels.ops.gemm.cutedsl_dsv3_fused_a_gemm import (
            dsv3_fused_a_gemm as impl,
        )
    return impl(mat_a, mat_b, output)
