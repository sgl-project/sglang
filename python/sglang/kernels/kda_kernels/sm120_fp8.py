"""Adaptive SM12x dispatch for small-M static per-tensor FP8 linear."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch


_SUPPORTED_KDA_M = (2, 4, 8, 9)

# The streaming GEMV is faster for supported M=1 shapes, so the CUTLASS path
# keeps only the oversized M=1 projection that GEMV cannot serve. M>=2 shapes
# are enabled only after BF16 comparison, cold-L2 benchmarking, and model-level
# E2E validation on RTX PRO 6000 Blackwell.
_KDA_M_BY_PROJECTION = {
    (5120, 8192): _SUPPORTED_KDA_M,
    (5120, 16384): (2, 4, 8),
    (6144, 5120): _SUPPORTED_KDA_M,
    (4096, 4096): _SUPPORTED_KDA_M,
    (5120, 7168): _SUPPORTED_KDA_M,
    (5120, 5120): (2, 4, 8, 9),
    (5120, 34816): (1, 2, 4, 8, 9),
}
_KDA_PRODUCTION_SHAPES = frozenset(
    (m, k, n)
    for (k, n), supported_m in _KDA_M_BY_PROJECTION.items()
    for m in supported_m
)


@functools.cache
def _device_capability(device: torch.device) -> tuple[int, int]:
    import torch

    return torch.cuda.get_device_capability(device)


@functools.cache
def _has_kda_runtime() -> bool:
    try:
        from sglang.kernels.jit.utils.deps import get_cutlass_include_paths

        get_cutlass_include_paths()
        return True
    except (ModuleNotFoundError, RuntimeError):
        return False


def _supports_common(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: Optional[torch.Tensor],
    output_scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
) -> bool:
    import torch

    if (
        input.device.type != "cuda"
        or input.ndim != 2
        or input.dtype != torch.bfloat16
        or not input.is_contiguous()
        or input_scale is None
        or output_scale is None
        or bias is not None
    ):
        return False

    m, k = input.shape
    if weight.ndim != 2:
        return False
    n = weight.shape[1]
    if (
        weight.dtype != torch.float8_e4m3fn
        or weight.shape[0] != k
        or weight.stride() != (1, k)
    ):
        return False
    if (
        input_scale.dtype != torch.float32
        or input_scale.numel() != 1
        or not input_scale.is_contiguous()
        or output_scale.dtype != torch.float32
        or output_scale.numel() != 1
        or not output_scale.is_contiguous()
    ):
        return False
    if any(t.device != input.device for t in (weight, input_scale, output_scale)):
        return False
    return True


def try_sm120_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: Optional[torch.Tensor],
    output_scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor | None:
    """Run the best qualified SM12x FP8 small-M kernel, or return ``None``."""
    if not _supports_common(
        input,
        weight,
        input_scale,
        output_scale,
        bias,
    ):
        return None

    assert input_scale is not None
    assert output_scale is not None
    capability = _device_capability(input.device)
    if capability[0] != 12:
        return None

    m, k = input.shape
    n = weight.shape[1]
    use_native_gemv = False
    if m == 1:
        from sglang.kernels.ops.gemm.sm120_fp8_gemv import use_sm120_fp8_gemv

        use_native_gemv = use_sm120_fp8_gemv(m, n, k)

    use_kda_gemm = (
        not use_native_gemv
        and capability == (12, 0)
        and (m, k, n) in _KDA_PRODUCTION_SHAPES
        and _has_kda_runtime()
    )
    if not use_native_gemv and not use_kda_gemm:
        return None

    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8

    quantized, _ = static_quant_fp8(input, input_scale, repeat_scale=False)
    if use_native_gemv:
        from sglang.kernels.ops.gemm.sm120_fp8_gemv import sm120_fp8_gemv

        return sm120_fp8_gemv(quantized, weight.t(), output_scale.reshape(1))

    from sglang.kernels.kda_kernels.sm120_fp8_skinny_gemm_sm120 import (
        _run_sm120_fp8_skinny_gemm_quantized,
    )

    return _run_sm120_fp8_skinny_gemm_quantized(quantized, weight, output_scale)


__all__ = ["try_sm120_fp8_linear"]
