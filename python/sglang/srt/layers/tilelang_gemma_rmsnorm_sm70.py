# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Mixed-dtype Gemma residual/RMSNorm fusion for Volta.

This is the post-TP-reduction contract used by the large Qwen3.5/Qwen3.8
prefill path: add an FP16 activation into an FP32 residual, keep the updated
residual in FP32, reduce its variance in FP32, apply Gemma's ``weight + 1``,
and overwrite the activation with FP16 normalized output.
"""

import os
from functools import lru_cache

import tilelang
import tilelang.language as T
import torch

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}
if hasattr(tilelang.PassConfigKey, "TL_DISABLE_FAST_MATH"):
    _PASS_CONFIGS[tilelang.PassConfigKey.TL_DISABLE_FAST_MATH] = True
elif hasattr(tilelang.PassConfigKey, "TL_ENABLE_FAST_MATH"):
    _PASS_CONFIGS[tilelang.PassConfigKey.TL_ENABLE_FAST_MATH] = False

_THREADS = 256
_MIN_ROWS = 256
_EXECUTION_BACKEND = "cython"


def _enabled() -> bool:
    value = os.environ.get("SGLANG_V100_GEMMA_RMSNORM", "1").strip().lower()
    if value not in ("0", "false", "off", "no", "1", "true", "on", "yes"):
        raise ValueError("SGLANG_V100_GEMMA_RMSNORM must be a boolean value")
    return value in ("1", "true", "on", "yes")


@tilelang.jit(
    execution_backend=_EXECUTION_BACKEND,
    pass_configs=_PASS_CONFIGS,
)
def _gemma_fused_add_rmsnorm_kernel(hidden_size: int, weight_fp32: bool):
    rows = T.dynamic("rows")
    weight_dtype = T.float32 if weight_fp32 else T.float16

    @T.prim_func
    def main(
        X: T.Tensor([rows, hidden_size], T.float16),
        Residual: T.Tensor([rows, hidden_size], T.float32),
        Weight: T.Tensor([hidden_size], weight_dtype),
        Epsilon: T.float32,
    ):
        with T.Kernel(rows, threads=_THREADS) as row:
            values = T.alloc_fragment([hidden_size], T.float32)
            squares = T.alloc_fragment([hidden_size], T.float32)
            square_sum = T.alloc_fragment([1], T.float32)

            # Each thread owns hidden_size / 256 values (20 for D=5120). Keep
            # them register-resident across the collective reduction so the
            # newly written FP32 residual is not read from global memory again.
            for col in T.Parallel(hidden_size):
                values[col] = T.cast(X[row, col], T.float32) + Residual[row, col]
                squares[col] = values[col] * values[col]
                Residual[row, col] = values[col]

            T.reduce_sum(squares, square_sum, dim=0)
            inverse_rms = T.rsqrt(square_sum[0] / hidden_size + Epsilon)
            for col in T.Parallel(hidden_size):
                X[row, col] = T.cast(
                    values[col] * inverse_rms * (T.cast(Weight[col], T.float32) + 1.0),
                    T.float16,
                )

    return main


@lru_cache(maxsize=None)
def _get_kernel(hidden_size: int, weight_fp32: bool):
    return _gemma_fused_add_rmsnorm_kernel(hidden_size, weight_fp32)


def can_use_gemma_fused_add_rmsnorm_sm70(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    post_residual_addition: torch.Tensor | None,
) -> bool:
    return (
        _enabled()
        and x.device.type == "cuda"
        and torch.cuda.get_device_capability(x.device) == (7, 0)
        and x.ndim == 2
        and x.shape[0] >= _MIN_ROWS
        and x.shape[1] == 5120
        and x.dtype == torch.float16
        and x.is_contiguous()
        and residual is not None
        and residual.shape == x.shape
        and residual.dtype == torch.float32
        and residual.is_contiguous()
        and post_residual_addition is None
        and weight.shape == (x.shape[1],)
        and weight.dtype in (torch.float16, torch.float32)
        and weight.is_contiguous()
    )


def gemma_fused_add_rmsnorm_sm70(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update ``x`` and ``residual`` in place and return their original views."""
    kernel = _get_kernel(x.shape[1], weight.dtype == torch.float32)
    kernel(x, residual, weight, float(epsilon))
    return x, residual
