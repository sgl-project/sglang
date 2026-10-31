"""H3 residual + norm + per-token FP8 producers (SM100+).

Uses FP32 FMA and statistics, two-pass LayerNorm variance, and scale = amax/448.
Zero rows produce zero scale and codes. Results are close, not bit-exact, to
eager FP32 norm followed by FP8 quantization. Rows must be contiguous and
16-byte aligned; affine vectors stay FP32.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.ops.diffusion.common.platform import is_cuda_sm_at_least
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_MIN_SM = (10, 0)
_ALIGNMENT = 16
_ROW_DTYPES = (torch.float32, torch.float16)
# kBlockSize * kVecSize in minimax_h3_vae_residual_norm.cuh.
_WIDTH_MULTIPLE = 1024
# The row lives in registers (width / 256 floats per thread); 2048 is the
# verified MiniMax-H3 width, 4096 keeps the per-thread budget at 16 floats.
_MAX_WIDTH = 4096

_LAUNCHERS = {
    "rmsnorm_fp8": "RmsNormFp8Kernel",
    "residual_rmsnorm_fp8": "ResidualRmsNormFp8Kernel",
    "residual_layernorm": "ResidualLayerNormKernel",
}


@cache_once
def _jit_minimax_h3_vae_residual_norm_module(
    mode: str, dtype: torch.dtype, projected_dtype: torch.dtype | None, width: int
) -> Module:
    if mode not in _LAUNCHERS:
        raise RuntimeError(f"Unknown MiniMax-H3 residual norm mode {mode!r}")
    if dtype not in _ROW_DTYPES:
        raise RuntimeError(
            f"Unsupported row dtype {dtype}; expected float32 or float16"
        )
    if (projected_dtype is None) != (mode == "rmsnorm_fp8"):
        raise RuntimeError(f"mode {mode!r} takes a projected dtype iff residual")
    if projected_dtype is not None and projected_dtype not in _ROW_DTYPES:
        raise RuntimeError(
            f"Unsupported projected dtype {projected_dtype}; "
            "expected float32 or float16"
        )
    if width <= 0 or width % _WIDTH_MULTIPLE != 0 or width > _MAX_WIDTH:
        raise RuntimeError(
            f"Unsupported row width {width}; expected a multiple of "
            f"{_WIDTH_MULTIPLE} up to {_MAX_WIDTH}"
        )
    dtypes = (dtype,) if projected_dtype is None else (dtype, projected_dtype)
    args = make_cpp_args(*dtypes, width)
    launcher = f"minimax_h3_vae_residual_norm::{_LAUNCHERS[mode]}<{args}>::run"
    return load_jit(
        "diffusion_minimax_h3_vae_residual_norm",
        mode,
        *args,
        cuda_files=["diffusion/minimax_h3_vae_residual_norm.cuh"],
        cuda_wrappers=[(mode, launcher)],
    )


@register_custom_op(mutates_args=["out_q", "out_s"])
def minimax_h3_vae_rmsnorm_fp8_raw(
    x: torch.Tensor,
    weight: torch.Tensor,
    out_q: torch.Tensor,
    out_s: torch.Tensor,
    eps: float,
) -> None:
    """``out_q, out_s = fp8(RMSNorm(x) * weight)`` on 2-D rows."""
    module = _jit_minimax_h3_vae_residual_norm_module(
        "rmsnorm_fp8", x.dtype, None, x.shape[-1]
    )
    module.rmsnorm_fp8(x, weight, out_q, out_s, eps)


@register_custom_op(mutates_args=["residual_out", "out_q", "out_s"])
def minimax_h3_vae_residual_rmsnorm_fp8_raw(
    x: torch.Tensor,
    projected: torch.Tensor,
    layer_scale: torch.Tensor,
    weight: torch.Tensor,
    residual_out: torch.Tensor,
    out_q: torch.Tensor,
    out_s: torch.Tensor,
    eps: float,
) -> None:
    """``residual_out = fma(projected, layer_scale, x)``, then RMSNorm + FP8."""
    module = _jit_minimax_h3_vae_residual_norm_module(
        "residual_rmsnorm_fp8", x.dtype, projected.dtype, x.shape[-1]
    )
    module.residual_rmsnorm_fp8(
        x, projected, layer_scale, weight, residual_out, out_q, out_s, eps
    )


@register_custom_op(mutates_args=["out"])
def minimax_h3_vae_residual_layernorm_raw(
    x: torch.Tensor,
    projected: torch.Tensor,
    layer_scale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    eps: float,
) -> None:
    """``out = LayerNorm(fma(projected, layer_scale, x)) * weight + bias``."""
    module = _jit_minimax_h3_vae_residual_norm_module(
        "residual_layernorm", x.dtype, projected.dtype, x.shape[-1]
    )
    module.residual_layernorm(x, projected, layer_scale, weight, bias, out, eps)


def minimax_h3_vae_rmsnorm_fp8(
    x: torch.Tensor, weight: torch.Tensor, *, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(q, scales)``: fp8 e4m3 ``[M, width]`` and fp32 ``[M, 1]``."""
    if x.dim() < 2:
        raise ValueError("x must have at least two dimensions")
    rows = x.view(-1, x.shape[-1])
    out_q = torch.empty_like(rows, dtype=torch.float8_e4m3fn)
    out_s = torch.empty((rows.shape[0], 1), dtype=torch.float32, device=x.device)
    minimax_h3_vae_rmsnorm_fp8_raw(rows, weight, out_q, out_s, float(eps))
    return out_q, out_s


def minimax_h3_vae_residual_rmsnorm_fp8(
    x: torch.Tensor,
    projected: torch.Tensor,
    layer_scale: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(residual, q, scales)``; ``residual`` is fp32 with ``x.shape``."""
    if x.dim() < 2 or projected.shape != x.shape:
        raise ValueError(
            "x and projected must have the same shape and at least two dimensions"
        )
    rows = x.view(-1, x.shape[-1])
    residual = torch.empty_like(rows, dtype=torch.float32)
    out_q = torch.empty_like(rows, dtype=torch.float8_e4m3fn)
    out_s = torch.empty((rows.shape[0], 1), dtype=torch.float32, device=x.device)
    minimax_h3_vae_residual_rmsnorm_fp8_raw(
        rows,
        projected.view_as(rows),
        layer_scale,
        weight,
        residual,
        out_q,
        out_s,
        float(eps),
    )
    return residual.view(x.shape), out_q, out_s


def minimax_h3_vae_residual_layernorm(
    x: torch.Tensor,
    projected: torch.Tensor,
    layer_scale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Return the fp32 LayerNorm of the updated residual, shaped like ``x``."""
    if x.dim() < 2 or projected.shape != x.shape:
        raise ValueError(
            "x and projected must have the same shape and at least two dimensions"
        )
    rows = x.view(-1, x.shape[-1])
    out = torch.empty_like(rows, dtype=torch.float32)
    minimax_h3_vae_residual_layernorm_raw(
        rows, projected.view_as(rows), layer_scale, weight, bias, out, float(eps)
    )
    return out.view(x.shape)


def _row_ok(x: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.dim() >= 2
        and x.dtype in _ROW_DTYPES
        and 0 < x.shape[-1] <= _MAX_WIDTH
        and x.shape[-1] % _WIDTH_MULTIPLE == 0
        and x.numel() > 0
        and x.is_contiguous()
        and x.data_ptr() % _ALIGNMENT == 0
    )


def _projected_ok(projected: torch.Tensor, x: torch.Tensor) -> bool:
    return (
        projected.shape == x.shape
        and projected.dtype in _ROW_DTYPES
        and projected.device == x.device
        and projected.is_contiguous()
        and projected.data_ptr() % _ALIGNMENT == 0
    )


def _vector_ok(vector: torch.Tensor, x: torch.Tensor) -> bool:
    return (
        vector.shape == (x.shape[-1],)
        and vector.dtype == torch.float32
        and vector.device == x.device
        and vector.is_contiguous()
        and vector.data_ptr() % _ALIGNMENT == 0
    )


def can_use_minimax_h3_vae_rmsnorm_fp8(
    x: torch.Tensor, weight: torch.Tensor, *, eps: float
) -> bool:
    return (
        not torch.compiler.is_compiling()
        and _row_ok(x)
        and _vector_ok(weight, x)
        and eps > 0
        and is_cuda_sm_at_least(_MIN_SM, x.device)
    )


def can_use_minimax_h3_vae_residual_rmsnorm_fp8(
    x: torch.Tensor,
    projected: torch.Tensor,
    layer_scale: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
) -> bool:
    return (
        not torch.compiler.is_compiling()
        and _row_ok(x)
        and _projected_ok(projected, x)
        and _vector_ok(layer_scale, x)
        and _vector_ok(weight, x)
        and eps > 0
        and is_cuda_sm_at_least(_MIN_SM, x.device)
    )


def can_use_minimax_h3_vae_residual_layernorm(
    x: torch.Tensor,
    projected: torch.Tensor,
    layer_scale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    eps: float,
) -> bool:
    return (
        not torch.compiler.is_compiling()
        and _row_ok(x)
        and _projected_ok(projected, x)
        and _vector_ok(layer_scale, x)
        and _vector_ok(weight, x)
        and _vector_ok(bias, x)
        and eps > 0
        and is_cuda_sm_at_least(_MIN_SM, x.device)
    )


__all__ = [
    "can_use_minimax_h3_vae_residual_layernorm",
    "can_use_minimax_h3_vae_residual_rmsnorm_fp8",
    "can_use_minimax_h3_vae_rmsnorm_fp8",
    "minimax_h3_vae_residual_layernorm",
    "minimax_h3_vae_residual_layernorm_raw",
    "minimax_h3_vae_residual_rmsnorm_fp8",
    "minimax_h3_vae_residual_rmsnorm_fp8_raw",
    "minimax_h3_vae_rmsnorm_fp8",
    "minimax_h3_vae_rmsnorm_fp8_raw",
]
