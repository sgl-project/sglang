"""SiLU(gate) * up with per-token FP8 quantization (SM100+).

Computes in FP32 with expf and no intermediate FP16/BF16 rounding. Quantization
uses scale = amax/448; zero rows produce zero scale and codes. Inputs must be
contiguous and 16-byte aligned. Results are close, not bit-exact, to eager.
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
_DTYPES = (torch.float16, torch.bfloat16)
# kBlockSize * kVecSize in silu_mul_quant_fp8.cuh for 16-bit inputs.
_WIDTH_MULTIPLE = 2048
# The fp32 products live in registers (width / 256 floats per thread); 8192 is
# the verified MiniMax-H3 FFN width and already 32 floats per thread.
_MAX_WIDTH = 8192


@cache_once
def _jit_silu_mul_quant_fp8_module(dtype: torch.dtype, width: int) -> Module:
    if dtype not in _DTYPES:
        raise RuntimeError(f"Unsupported dtype {dtype}; expected float16 or bfloat16")
    if width <= 0 or width % _WIDTH_MULTIPLE != 0 or width > _MAX_WIDTH:
        raise RuntimeError(
            f"Unsupported output width {width}; expected a multiple of "
            f"{_WIDTH_MULTIPLE} up to {_MAX_WIDTH}"
        )
    args = make_cpp_args(dtype, width)
    return load_jit(
        "diffusion_silu_mul_quant_fp8",
        *args,
        cuda_files=["diffusion/silu_mul_quant_fp8.cuh"],
        cuda_wrappers=[
            (
                "silu_mul_quant_fp8",
                f"silu_mul_quant_fp8::SiluMulQuantFp8Kernel<{args}>::run",
            )
        ],
    )


@register_custom_op(mutates_args=["out_q", "out_s"])
def silu_mul_quant_fp8_raw(
    x: torch.Tensor, out_q: torch.Tensor, out_s: torch.Tensor
) -> None:
    """``out_q, out_s = fp8(silu(gate) * up)`` on 2-D ``[gate | up]`` rows."""
    module = _jit_silu_mul_quant_fp8_module(x.dtype, out_q.shape[-1])
    module.silu_mul_quant_fp8(x, out_q, out_s)


def silu_mul_quant_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(q, scales)``: fp8 e4m3 ``[M, width]`` and fp32 ``[M, 1]``."""
    if x.dim() < 2 or x.shape[-1] % 2:
        raise ValueError(
            "x must have at least two dimensions and an even gate/up width"
        )
    width = x.shape[-1] // 2
    rows = x.view(-1, 2 * width)
    out_q = torch.empty(
        (rows.shape[0], width), dtype=torch.float8_e4m3fn, device=x.device
    )
    out_s = torch.empty((rows.shape[0], 1), dtype=torch.float32, device=x.device)
    silu_mul_quant_fp8_raw(rows, out_q, out_s)
    return out_q, out_s


def can_use_silu_mul_quant_fp8(x: torch.Tensor) -> bool:
    return (
        not torch.compiler.is_compiling()
        and x.is_cuda
        and x.dim() >= 2
        and x.dtype in _DTYPES
        and 0 < x.shape[-1] <= 2 * _MAX_WIDTH
        and x.shape[-1] % (2 * _WIDTH_MULTIPLE) == 0
        and x.numel() > 0
        and x.is_contiguous()
        and x.data_ptr() % _ALIGNMENT == 0
        and is_cuda_sm_at_least(_MIN_SM, x.device)
    )


__all__ = [
    "can_use_silu_mul_quant_fp8",
    "silu_mul_quant_fp8",
    "silu_mul_quant_fp8_raw",
]
