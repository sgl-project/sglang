"""JIT GEMM computing y[m, n] = x[m, k](bf16) @ w[n, k]^T(fp32) on SM90.

The fp32 weight is pre-split into two bf16 tensors (``split_fp32_weight``):
    w_high = w.to(bf16)
    w_low  = ((w - w_high.float()) / scale).to(bf16)
and the kernel recovers near-fp32 accuracy at bf16 GMMA throughput by
combining the two GMMA results as ``y = y_high + scale * y_low``.

The kernel and its shape heuristic are adapted from Tencent HPC-Ops
(https://github.com/Tencent/hpc-ops, MIT License); see
csrc/gemm/gemm_bf16xfp32_sm90.cuh for the CUDA source.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_hip_runtime,
    load_jit,
    make_cpp_args,
    override_jit_cuda_arch,
)
from sglang.kernel_api_logging import debug_kernel_api
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Scale used to split an fp32 weight into (w_high, w_low) bf16 pairs. 2^-8
# shifts the residual into bf16's representable range without overflow.
WEIGHT_SPLIT_SCALE = 1.0 / 256.0


def is_gemm_bf16xfp32_supported(device: Optional[torch.device] = None) -> bool:
    """The kernel uses SM90 GMMA/TMA instructions and is Hopper-only."""
    if is_hip_runtime() or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(device)[0] == 9


def split_fp32_weight(
    w: torch.Tensor, scale: float = WEIGHT_SPLIT_SCALE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split an fp32 weight into the (w_high, w_low) bf16 pair the kernel expects."""
    w_high = w.to(torch.bfloat16).contiguous()
    w_low = ((w - w_high.float()) / scale).to(torch.bfloat16).contiguous()
    return w_high, w_low


def _select_launch_config(
    m: int, n: int, k: int
) -> Tuple[int, int, int, int, int, int]:
    """Pick (tile_m, tile_n, tile_k, stage, wgn, split_k) for a problem shape.

    Port of ``select_config`` in hpc-ops src/gemm/sm90/entry.cc plus the
    (stage, tile_k) resolution in ``gemm_bf16xfp32_async``. The thresholds are
    over ``norm_m``, the workload normalized to a reference n=192, k=4096
    problem, and were tuned upstream on SM90 parts.
    """
    norm_m = (m * n * 4096 + 192 * k - 1) // (192 * k)

    if 624 < norm_m <= 832:
        split_k, wgn, tile_m = 2, 1, 64
    elif 832 < norm_m <= 896:
        split_k, wgn, tile_m = 2, 2, 16
    elif 1024 < norm_m <= 1088:
        split_k, wgn, tile_m = 1, 2, 16
    elif 1088 < norm_m <= 1152:
        split_k, wgn, tile_m = 4, 1, 64
    elif 1152 < norm_m <= 1536:
        split_k, wgn, tile_m = 1, 1, 64
    elif 1536 < norm_m <= 2048:
        split_k, wgn, tile_m = 4, 1, 64
    elif norm_m > 2048:
        split_k, wgn, tile_m = 1, 1, 64
    else:
        # tile_m=16 path: split_k by workload, then wgn by occupancy.
        if norm_m <= 64:
            split_k = 8
        elif norm_m <= 144:
            split_k = 4
        elif norm_m <= 304:
            split_k = 2
        else:
            split_k = 1
        tiles_with_wgn2 = -(m // -16) * -(n // -128) * split_k
        wgn = 1 if tiles_with_wgn2 < 64 else 2
        tile_m = 16

    if tile_m == 64:
        stage = 5 if split_k == 4 else 3
        return (64, 64, 64, stage, 1, split_k)
    return (16, 64, 128, 3, wgn, split_k)


@cache_once
def _jit_gemm_bf16xfp32_module(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    stage: int,
    wgn: int,
    split_k: int,
    fp32_out: bool,
) -> Module:
    args = make_cpp_args(tile_m, tile_n, tile_k, stage, wgn, split_k, fp32_out)
    with override_jit_cuda_arch(9, 0, "a"):
        return load_jit(
            "gemm_bf16xfp32",
            *args,
            cuda_files=["gemm/gemm_bf16xfp32_sm90.cuh"],
            cuda_wrappers=[
                ("gemm_bf16xfp32", f"GemmBf16xFp32Kernel<{args}>::run"),
            ],
            extra_cuda_cflags=[
                "-DNDEBUG",
                "-DCUTE_USE_PACKED_TUPLE=1",
                "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
            ],
            extra_dependencies=["cutlass"],
        )


@register_custom_op(
    op_name="gemm_bf16xfp32",
    mutates_args=["output"],
)
def _gemm_bf16xfp32_custom_op(
    x: torch.Tensor,
    w_high: torch.Tensor,
    w_low: torch.Tensor,
    output: torch.Tensor,
    scale: float,
) -> None:
    m, k = x.shape
    n = w_high.shape[0]
    tile_m, tile_n, tile_k, stage, wgn, split_k = _select_launch_config(m, n, k)
    module = _jit_gemm_bf16xfp32_module(
        tile_m, tile_n, tile_k, stage, wgn, split_k, output.dtype == torch.float32
    )
    if split_k > 1:
        # Per-call workspaces; the kernel resets split_flag to zero before it
        # exits, so a captured (zeros + kernel) pair stays replay-safe under
        # CUDA graphs.
        split_y = torch.empty((split_k, m, n), dtype=torch.float32, device=x.device)
        split_flag = torch.zeros(
            (-(m // -tile_m), -(n // -(tile_n * wgn))),
            dtype=torch.int32,
            device=x.device,
        )
    else:
        split_y = None
        split_flag = None
    module.gemm_bf16xfp32(x, w_high, w_low, output, split_y, split_flag, scale)
    return None


@debug_kernel_api
def gemm_bf16xfp32(
    x: torch.Tensor,
    w_high: torch.Tensor,
    w_low: torch.Tensor,
    scale: float = WEIGHT_SPLIT_SCALE,
    *,
    out_dtype: torch.dtype = torch.float32,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute ``x @ w^T`` for bf16 activations and a split fp32 weight.

    Args:
        x: Activation tensor of shape [m, k], bfloat16, contiguous.
        w_high: High bf16 part of the fp32 weight, shape [n, k], contiguous.
        w_low: Low (residual) bf16 part, shape [n, k], contiguous.
            ``n`` must be divisible by 64. See ``split_fp32_weight``.
        scale: Scaling factor applied to the low-part GEMM result.
        out_dtype: torch.float32 or torch.bfloat16.
        out: Optional pre-allocated output tensor of shape [m, n].

    Returns:
        Output tensor of shape [m, n] with dtype ``out_dtype``.
    """
    assert x.dim() == 2 and w_high.dim() == 2 and w_low.dim() == 2
    assert x.dtype == torch.bfloat16
    assert w_high.dtype == torch.bfloat16 and w_low.dtype == torch.bfloat16
    assert w_high.shape == w_low.shape and x.shape[1] == w_high.shape[1]
    assert w_high.shape[0] % 64 == 0, "n must be divisible by 64"
    assert x.is_contiguous() and w_high.is_contiguous() and w_low.is_contiguous()
    if out is None:
        out = torch.empty(x.shape[0], w_high.shape[0], dtype=out_dtype, device=x.device)
    _gemm_bf16xfp32_custom_op(x, w_high, w_low, out, scale)
    return out
