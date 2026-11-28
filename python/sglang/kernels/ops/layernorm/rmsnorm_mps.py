"""RMSNorm and fused add + RMSNorm for the Torch MPS eager path (Metal JIT)."""

from __future__ import annotations

import functools
from typing import Optional, Tuple, Union

import torch

# One threadgroup normalizes one row. Every reduction and the residual sum run
# in fp32, matching RMSNorm.forward_native; only the stores round to T.
_METAL_SOURCE = """
#include <metal_stdlib>
using namespace metal;

template <typename T>
inline float row_inv_rms(
    device const T* x, device const T* residual, uint n, float eps,
    uint tid, uint group_size, uint lane, uint simd_id,
    threadgroup float* partials
) {
    float sum = 0.0f;
    for (uint i = tid; i < n; i += group_size) {
        float h = float(x[i]);
        if (residual != nullptr) {
            h += float(residual[i]);
        }
        sum += h * h;
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        partials[simd_id] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        uint num_simds = (group_size + 31) / 32;
        sum = lane < num_simds ? partials[lane] : 0.0f;
        sum = simd_sum(sum);
        if (lane == 0) {
            partials[0] = rsqrt(sum / float(n) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return partials[0];
}

#define DEFINE_RMSNORM(T, NAME)                                               \\
kernel void rmsnorm_##NAME(                                                   \\
    device const T* x [[buffer(0)]],                                          \\
    device const T* weight [[buffer(1)]],                                     \\
    device T* out [[buffer(2)]],                                              \\
    constant float& eps [[buffer(3)]],                                        \\
    constant uint& n [[buffer(4)]],                                           \\
    uint row [[threadgroup_position_in_grid]],                                \\
    uint tid [[thread_position_in_threadgroup]],                              \\
    uint group_size [[threads_per_threadgroup]],                              \\
    uint lane [[thread_index_in_simdgroup]],                                  \\
    uint simd_id [[simdgroup_index_in_threadgroup]]                           \\
) {                                                                           \\
    threadgroup float partials[32];                                           \\
    x += row * n;                                                             \\
    out += row * n;                                                           \\
    float inv_rms = row_inv_rms<T>(                                           \\
        x, nullptr, n, eps, tid, group_size, lane, simd_id, partials);        \\
    for (uint i = tid; i < n; i += group_size) {                              \\
        out[i] = T(float(x[i]) * inv_rms * float(weight[i]));                 \\
    }                                                                         \\
}                                                                             \\
                                                                              \\
kernel void fused_add_rmsnorm_##NAME(                                         \\
    device const T* x [[buffer(0)]],                                          \\
    device const T* residual [[buffer(1)]],                                   \\
    device const T* weight [[buffer(2)]],                                     \\
    device T* out [[buffer(3)]],                                              \\
    device T* residual_out [[buffer(4)]],                                     \\
    constant float& eps [[buffer(5)]],                                        \\
    constant uint& n [[buffer(6)]],                                           \\
    uint row [[threadgroup_position_in_grid]],                                \\
    uint tid [[thread_position_in_threadgroup]],                              \\
    uint group_size [[threads_per_threadgroup]],                              \\
    uint lane [[thread_index_in_simdgroup]],                                  \\
    uint simd_id [[simdgroup_index_in_threadgroup]]                           \\
) {                                                                           \\
    threadgroup float partials[32];                                           \\
    x += row * n;                                                             \\
    residual += row * n;                                                      \\
    out += row * n;                                                           \\
    residual_out += row * n;                                                  \\
    float inv_rms = row_inv_rms<T>(                                           \\
        x, residual, n, eps, tid, group_size, lane, simd_id, partials);       \\
    for (uint i = tid; i < n; i += group_size) {                              \\
        float h = float(x[i]) + float(residual[i]);                           \\
        residual_out[i] = T(h);                                               \\
        out[i] = T(h * inv_rms * float(weight[i]));                           \\
    }                                                                         \\
}

DEFINE_RMSNORM(float, float)
DEFINE_RMSNORM(half, half)
DEFINE_RMSNORM(bfloat, bfloat)
"""

_DTYPE_SUFFIX = {
    torch.float32: "float",
    torch.float16: "half",
    torch.bfloat16: "bfloat",
}
_MAX_GROUP_SIZE = 1024


@functools.cache
def _library():
    return torch.mps.compile_shader(_METAL_SOURCE)


def rmsnorm_mps_supported(
    x: torch.Tensor, weight: torch.Tensor, residual: Optional[torch.Tensor] = None
) -> bool:
    """Whether :func:`rmsnorm_mps` handles these operands."""
    return (
        x.device.type == "mps"
        and x.dtype in _DTYPE_SUFFIX
        and x.numel() > 0
        and weight.dtype == x.dtype
        and weight.shape == (x.shape[-1],)
        and (
            residual is None
            or (residual.dtype == x.dtype and residual.shape == x.shape)
        )
    )


def rmsnorm_mps(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """``x * rsqrt(mean(x**2) + eps) * weight`` over the last dim.

    With ``residual``, normalizes ``x + residual`` and also returns that sum.
    Neither input is modified. Callers check :func:`rmsnorm_mps_supported`.
    """
    hidden_size = x.shape[-1]
    rows = x.numel() // hidden_size
    x = x.contiguous()
    weight = weight.contiguous()
    suffix = _DTYPE_SUFFIX[x.dtype]
    # A multiple of the 32-wide SIMD group, so every simd_sum lane is in range.
    group_size = min(_MAX_GROUP_SIZE, (hidden_size + 31) // 32 * 32)
    launch = dict(threads=rows * group_size, group_size=group_size)

    out = torch.empty_like(x)
    if residual is None:
        getattr(_library(), f"rmsnorm_{suffix}")(
            x, weight, out, eps, hidden_size, **launch
        )
        return out

    residual_out = torch.empty_like(x)
    getattr(_library(), f"fused_add_rmsnorm_{suffix}")(
        x, residual.contiguous(), weight, out, residual_out, eps, hidden_size, **launch
    )
    return out, residual_out
