"""RMSNorm with the MXFP8 epilogue of ``ops.quantization.mxfp8_epilogue``."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.mxfp8_epilogue import mxfp8_epilogue


@triton.jit
def _rmsnorm_mxfp8_kernel(
    X,
    W,
    Y,
    Q,
    S,
    SX: tl.constexpr,
    M,
    K: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    GROUPS: tl.constexpr,
):
    row = tl.program_id(0)
    if row < M:
        h = tl.arange(0, BLOCK)
        v = tl.load(X + row * SX + h, h < K, 0).to(tl.float32)
        weight = tl.load(W + h, h < K, 0).to(tl.float32)
        inv = tl.rsqrt(tl.sum(v * v, 0) / K + EPS)
        y = (v * inv * weight).to(tl.bfloat16)
        tl.store(Y + row * K + h, y, h < K)
        mxfp8_epilogue(y, row, Q, S, K, BLOCK, GROUPS, 0, GROUPS)
    else:
        # Padding scale entries are disjoint from the live rows above.
        last = (tl.cdiv(M, 128) - 1) * 128
        local = (row - M) * 512 + tl.arange(0, 512)
        pad_row = last + ((local // 4) % 4) * 32 + ((local // 16) % 32)
        tl.store(S + last * GROUPS + local, 0, (local < GROUPS * 128) & (pad_row >= M))


def rmsnorm_mxfp8(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RMSNorm returning ``(y_bf16, y_q, y_sf)`` with the same MXFP8 epilogue."""
    m, k = x.shape
    assert m > 0 and k % 128 == 0
    assert x.dtype == weight.dtype == torch.bfloat16 and x.stride(1) == 1
    y = torch.empty_like(x, memory_format=torch.contiguous_format)
    q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty(
        (k // 32) * triton.cdiv(m, 128) * 128, dtype=torch.uint8, device=x.device
    )
    _rmsnorm_mxfp8_kernel[(m + triton.cdiv((k // 32) * 128, 512),)](
        x,
        weight,
        y,
        q,
        s,
        SX=x.stride(0),
        M=m,
        K=k,
        EPS=eps,
        BLOCK=triton.next_power_of_2(k),
        GROUPS=k // 32,
        num_warps=8,
        enable_fp_fusion=False,
    )
    return y, q, s
