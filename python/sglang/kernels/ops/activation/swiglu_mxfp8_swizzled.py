# SPDX-License-Identifier: Apache-2.0
"""SwiGLU producers of MXFP8: ``silu(gate) * up`` or the clamped SwiGLU-OAI,
quantized in the same pass with the scale layouts of
``ops.quantization.mxfp8_swizzled_triton``. Each quantizes the bf16-rounded
value the unfused activation stores, so it is byte-exact against that kernel
followed by ``mxfp8_quantize_swizzled``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.mxfp8_swizzled_triton import (
    _E4M3,
    _alloc,
    _mx_e8m0_from_amax,
    _mx_scale_offsets,
    can_use_mxfp8_swizzled,
)


@triton.jit
def _silu_mul_mxfp8_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    rows,
    hidden,
    n_groups,
    n_col_blocks,
    stride_row,
    dg_rows,
    ALPHA: tl.constexpr,
    LIMIT: tl.constexpr,
    DEEPGEMM: tl.constexpr,
    BLOCK_R: tl.constexpr,
    G: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    g = pid_c * G + tl.arange(0, G)
    c = pid_c * (G * 32) + tl.arange(0, G * 32)
    rmask = r < rows
    mask = rmask[:, None] & (c < hidden)[None, :]
    base = x_ptr + r[:, None].to(tl.int64) * stride_row
    gate = tl.load(base + c[None, :], mask=mask, other=0.0).to(tl.float32)
    up = tl.load(base + hidden + c[None, :], mask=mask, other=0.0).to(tl.float32)
    if ALPHA is None:
        act = (gate * tl.sigmoid(gate)).to(tl.bfloat16).to(tl.float32)
        prod = (act * up).to(tl.bfloat16).to(tl.float32)
    else:
        # the op order Inductor emits for swiglu_no_interleaved_with_alpha_and_limit
        gate = tl.minimum(gate, LIMIT, propagate_nan=tl.PropagateNan.ALL)
        up = tl.maximum(up, -LIMIT, propagate_nan=tl.PropagateNan.ALL)
        up = tl.minimum(up, LIMIT, propagate_nan=tl.PropagateNan.ALL)
        prod = gate * tl.sigmoid(gate * ALPHA) * (up + 1.0)
        prod = prod.to(tl.bfloat16).to(tl.float32)
    p3 = tl.reshape(prod, [BLOCK_R, G, 32])
    amax = tl.max(tl.abs(p3), axis=2)
    if DEEPGEMM:
        amax = tl.maximum(amax, 1e-10)
    sbyte, inv = _mx_e8m0_from_amax(amax)
    q = tl.reshape(p3 * inv[:, :, None], [BLOCK_R, G * 32])
    tl.store(
        q_ptr + r[:, None].to(tl.int64) * hidden + c[None, :],
        q.to(tl.float8e4nv),
        mask=mask,
    )
    valid = rmask[:, None] & (g < n_groups)[None, :]
    if DEEPGEMM:
        # 4 bytes per int32 along K, MN-major, rows padded to 4
        soff = (
            (g[None, :] // 4).to(tl.int64) * dg_rows * 4
            + r[:, None] * 4
            + g[None, :] % 4
        )
        tl.store(s_ptr + soff, sbyte.to(tl.uint8), mask=valid)
    else:
        smask = (r < tl.cdiv(rows, 128) * 128)[:, None] & (g < n_col_blocks * 4)[
            None, :
        ]
        tl.store(
            s_ptr + _mx_scale_offsets(r[:, None], g[None, :], n_col_blocks),
            tl.where(valid, sbyte, 0).to(tl.uint8),
            mask=smask,
        )


def can_use_silu_mul_mxfp8(hidden: torch.Tensor) -> bool:
    return can_use_mxfp8_swizzled(hidden) and (hidden.shape[-1] // 2) % 32 == 0


def _swiglu_mxfp8(
    hidden: torch.Tensor,
    alpha: float | None,
    limit: float | None,
    deepgemm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not can_use_silu_mul_mxfp8(hidden):
        raise ValueError(
            "expected a row-major bf16 CUDA [rows, 2 * n] tensor, n % 32 == 0"
        )
    rows, twice = hidden.shape
    n = twice // 2
    if deepgemm:
        if n % 128:
            raise ValueError(f"DeepGEMM MXFP8 scales need n % 128 == 0, got {n=}")
        q = torch.empty(rows, n, dtype=_E4M3, device=hidden.device)
        dg_rows = triton.cdiv(rows, 4) * 4
        base = torch.empty(n // 128, dg_rows, dtype=torch.int32, device=hidden.device)
        s = base.T[:rows]
        s_bytes = base.view(torch.uint8)
        padded_rows = rows
    else:
        q, s = _alloc(rows, n, hidden.device, zero_scales=False)
        s_bytes = s
        dg_rows = 0
        padded_rows = triton.cdiv(rows, 128) * 128
    if rows == 0:
        return q, s
    n_groups = n // 32
    n_col_blocks = triton.cdiv(n_groups, 4)
    block_r, g = (4 if rows <= 1024 else 16), 8
    grid = (triton.cdiv(padded_rows, block_r), triton.cdiv(n_groups, g))
    with torch.get_device_module().device(hidden.device):
        _silu_mul_mxfp8_kernel[grid](
            hidden,
            q,
            s_bytes,
            rows,
            n,
            n_groups,
            n_col_blocks,
            hidden.stride(0),
            dg_rows,
            ALPHA=alpha,
            LIMIT=limit,
            DEEPGEMM=deepgemm,
            BLOCK_R=block_r,
            G=g,
            num_warps=4,
        )
    return q, s


def silu_mul_mxfp8(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``hidden [rows, 2n]`` bf16 (gate | up) -> quantized ``silu(gate) * up``."""
    return _swiglu_mxfp8(hidden, None, None)


def swiglu_oai_mxfp8(
    hidden: torch.Tensor, alpha: float, limit: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """``hidden [rows, 2n]`` bf16 (gate | up) -> quantized SwiGLU-OAI,
    ``clamp(gate, max=limit) * sigmoid(alpha * gate) * (clamp(up, +-limit) + 1)``."""
    return _swiglu_mxfp8(hidden, alpha, limit)


def swiglu_oai_mxfp8_deepgemm(
    hidden: torch.Tensor, alpha: float, limit: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same values as ``swiglu_oai_mxfp8``; scales in DeepGEMM's layout, the
    ``sglang_per_token_group_quant_fp8(..., 32, column_major_scales=True,
    scale_tma_aligned=True, scale_ue8m0=True)`` output: int32 ``[rows, n / 128]``,
    four UE8M0 bytes each, MN-major with rows padded to 4."""
    return _swiglu_mxfp8(hidden, alpha, limit, deepgemm=True)


__all__ = [
    "can_use_silu_mul_mxfp8",
    "silu_mul_mxfp8",
    "swiglu_oai_mxfp8",
    "swiglu_oai_mxfp8_deepgemm",
]
