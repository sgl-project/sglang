# SPDX-License-Identifier: Apache-2.0
"""MXFP8 producers: e4m3 payload plus one E8M0 scale per 32 elements along K,
the scales in the cuBLASLt ``SWIZZLE_32_4_4`` layout that
``torch.nn.functional.scaled_mm(..., BlockWise1x32)`` consumes on SM100.

Scale ``(r, c)`` of the ``[rows, K/32]`` scale matrix lives at byte
``((r // 128) * ceil(K/32 / 4) + c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4``,
rows padded to 128 and scale columns to 4, padding zero. Exponent
``e = ceil(log2(amax / 448))`` exactly from the float bits; ``q = e4m3(x * 2**-e)``;
scale byte ``e + 127``. Every producer quantizes the bf16-rounded value the
unfused bf16 kernel stores, so each is byte-exact against that kernel followed
by ``mxfp8_quantize_swizzled`` (itself byte-exact vs ``flashinfer.mxfp8_quantize``).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_E4M3 = torch.float8_e4m3fn


def _scale_numel(rows: int, k: int) -> int:
    n_groups = k // 32
    return triton.cdiv(rows, 128) * 128 * (triton.cdiv(n_groups, 4) * 4)


@triton.jit
def _mx_e8m0_from_amax(amax):
    bits = amax.to(tl.int32, bitcast=True)
    e0 = ((bits >> 23) & 0xFF) - 135
    # amax / 2**e0 lies in [256, 512); bump e0 when it exceeds 448 = 1.75 * 2**8
    thr = (((e0 + 135) << 23) | 0x600000).to(tl.float32, bitcast=True)
    e = e0 + (amax > thr).to(tl.int32)
    e = tl.maximum(e, -127)
    inv = ((127 - e) << 23).to(tl.float32, bitcast=True)
    return e + 127, inv


@triton.jit
def _mx_scale_offsets(r, c, n_col_blocks):
    tile = (r // 128) * n_col_blocks + (c // 4)
    return tile * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + (c % 4)


@triton.jit
def _mxfp8_quant_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    rows,
    k,
    n_groups,
    n_col_blocks,
    stride_x,
    BLOCK_R: tl.constexpr,
    G: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_g = tl.program_id(1)
    r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    g = pid_g * G + tl.arange(0, G)
    c = pid_g * (G * 32) + tl.arange(0, G * 32)
    rmask = r < rows
    mask = rmask[:, None] & (c < k)[None, :]
    x = tl.load(
        x_ptr + r[:, None].to(tl.int64) * stride_x + c[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    x3 = tl.reshape(x, [BLOCK_R, G, 32])
    amax = tl.max(tl.abs(x3), axis=2)
    sbyte, inv = _mx_e8m0_from_amax(amax)
    q = tl.reshape(x3 * inv[:, :, None], [BLOCK_R, G * 32])
    tl.store(
        q_ptr + r[:, None].to(tl.int64) * k + c[None, :],
        q.to(tl.float8e4nv),
        mask=mask,
    )
    smask = rmask[:, None] & (g < n_groups)[None, :]
    tl.store(
        s_ptr + _mx_scale_offsets(r[:, None], g[None, :], n_col_blocks),
        sbyte.to(tl.uint8),
        mask=smask,
    )


def can_use_mxfp8_swizzled(x: torch.Tensor) -> bool:
    """Row-major bf16 CUDA 2D tensor with K % 32 == 0, outside torch.compile."""
    return (
        x.is_cuda
        and x.ndim == 2
        and x.dtype == torch.bfloat16
        and x.stride(-1) == 1
        and x.shape[-1] % 32 == 0
        and not torch.compiler.is_compiling()
    )


def _alloc(
    rows: int, k: int, device: torch.device, zero_scales: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.empty(rows, k, dtype=_E4M3, device=device)
    alloc = torch.zeros if zero_scales else torch.empty
    s = alloc(_scale_numel(rows, k), dtype=torch.uint8, device=device)
    return q, s


def mxfp8_quantize_swizzled(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 ``[rows, k]`` -> ``(fp8 [rows, k], swizzled e8m0 scale bytes)``."""
    if not can_use_mxfp8_swizzled(x):
        raise ValueError("expected a row-major bf16 CUDA [rows, k] tensor, k % 32 == 0")
    rows, k = x.shape
    q, s = _alloc(rows, k, x.device)
    if rows == 0:
        return q, s
    n_groups = k // 32
    n_col_blocks = triton.cdiv(n_groups, 4)
    block_r, g = 32, 8
    grid = (triton.cdiv(rows, block_r), triton.cdiv(n_groups, g))
    with torch.get_device_module().device(x.device):
        _mxfp8_quant_kernel[grid](
            x,
            q,
            s,
            rows,
            k,
            n_groups,
            n_col_blocks,
            x.stride(0),
            BLOCK_R=block_r,
            G=g,
            num_warps=4,
        )
    return q, s


__all__ = [
    "can_use_mxfp8_swizzled",
    "mxfp8_quantize_swizzled",
]
