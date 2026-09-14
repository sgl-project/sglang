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

from sglang.kernels.ops.diffusion.common.numerics import round_bf16_to_fp32

_E4M3 = torch.float8_e4m3fn


def _scale_numel(rows: int, k: int) -> int:
    n_groups = k // 32
    return -(-rows // 128) * 128 * (-(-n_groups // 4) * 4)


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
    act = (gate * tl.sigmoid(gate)).to(tl.bfloat16).to(tl.float32)
    prod = (act * up).to(tl.bfloat16).to(tl.float32)
    p3 = tl.reshape(prod, [BLOCK_R, G, 32])
    amax = tl.max(tl.abs(p3), axis=2)
    sbyte, inv = _mx_e8m0_from_amax(amax)
    q = tl.reshape(p3 * inv[:, :, None], [BLOCK_R, G * 32])
    tl.store(
        q_ptr + r[:, None].to(tl.int64) * hidden + c[None, :],
        q.to(tl.float8e4nv),
        mask=mask,
    )
    smask = rmask[:, None] & (g < n_groups)[None, :]
    tl.store(
        s_ptr + _mx_scale_offsets(r[:, None], g[None, :], n_col_blocks),
        sbyte.to(tl.uint8),
        mask=smask,
    )


@triton.jit
def _indexed_scale_shift_mxfp8_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    hidden_size,
    n_groups,
    n_col_blocks,
    stride_x_row,
    stride_shift_row,
    stride_scale_row,
    stride_indices,
    STORE_BF16: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_N)
    mask = columns < hidden_size
    index = tl.load(indices_ptr + row * stride_indices)
    xrow = x_ptr + row.to(tl.int64) * stride_x_row
    x = tl.load(xrow + columns, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(
        shift_ptr + index * stride_shift_row + columns, mask=mask, other=0.0
    ).to(tl.float32)
    scale = tl.load(
        scale_ptr + index * stride_scale_row + columns, mask=mask, other=0.0
    ).to(tl.float32)
    # the rounding points of _indexed_scale_shift_bf16_kernel
    one_plus_scale = round_bf16_to_fp32(1.0 + scale)
    scaled = round_bf16_to_fp32(x * one_plus_scale)
    out = round_bf16_to_fp32(scaled + shift)
    if STORE_BF16:
        tl.store(xrow + columns, out, mask=mask)
    v3 = tl.reshape(out, [BLOCK_N // 32, 32])
    amax = tl.max(tl.abs(v3), axis=1)
    sbyte, inv = _mx_e8m0_from_amax(amax)
    q = tl.reshape(v3 * inv[:, None], [BLOCK_N])
    tl.store(
        q_ptr + row.to(tl.int64) * hidden_size + columns,
        q.to(tl.float8e4nv),
        mask=mask,
    )
    g = tl.arange(0, BLOCK_N // 32)
    tl.store(
        s_ptr + _mx_scale_offsets(row, g, n_col_blocks),
        sbyte.to(tl.uint8),
        mask=g < n_groups,
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
    rows: int, k: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.empty(rows, k, dtype=_E4M3, device=device)
    s = torch.zeros(_scale_numel(rows, k), dtype=torch.uint8, device=device)
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
    n_col_blocks = -(-n_groups // 4)
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


def can_use_silu_mul_mxfp8(hidden: torch.Tensor) -> bool:
    return can_use_mxfp8_swizzled(hidden) and (hidden.shape[-1] // 2) % 32 == 0


def silu_mul_mxfp8(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``hidden [rows, 2n]`` bf16 (gate | up) -> quantized ``silu(gate) * up``."""
    if not can_use_silu_mul_mxfp8(hidden):
        raise ValueError(
            "expected a row-major bf16 CUDA [rows, 2 * n] tensor, n % 32 == 0"
        )
    rows, twice = hidden.shape
    n = twice // 2
    q, s = _alloc(rows, n, hidden.device)
    if rows == 0:
        return q, s
    n_groups = n // 32
    n_col_blocks = -(-n_groups // 4)
    block_r, g = 16, 8
    grid = (triton.cdiv(rows, block_r), triton.cdiv(n_groups, g))
    with torch.get_device_module().device(hidden.device):
        _silu_mul_mxfp8_kernel[grid](
            hidden,
            q,
            s,
            rows,
            n,
            n_groups,
            n_col_blocks,
            hidden.stride(0),
            BLOCK_R=block_r,
            G=g,
            num_warps=4,
        )
    return q, s


def indexed_scale_shift_mxfp8_(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    *,
    keep_bf16: bool,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """``x * (1 + scale[idx]) + shift[idx]`` -> ``(x | None, fp8, scales)``; with
    ``keep_bf16`` the bf16 result is also written into ``x`` and returned."""
    if not can_use_mxfp8_swizzled(x):
        raise ValueError(
            "expected a row-major bf16 CUDA [rows, hidden] tensor, hidden % 32 == 0"
        )
    rows, hidden_size = x.shape
    q, s = _alloc(rows, hidden_size, x.device)
    if rows == 0:
        return (x if keep_bf16 else None), q, s
    n_groups = hidden_size // 32
    n_col_blocks = -(-n_groups // 4)
    block_n = triton.next_power_of_2(hidden_size)
    with torch.get_device_module().device(x.device):
        _indexed_scale_shift_mxfp8_kernel[(rows,)](
            x,
            q,
            s,
            shift,
            scale,
            indices,
            hidden_size,
            n_groups,
            n_col_blocks,
            x.stride(0),
            shift.stride(0),
            scale.stride(0),
            indices.stride(0),
            STORE_BF16=keep_bf16,
            BLOCK_N=block_n,
            num_warps=8,
        )
    return (x if keep_bf16 else None), q, s


__all__ = [
    "can_use_mxfp8_swizzled",
    "can_use_silu_mul_mxfp8",
    "indexed_scale_shift_mxfp8_",
    "mxfp8_quantize_swizzled",
    "silu_mul_mxfp8",
]
