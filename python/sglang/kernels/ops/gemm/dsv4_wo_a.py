from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.gemm.gfx95_batched_gemm_bf16_fp8_grid import (
    _batched_gemm_bf16_fp8_grid_kernel,
    _batched_gemm_bf16_split_k_partial_kernel,
    _batched_gemm_split_k_reduce_kernel,
)
from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
    Mxfp8Activation,
    fp8_grid_quant,
)
from sglang.kernels.ops.quantization.mxfp8_epilogue import ue8m0_scale


@triton.jit
def _wo_a_bf16_gemv_kernel(X, W, Y, R: tl.constexpr, D: tl.constexpr, BN: tl.constexpr):
    group = tl.program_id(1)
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    columns = tl.arange(0, D)
    x = tl.load(X + group * D + columns).to(tl.float32)
    w = tl.load(
        W + (group * R + rows[:, None]) * D + columns[None, :],
        rows[:, None] < R,
        0,
    ).to(tl.float32)
    result = tl.sum(w * x[None, :], axis=1)
    tl.store(Y + group * R + rows, result, rows < R)


def wo_a_bf16_gemv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute ``einsum('tgd,grd->tgr', x, weight)`` for one token."""
    assert x.shape[0] == 1 and x.ndim == weight.ndim == 3
    assert x.dtype == weight.dtype == torch.bfloat16
    assert x.is_cuda and x.device == weight.device
    assert x.is_contiguous() and weight.is_contiguous()
    groups, rows, dim = weight.shape
    assert x.shape[1:] == (groups, dim) and dim == triton.next_power_of_2(dim)
    result = torch.empty((1, groups, rows), dtype=x.dtype, device=x.device)
    # One output row per CTA keeps register use low and exposes enough
    # independent weight loads for single-token decode.
    _wo_a_bf16_gemv_kernel[(rows, groups)](
        x,
        weight,
        result,
        rows,
        dim,
        1,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return result


@triton.jit
def _wo_a_partial(X, W, P, M: tl.constexpr, SX: tl.constexpr):
    tile, group, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m = tl.arange(0, 16)
    n = tile * 64 + tl.arange(0, 64)
    k = split * 512 + tl.arange(0, 128)
    acc = tl.zeros((16, 64), tl.float32)
    for i in range(4):
        offsets = k + i * 128
        x = tl.load(
            X + m[:, None] * SX + group * 4096 + offsets[None, :], m[:, None] < M, 0
        )
        w = tl.load(W + (group * 1024 + n[None, :]) * 4096 + offsets[:, None])
        acc += tl.dot(x, w)
    tl.store(
        P + ((split * M + m[:, None]) * 2 + group) * 1024 + n[None, :],
        acc,
        m[:, None] < M,
    )


@triton.jit
def _wo_a_reduce(P, Y, E: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    split = tl.arange(0, 8)
    values = tl.load(P + split[:, None] * E + i[None, :], i[None, :] < E, 0)
    tl.store(Y + i, tl.sum(values, 0), i < E)


def wo_a_bf16_small_batch(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute ``einsum('tgd,grd->tgr', x, weight)`` for the TP4 WO-A shape.
    Partial sums stay in FP32 until the final BF16, token-major store."""
    m = x.shape[0]
    assert 2 <= m <= 8 and x.shape[1:] == (2, 4096)
    assert weight.shape == (2, 1024, 4096) and weight.is_contiguous()
    assert x.dtype == weight.dtype == torch.bfloat16
    assert x.is_cuda and x.device == weight.device
    assert x.stride(2) == 1 and x.stride(1) == 4096 and x.stride(0) >= 8192
    result = torch.empty((m, 2, 1024), dtype=x.dtype, device=x.device)
    partial = torch.empty((8, m, 2, 1024), dtype=torch.float32, device=x.device)
    _wo_a_partial[(16, 2, 8)](
        x, weight, partial, m, x.stride(0), num_warps=4, num_stages=3
    )
    _wo_a_reduce[(triton.cdiv(m * 2048, 256),)](partial, result, m * 2048, num_warps=4)
    return result


@triton.jit
def _wo_a_reduce_quant(P, Q, S, M: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    i = tile * 256 + tl.arange(0, 256)
    split = tl.arange(0, 8)
    v = tl.load(P + split[:, None] * (M * 2048) + row * 2048 + i[None, :])
    y = tl.sum(v, 0).to(tl.bfloat16).to(tl.float32).reshape((8, 32))
    amax = tl.max(tl.abs(y), 1)
    sf, inv = ue8m0_scale(amax)
    quant = tl.minimum(tl.maximum(y * inv[:, None], -448.0), 448.0).to(tl.float8e4nv)
    tl.store(Q + row * 2048 + i, quant.reshape((256,)))
    col = tile * 8 + tl.arange(0, 8)
    off = (col // 4) * 512 + row * 16 + col % 4
    tl.store(S + off, sf.to(tl.uint8))
    # Zero only padding rows; valid scale bytes have disjoint writers above.
    for z in tl.static_range(triton.cdiv(8192, M * 8 * 256)):
        s = (row * 8 + tile) * 256 + tl.arange(0, 256) + z * (M * 8 * 256)
        sr = (s % 512) // 16 + ((s % 16) // 4) * 32
        tl.store(S + s, 0, (s < 8192) & (sr >= M))


def _quantize_partial(p):
    m = p.shape[1]
    q = torch.empty((m, 2048), device=p.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(8192, device=p.device, dtype=torch.uint8)
    _wo_a_reduce_quant[(m, 8)](p, q, s, m, num_warps=4)
    return q, s


def wo_a_bf16_small_batch_mxfp8(x: torch.Tensor, weight: torch.Tensor):
    """WO-A with BF16 rounding followed by FlashInfer-compatible MXFP8 quantization."""
    m = x.shape[0]
    assert 2 <= m <= 8 and x.shape[1:] == (2, 4096)
    assert x.dtype == weight.dtype == torch.bfloat16
    assert weight.shape == (2, 1024, 4096) and weight.is_contiguous()
    assert x.is_cuda and x.device == weight.device
    assert x.stride(2) == 1 and x.stride(1) == 4096 and x.stride(0) >= 8192
    partial = torch.empty((8, m, 2, 1024), dtype=torch.float32, device=x.device)
    _wo_a_partial[(16, 2, 8)](
        x, weight, partial, m, x.stride(0), num_warps=4, num_stages=3
    )
    return _quantize_partial(partial)


@triton.jit
def _wo_a_reduce_mxfp8(P, Q, S, E: tl.constexpr):
    # round to bf16 first, as the bf16 route does before wo_b's quant
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    split = tl.arange(0, 8)
    values = tl.load(P + split[:, None] * E + i[None, :], i[None, :] < E, 0)
    y = tl.sum(values, 0).to(tl.bfloat16).to(tl.float32)
    q, e = fp8_grid_quant(tl.reshape(y, (8, 32)))
    tl.store(Q + i, tl.reshape(q, (256,)))
    tl.store(S + tl.program_id(0) * 8 + tl.arange(0, 8), e.to(tl.uint8))


WO_A_MXFP8_GFX95_MAX_M = 256
# (max rows, BLOCK_M, SPLIT_K, BLOCK_K, num_warps); BLOCK_N is one 32-group for the epilogue quant
_MXFP8_GFX95_CONFIGS = (
    (16, 16, 8, 256, 2),
    (64, 32, 4, 256, 4),
    (128, 32, 1, 512, 4),
    (WO_A_MXFP8_GFX95_MAX_M, 32, 1, 256, 4),
)


def wo_a_bf16_mxfp8_gfx95(x: torch.Tensor, weight: torch.Tensor) -> Mxfp8Activation:
    """TP4 wo_a at 2-256 rows, emitting wo_b's gfx950 MXFP8 input (fp8 [M, 2048], ue8m0 [M, 64])."""
    m = x.shape[0]
    assert 2 <= m <= WO_A_MXFP8_GFX95_MAX_M and x.shape[1:] == (2, 4096)
    assert weight.shape == (2, 1024, 4096) and weight.is_contiguous()
    assert x.dtype == weight.dtype == torch.bfloat16
    assert x.is_cuda and x.device == weight.device
    assert x.stride(2) == 1 and x.stride(1) == 4096 and x.stride(0) >= 8192
    q = torch.empty((m, 2048), dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty((m, 64), dtype=torch.uint8, device=x.device)
    if m <= 8:
        partial = torch.empty((8, m, 2, 1024), dtype=torch.float32, device=x.device)
        _wo_a_partial[(16, 2, 8)](
            x, weight, partial, m, x.stride(0), num_warps=4, num_stages=3
        )
        _wo_a_reduce_mxfp8[(m * 8,)](partial, q, scale, m * 2048, num_warps=4)
        return Mxfp8Activation(q, scale)
    _, block_m, split_k, block_k, num_warps = next(
        c for c in _MXFP8_GFX95_CONFIGS if m <= c[0]
    )
    row_tiles = triton.cdiv(m, block_m)
    strides = (
        x.stride(1),
        x.stride(0),
        x.stride(2),
        weight.stride(0),
        weight.stride(2),
        weight.stride(1),
    )
    tuning = dict(num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=16)
    if split_k == 1:
        _batched_gemm_bf16_fp8_grid_kernel[(2, row_tiles * 32)](
            x,
            weight,
            q,
            m,
            1024,
            4096,
            *strides,
            1024,
            2048,
            1,
            scale,
            32,
            64,
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=32,
            BLOCK_SIZE_K=block_k,
            EVEN_K=True,
            FP8_GRID=False,
            EMIT_FP8=True,
            cache_modifier=".cg",
            num_warps=num_warps,
            **tuning,
        )
        return Mxfp8Activation(q, scale)
    partial = torch.empty(
        (2, split_k, row_tiles * block_m, 1024), dtype=torch.float32, device=x.device
    )
    _batched_gemm_bf16_split_k_partial_kernel[(2, row_tiles * 32, split_k)](
        x,
        weight,
        partial,
        m,
        1024,
        4096,
        *strides,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=block_k,
        SPLIT_K=split_k,
        cache_modifier=".cg",
        num_warps=num_warps,
        **tuning,
    )
    _batched_gemm_split_k_reduce_kernel[(2, row_tiles * 32)](
        partial,
        q,
        m,
        1024,
        1024,
        2048,
        scale,
        32,
        64,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=32,
        SPLIT_K=split_k,
        FP8_GRID=False,
        EMIT_FP8=True,
        num_warps=num_warps,
    )
    return Mxfp8Activation(q, scale)
