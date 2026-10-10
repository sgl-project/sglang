from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.layernorm.mxfp8_epilogue import ue8m0_scale


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
def _wo_a_fp8_tile(WQ, WS, group, tile, k0):
    # BF16 [128 (k), 64 (n)] operand from E4M3 bytes and their 32x32 power-of-two
    # block scales; the product is exact, so this equals the dequantized weight.
    n = tile * 64 + tl.arange(0, 64)
    k = k0 + tl.arange(0, 128)
    w = tl.load(WQ + (group * 1024 + n[None, :]) * 4096 + k[:, None])
    kb = k0 // 32 + tl.arange(0, 4)
    nb = tile * 2 + tl.arange(0, 2)
    sb = tl.load(WS + (group * 32 + nb[None, :]) * 128 + kb[:, None])
    s = tl.reshape(
        tl.broadcast_to(tl.reshape(sb, (4, 1, 2, 1)), (4, 32, 2, 32)), (128, 64)
    )
    return (w.to(tl.float32) * s).to(tl.bfloat16)


@triton.jit(noinline=True)
def _wo_a_partial_bf16_fallback(X, W, P, M, SX, group, tile, split, BM: tl.constexpr):
    # Not inlined, and pipelined over two stages only, so this rarely taken path
    # adds no registers or shared memory to the E4M3 kernel.
    m = tl.arange(0, BM)
    n = tile * 64 + tl.arange(0, 64)
    k = split * 512 + tl.arange(0, 128)
    acc = tl.zeros((BM, 64), tl.float32)
    for i in tl.range(0, 4, num_stages=2):
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
def _wo_a_partial(
    X,
    W,
    WQ,
    WS,
    USE_WQ,
    P,
    M: tl.constexpr,
    SX: tl.constexpr,
    BM: tl.constexpr = 16,
    FP8_WEIGHT: tl.constexpr = False,
):
    tile, group, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    # A device flag, so captured CUDA graphs can fall back to the BF16 weight.
    if FP8_WEIGHT:
        if tl.load(USE_WQ) == 0:
            _wo_a_partial_bf16_fallback(X, W, P, M, SX, group, tile, split, BM)
            return
    m = tl.arange(0, BM)
    n = tile * 64 + tl.arange(0, 64)
    k = split * 512 + tl.arange(0, 128)
    acc = tl.zeros((BM, 64), tl.float32)
    for i in range(4):
        offsets = k + i * 128
        x = tl.load(
            X + m[:, None] * SX + group * 4096 + offsets[None, :], m[:, None] < M, 0
        )
        if FP8_WEIGHT:
            w = _wo_a_fp8_tile(WQ, WS, group, tile, split * 512 + i * 128)
        else:
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
        x, weight, None, None, None, partial, m, x.stride(0), num_warps=4, num_stages=3
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
    # FlashInfer's 128x4 layout: rows 32 apart share a 16-byte line.
    off = (col // 4) * 512 + (row % 32) * 16 + (row // 32) * 4 + col % 4
    tl.store(S + off, sf.to(tl.uint8))
    # Zero only padding rows; valid scale bytes have disjoint writers above.
    for z in tl.static_range(triton.cdiv(8192, M * 8 * 256)):
        s = (row * 8 + tile) * 256 + tl.arange(0, 256) + z * (M * 8 * 256)
        sr = (s % 512) // 16 + ((s % 16) // 4) * 32
        tl.store(S + s, 0, (s < 8192) & (sr >= M))


def _quantize_partial(p):
    m = p.shape[1]
    assert m <= 128  # the scale buffer is one 128-row swizzle tile
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
        x, weight, None, None, None, partial, m, x.stride(0), num_warps=4, num_stages=3
    )
    return _quantize_partial(partial)


# Rows the E4M3-weight kernels accept; torch.bmm on the BF16 weight catches up
# near 96 rows.
WO_A_FP8_MAX_ROWS = 64


def quantize_wo_a_fp8(weight: torch.Tensor):
    """E4M3 copy of a BF16 [2, 1024, 4096] WO-A weight with FP32 power-of-two
    32x32 block scales [2, 32, 128], or None unless the copy reproduces the weight
    bit for bit (as it does for a weight dequantized from a 32x32-block FP8
    checkpoint)."""
    if weight.shape != (2, 1024, 4096) or weight.dtype != torch.bfloat16:
        return None
    blocks = weight.view(64, 32, 128, 32)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True).float()
    scale = torch.where(amax > 0, torch.exp2(torch.ceil(torch.log2(amax / 448.0))), 1.0)
    # Power-of-two scaling is exact in BF16, so only the E4M3 cast can round.
    q = (blocks / scale.bfloat16()).to(torch.float8_e4m3fn)
    if not torch.equal(q.bfloat16() * scale.bfloat16(), blocks):
        return None
    return q.view(2, 1024, 4096), scale.view(2, 32, 128)


def _wo_a_fp8_partial(x, weight, weight_q, weight_scale, use_fp8):
    m = x.shape[0]
    assert 1 <= m <= WO_A_FP8_MAX_ROWS and x.shape[1:] == (2, 4096)
    assert weight.shape == weight_q.shape == (2, 1024, 4096)
    assert weight.is_contiguous() and weight_q.is_contiguous()
    assert weight_scale.shape == (2, 32, 128) and weight_scale.is_contiguous()
    assert x.dtype == weight.dtype == torch.bfloat16
    assert weight_q.dtype == torch.float8_e4m3fn
    assert weight_scale.dtype == torch.float32
    assert use_fp8.shape == (1,) and use_fp8.dtype == torch.int32
    assert x.is_cuda
    assert x.device == weight.device == weight_q.device == weight_scale.device
    assert x.stride(2) == 1 and x.stride(1) == 4096 and x.stride(0) >= 8192
    partial = torch.empty((8, m, 2, 1024), dtype=torch.float32, device=x.device)
    bm = max(16, triton.next_power_of_2(m))
    _wo_a_partial[(16, 2, 8)](
        x,
        weight,
        weight_q,
        weight_scale,
        use_fp8,
        partial,
        m,
        x.stride(0),
        BM=bm,
        FP8_WEIGHT=True,
        num_warps=4 if bm <= 32 else 8,
        num_stages=3,
    )
    return partial


def wo_a_fp8_small_batch(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_q: torch.Tensor,
    weight_scale: torch.Tensor,
    use_fp8: torch.Tensor,
) -> torch.Tensor:
    """WO-A for 1-64 rows from the quantize_wo_a_fp8 copy of the BF16 weight while
    the int32 device flag use_fp8 is nonzero, else from the BF16 weight itself.
    The flag lets a captured CUDA graph fall back when the weight changes. The BF16
    operands are identical, but a small fraction of outputs can differ from
    wo_a_bf16_small_batch by one BF16 ulp."""
    m = x.shape[0]
    partial = _wo_a_fp8_partial(x, weight, weight_q, weight_scale, use_fp8)
    result = torch.empty((m, 2, 1024), dtype=x.dtype, device=x.device)
    _wo_a_reduce[(triton.cdiv(m * 2048, 256),)](partial, result, m * 2048, num_warps=4)
    return result


def wo_a_fp8_small_batch_mxfp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_q: torch.Tensor,
    weight_scale: torch.Tensor,
    use_fp8: torch.Tensor,
):
    """wo_a_fp8_small_batch followed by FlashInfer-compatible MXFP8 quantization."""
    return _quantize_partial(
        _wo_a_fp8_partial(x, weight, weight_q, weight_scale, use_fp8)
    )
