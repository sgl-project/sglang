from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.mxfp8_epilogue import ue8m0_scale

# Measured on H100 against rope + einsum: wins through 16 tokens, loses from 32.
WO_A_FUSED_INV_ROPE_MAX_TOKENS = 16


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
def _wo_a_inv_rope_kernel(
    X, W, F, POS, P, Y, ARRIVALS, M, SX, G: tl.constexpr, BM: tl.constexpr
):
    # One program per (64-row N tile, group, head): split-K by head puts each
    # program's single 64-wide RoPE window in its last K step.
    tile, group, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m = tl.arange(0, BM)
    n = tile * 64 + tl.arange(0, 64)
    rows = m[:, None] < M
    x_ptr = X + m[:, None] * SX + group * 4096 + head * 512
    w_ptr = W + (group * 1024 + n[None, :]) * 4096 + head * 512
    acc = tl.zeros((BM, 64), tl.float32)
    for i in range(7):
        k = i * 64 + tl.arange(0, 64)
        x = tl.load(x_ptr + k[None, :], rows, 0)
        acc = tl.dot(x, tl.load(w_ptr + k[:, None]), acc)
    # Inverse RoPE on interleaved (re, im) pairs, rounded to bf16 like the
    # standalone fused_rope_inplace(inverse=True) output.
    pair = 448 + 2 * tl.arange(0, 32)
    re = tl.load(x_ptr + pair[None, :], rows, 0).to(tl.float32)
    im = tl.load(x_ptr + pair[None, :] + 1, rows, 0).to(tl.float32)
    pos = tl.load(POS + m, m < M, 0)
    freq = F + pos[:, None] * 64 + 2 * tl.arange(0, 32)[None, :]
    cos = tl.load(freq, rows, 0)
    sin = tl.load(freq + 1, rows, 0)
    rot_re = (re * cos + im * sin).to(tl.bfloat16)
    rot_im = (im * cos - re * sin).to(tl.bfloat16)
    rot = tl.join(rot_re, rot_im).reshape(BM, 64)
    k = 448 + tl.arange(0, 64)
    acc = tl.dot(rot, tl.load(w_ptr + k[:, None]), acc)

    out = (m[:, None] * G + group) * 1024 + n[None, :]
    split_stride = M * G * 1024
    tl.store(P + head * split_stride + out, acc, rows)
    tl.debug_barrier()
    # The eighth head to arrive reduces the tile. Counters are never reset, so
    # arrivals are taken mod 8; the mask stays correct across int32 wrap.
    arrived = tl.atomic_add(ARRIVALS + tile * G + group, 1, sem="acq_rel")
    if (arrived & 7) == 7:
        total = tl.zeros((BM, 64), tl.float32)
        for h in range(8):
            total += tl.load(P + h * split_stride + out, rows, 0, cache_modifier=".cg")
        tl.store(Y + out, total.to(tl.bfloat16), rows)


_wo_a_arrivals: dict[torch.device, torch.Tensor] = {}


def wo_a_bf16_fused_inv_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    freqs: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Inverse RoPE on each head's last 64 dims, then ``einsum('tgd,grd->tgr')``.

    x: [T, G, 4096] bf16, eight [448 nope | 64 rope] heads per group; the token
    stride may be padded. weight: [G, 1024, 4096] bf16. freqs: [max_pos, 64] fp32,
    interleaved (cos, sin). Returns [T, G, 1024] bf16.

    Launches on one device must not overlap: they share split-K arrival counters.
    """
    m = x.shape[0]
    groups = weight.shape[0]
    assert 1 <= m <= WO_A_FUSED_INV_ROPE_MAX_TOKENS and groups <= 8
    assert x.shape[1:] == (groups, 4096) and weight.shape == (groups, 1024, 4096)
    assert x.dtype == weight.dtype == torch.bfloat16 and weight.is_contiguous()
    assert x.stride(2) == 1 and x.stride(1) == 4096
    assert freqs.dtype == torch.float32 and freqs.shape[1] == 64
    assert freqs.is_contiguous() and positions.shape == (m,)
    arrivals = _wo_a_arrivals.get(x.device)
    if arrivals is None:
        arrivals = torch.zeros(16 * 8, dtype=torch.int32, device=x.device)
        _wo_a_arrivals[x.device] = arrivals
    result = torch.empty((m, groups, 1024), dtype=x.dtype, device=x.device)
    partial = torch.empty((8, m, groups, 1024), dtype=torch.float32, device=x.device)
    _wo_a_inv_rope_kernel[(16, groups, 8)](
        x,
        weight,
        freqs,
        positions,
        partial,
        result,
        arrivals,
        m,
        x.stride(0),
        groups,
        16,
        num_warps=4,
        num_stages=6,
    )
    return result
