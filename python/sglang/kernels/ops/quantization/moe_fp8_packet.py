"""Bit-preserving pack/unpack of FP8 activations, FP32 scales and routing.

This is a local format conversion, with no process-group/runtime dependency.
The byte packet stores q, scales, weights and int32 IDs per row, padded to 16B.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack(
    Q,
    S,
    W,
    I,
    P,
    M,
    H: tl.constexpr,
    K: tl.constexpr,
    SS0,
    SS1,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    q = tl.load(Q + row * H + col, col < H, 0).to(tl.uint32)
    sg = (col - H) // 4
    s = tl.load(S + row * SS0 + sg * SS1, (col >= H) & (col < H + H // 128 * 4), 0).to(
        tl.uint32, bitcast=True
    )
    wp = col - H - H // 128 * 4
    w = tl.load(W + row * K + wp // 4, (wp >= 0) & (wp < K * 4), 0).to(
        tl.uint32, bitcast=True
    )
    ip = wp - K * 4
    ids = tl.load(I + row * K + ip // 4, (ip >= 0) & (ip < K * 4), 0).to(tl.uint32)
    value = tl.where(
        col < H,
        q,
        tl.where(
            col < H + H // 128 * 4,
            (s >> ((col % 4) * 8)) & 255,
            tl.where(
                wp < K * 4,
                (w >> ((wp % 4) * 8)) & 255,
                tl.where(ip < K * 4, (ids >> ((ip % 4) * 8)) & 255, 0),
            ),
        ),
    )
    tl.store(P + row * WIDTH + col, value.to(tl.uint8), col < WIDTH)


@triton.jit
def _unpack(
    P,
    Q,
    S,
    W,
    I,
    M,
    H: tl.constexpr,
    K: tl.constexpr,
    WIDTH: tl.constexpr,
    SM,
    R: tl.constexpr,
    C: tl.constexpr,
    GS: tl.constexpr,
    KS: tl.constexpr,
):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    cols = tl.program_id(1) * C + tl.arange(0, C)
    q = tl.load(
        P + rows[:, None] * WIDTH + cols[None, :],
        (rows[:, None] < M) & (cols[None, :] < H),
        0,
    )
    tl.store(
        Q + rows[:, None] * H + cols[None, :],
        q,
        (rows[:, None] < M) & (cols[None, :] < H),
    )
    if tl.program_id(1) == 0:
        gs = tl.arange(0, GS)
        soff = rows[:, None] * WIDTH + H + gs[None, :] * 4
        smask = (rows[:, None] < M) & (gs[None, :] < H // 128)
        bits = tl.load(P + soff, smask, 0).to(tl.uint32)
        bits |= tl.load(P + soff + 1, smask, 0).to(tl.uint32) << 8
        bits |= tl.load(P + soff + 2, smask, 0).to(tl.uint32) << 16
        bits |= tl.load(P + soff + 3, smask, 0).to(tl.uint32) << 24
        tl.store(
            S + rows[:, None] + gs[None, :] * SM,
            bits.to(tl.float32, bitcast=True),
            smask,
        )
        ks = tl.arange(0, KS)
        mask = (rows[:, None] < M) & (ks[None, :] < K)
        off = rows[:, None] * WIDTH + H + H // 128 * 4 + ks[None, :] * 4
        wb = tl.load(P + off, mask, 0).to(tl.uint32)
        wb |= tl.load(P + off + 1, mask, 0).to(tl.uint32) << 8
        wb |= tl.load(P + off + 2, mask, 0).to(tl.uint32) << 16
        wb |= tl.load(P + off + 3, mask, 0).to(tl.uint32) << 24
        off += K * 4
        ib = tl.load(P + off, mask, 0).to(tl.uint32)
        ib |= tl.load(P + off + 1, mask, 0).to(tl.uint32) << 8
        ib |= tl.load(P + off + 2, mask, 0).to(tl.uint32) << 16
        ib |= tl.load(P + off + 3, mask, 0).to(tl.uint32) << 24
        tl.store(
            W + rows[:, None] * K + ks[None, :], wb.to(tl.float32, bitcast=True), mask
        )
        tl.store(
            I + rows[:, None] * K + ks[None, :], ib.to(tl.int32, bitcast=True), mask
        )


def pack_moe_fp8(q, scale, weights, ids):
    """Pack contiguous FP8/routing rows and arbitrarily strided FP32 scales."""
    if (
        q.ndim != 2
        or not q.is_cuda
        or q.dtype != torch.float8_e4m3fn
        or not q.is_contiguous()
        or q.shape[1] <= 0
        or q.shape[1] % 128
    ):
        raise ValueError("Expected contiguous CUDA FP8 [rows, hidden] with group 128")
    m, hidden = q.shape
    if weights.ndim != 2 or weights.shape[0] != m or weights.shape[1] <= 0:
        raise ValueError("Expected routing weights [rows, topk] with positive topk")
    if (
        scale.shape != (m, hidden // 128)
        or scale.dtype != torch.float32
        or weights.dtype != torch.float32
        or ids.dtype != torch.int32
        or ids.shape != weights.shape
        or not weights.is_contiguous()
        or not ids.is_contiguous()
        or any(t.device != q.device for t in (scale, weights, ids))
    ):
        raise ValueError("Invalid FP32 scale/weight or int32 routing ID layout")
    topk = weights.shape[1]
    width = triton.cdiv(hidden + hidden // 128 * 4 + topk * 8, 16) * 16
    payload = torch.empty((m, width), device=q.device, dtype=torch.uint8)
    if m:
        _pack[(m,)](
            q.view(torch.uint8),
            scale,
            weights,
            ids,
            payload,
            m,
            hidden,
            topk,
            *scale.stride(),
            width,
            triton.next_power_of_2(width),
            num_warps=8,
        )
    return payload


def unpack_moe_fp8(payload, hidden, topk):
    """Restore exact values and TMA-aligned column-major FP32 scales."""
    if (
        payload.ndim != 2
        or not payload.is_cuda
        or payload.dtype != torch.uint8
        or not payload.is_contiguous()
        or hidden <= 0
        or hidden % 128
        or topk <= 0
    ):
        raise ValueError("Invalid CUDA byte payload, hidden or topk")
    m, width = payload.shape
    if width != triton.cdiv(hidden + hidden // 128 * 4 + topk * 8, 16) * 16:
        raise ValueError("Payload width does not match hidden/topk")
    q = torch.empty((m, hidden), device=payload.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty(
        (hidden // 128, triton.cdiv(m, 4) * 4),
        device=payload.device,
        dtype=torch.float32,
    ).t()[:m]
    weights = torch.empty((m, topk), device=payload.device, dtype=torch.float32)
    ids = torch.empty_like(weights, dtype=torch.int32)
    if m:
        _unpack[(triton.cdiv(m, 4), triton.cdiv(hidden, 1024))](
            payload,
            q.view(torch.uint8),
            scale,
            weights,
            ids,
            m,
            hidden,
            topk,
            width,
            scale.stride(1),
            4,
            1024,
            triton.next_power_of_2(hidden // 128),
            triton.next_power_of_2(topk),
            num_warps=4,
        )
    return q, scale, weights, ids
