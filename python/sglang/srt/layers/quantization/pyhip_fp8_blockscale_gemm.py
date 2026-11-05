# SPDX-License-Identifier: MIT
# Ported from luocheng25/pyhip at cab0421b1bfe71d052519542d66d5bc5e3427415:
# src/pyhip/ops/gemm/flydsl/gemm_fp8_blockscale_8w_942.py. Only Torch and FlyDSL are
# required; importing this module does not load the PyHIP package.
# Upstream license follows.
# The MIT License (MIT)
# Copyright © 2026 <copyright holders>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# ruff: noqa: F841
# Preserve the vendored DSL's intermediate SSA expressions.

"""gfx942 FP8 block-scale GEMM, 256x256 CTA tile, 8 waves, staggered Memory/Compute stages.

Y[M, N] (BF16) = sum_kb (X[:, kb] @ W[:, kb]^T) * xs[kb, m] * ws[n // 128, kb]
X [M, K] is row-major FP8 e4m3fnuz, W [N, K] row-major or aiter shuffle_weight(16, 16), xs the
activation scale [K / 128, M] (FP32, token-contiguous; aiter's transposed x_scale) and ws the
weight scale [ceil(N / 128), K / 128] (FP32).

Waves 0-3 (G0) own tokens 0-127, waves 4-7 (G1) tokens 128-255 of the tile; inside a group
wave q covers tokens 64 * (q >> 1) .. + 63 and N 128 * (q & 1) .. + 127 (4x2 wave grid).
The MFMA is v_mfma_f32_16x16x32_fp8_fp8 on C^T = W X^T (A = W rows, B = X rows); a wave owns
8 N-blocks x 4 M-blocks of 16x16.

Per K-block of 128 every wave runs four stages; G1 runs one stage behind G0, so on every SIMD
one wave issues MFMAs while the other loads:

    M1  ds_read X (M-blocks 0-3) and W N-blocks 0-3; one 16 KiB LDS-DMA packet
    C1  64 MFMAs (N-blocks 0-3 x M-blocks 0-3, four K-steps per block)
    M2  ds_read W N-blocks 4-7; one LDS-DMA packet; scales of the next K-block
    C2  64 MFMAs (N-blocks 4-7)

Every 16x16 block's K-block partial is retired two blocks late: its 4 FP32 FMAs
(C += partial * xs * ws) run between the MFMAs of a later block.

LDS: four 16 KiB slots, rows of 128 bytes with the 16-byte chunk XOR-swizzled by row % 8
(conflict-free ds_read_b128; preshuffled W keeps aiter's layout): X0 (tokens 0-127), X1 (tokens
128-255), P1 (N-blocks 0-3 of both N halves) and P2 (N-blocks 4-7). DMA duties keep every
packet two stages in flight:

    stage   G0                                  G1
    t       M1: read X0, P1; DMA P2(kb)         C2(kb-1)
    t+1     C1                                  M1: read X1, P1; DMA X0(kb+1)
    t+2     M2: read P2; DMA P1(kb+1)           C1
    t+3     C2                                  M2: read P2; DMA X1(kb+1)

Scheduling: full rounds of tiles run in XCD-chunked grouped order. A last partial round of at
most half the CUs is split K-wise into S >= 4 pieces that fill the round; each piece writes an
FP32 partial and a second kernel sums the S partials in piece order (deterministic) to BF16.
"""

import logging
import time

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, rocdl

BM = BN = 256
BK = 128
THREADS = 512
SLOT = 16384
X0_SLOT, X1_SLOT, P1_SLOT, P2_SLOT = 0, SLOT, 2 * SLOT, 3 * SLOT
LDS_BYTES = 4 * SLOT
# Stores of invalid N-blocks go past every buffer bound and are dropped.
_DROP = 0x40000000


# ---- Generic gfx942 helpers (as in pyhip.ops.mha.flydsl._common; this module only needs FlyDSL) ----


def _uniform(value):
    return fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(value).ir_value()))


def _min(a, b):
    return (a < b).select(a, b)


def _pin_i32(value):
    return fx.Int32(
        llvm.inline_asm(
            fx.Int32.ir_type,
            [fx.Int32(value).ir_value()],
            "",
            "=v,0",
            has_side_effects=True,
        )
    )


def _stage_end():
    # Compiler fences and CTA rendezvous, not a memory-counter wait.
    rocdl.sched_barrier(0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)


def _wait(*, vmcnt=63, expcnt=7, lgkmcnt=63):
    # gfx94x s_waitcnt encoding; an omitted counter is not waited on.
    rocdl.s_waitcnt(
        (vmcnt & 15) | (expcnt << 4) | (lgkmcnt << 8) | ((vmcnt >> 4) << 14)
    )


# FMAs after each of a 16x16 block's four MFMAs, retiring the partial of the block two before it (by then
# its last MFMA has long completed: no hazard wait).
_RETIRE = (1, 1, 1, 1)


def _schedule(blocks, group):
    """Per block: each MFMA followed by _RETIRE[step] VALU ops."""
    for _ in range(blocks):
        for count in _RETIRE:
            rocdl.sched_group_barrier(rocdl.mask_mfma, 1, group)
            if count:
                rocdl.sched_group_barrier(0x002, count, group)


def _buffer(tensor, size_bytes):
    """Raw buffer descriptor; loads at or past size_bytes return zero, stores are dropped."""
    address = fx.Int64(fx.ptrtoint(fx.get_iter(tensor)))
    pointer = llvm.inttoptr(ir.Type.parse("!llvm.ptr"), address.ir_value())
    return rocdl.make_buffer_rsrc(
        ir.Type.parse("!llvm.ptr<8>"),
        pointer,
        fx.Int16(0).ir_value(),
        fx.Int64(size_bytes).ir_value(),
        fx.Int32(0x27000).ir_value(),
    )


def _read_address(address, immediate=0):
    """One ds_read_b128 at a precomputed LDS byte address; asynchronous until lgkmcnt."""
    return fx.Vector(
        llvm.inline_asm(
            ir.VectorType.get([4], fx.Int32.ir_type),
            [address.ir_value()],
            f"ds_read_b128 $0, $1 offset:{immediate}",
            "=v,v,~{memory}",
            has_side_effects=True,
        )
    )


def _read_scalar_g2r(table, index):
    """s_load_dword of table[index]; _scalar_ready makes it usable."""
    address = fx.Int64(fx.ptrtoint(fx.get_iter(table) + index))
    return fx.Int32(
        llvm.inline_asm(
            fx.Int32.ir_type,
            [address.ir_value()],
            "s_load_dword $0, $1, 0",
            "=s,s,~{memory}",
            has_side_effects=True,
        )
    )


def _scalar_ready(value):
    # Keep the first read of the pending SMEM result inside the assembly, after its wait.
    return fx.Int32(
        llvm.inline_asm(
            fx.Int32.ir_type,
            [value.ir_value()],
            "s_waitcnt lgkmcnt(0)\ns_mov_b32 $0, $1",
            "=s,s",
            has_side_effects=True,
        )
    )


def _pack_bf16(values):
    # Software RNE FP32 -> BF16 (gfx942 has no packed conversion); one v_perm_b32 per pair.
    bits = fx.Vector(values).bitcast(fx.Uint32)
    rounded = bits + fx.Uint32(0x7FFF) + ((bits >> 16) & fx.Uint32(1))
    selector = fx.Int32(0x07060302)
    words = [
        fx.Uint32(
            llvm.inline_asm(
                fx.Int32.ir_type,
                [rounded[i + 1].ir_value(), rounded[i].ir_value(), selector.ir_value()],
                "v_perm_b32 $0, $1, $2, $3",
                "=v,v,v,s",
                has_side_effects=True,
            )
        )
        for i in range(0, values.numel, 2)
    ]
    return fx.Vector.from_elements(words, fx.Uint32)


# ---- GEMM pieces ----


def _dma(resource, storage, destination, voffset, soffset):
    """One buffer_load_dword ... lds: lane i writes LDS bytes destination + 4 i (wave-uniform M0)."""
    rocdl.raw_ptr_buffer_load_lds(
        resource,
        fx.to_llvm_ptr(fx.get_iter(storage) + destination),
        fx.Int32(4).ir_value(),
        fx.Int32(voffset).ir_value(),
        fx.Int32(soffset).ir_value(),
        fx.Int32(0).ir_value(),
        fx.Int32(0).ir_value(),
    )


def _dma_x(resource, storage, q, voffset, half, kb, LDA):
    """X rows 128 * half + 2 i, 2 i + 1 (i = q + 4 t) of K-block kb into slot X<half>: 16 DMAs per wave.
    LDA: bytes between X rows (K for a contiguous X)."""
    for t in range(16):
        _dma(
            resource,
            storage,
            half * SLOT + q * 256 + t * 1024,
            voffset,
            (128 * half + 2 * q + 8 * t) * LDA + kb * BK,
        )


def _dma_w(resource, storage, q, voffset, packet, kb, K, preshuffled=False):
    """W packet P1 (packet 0) or P2 (packet 1) of K-block kb: LDS row rho holds tile row
    (rho & 63) + 128 * (rho >> 6) + 64 * packet.

    preshuffled (aiter shuffle_weight 16x16: per 16-row group, each 16-byte K chunk of its 16 rows is
    256 contiguous bytes): LDS keeps that layout, group rho / 16 at 2 KiB, chunk c at 256 c.
    """
    slot = P1_SLOT if packet == 0 else P2_SLOT
    for t in range(16):
        if preshuffled:
            group = t >> 1
            row16 = (group & 3) + 8 * (group >> 2) + 4 * packet
            soffset = row16 * (16 * K) + (kb * 8 + q + 4 * (t & 1)) * 256
        else:
            row = 2 * q + 8 * (t & 7) + 128 * (t >> 3) + 64 * packet
            soffset = row * K + kb * BK
        _dma(resource, storage, slot + q * 256 + t * 1024, voffset, soffset)


def _read_frag(addresses, immediate):
    """Chunks 2 g + j (j = 0, 1; g = lane / 16) of the lane's row: K bytes 32 g .. 32 g + 31."""
    return [_read_address(addresses[j], immediate) for j in range(2)]


def _operand(words, step):
    """Eight FP8 values of MFMA K-step `step` (K bytes 32 g + 8 step .. + 7)."""
    return fx.Vector(words[step // 2]).bitcast(fx.Int64)[step % 2]


def _mfma(a, b, acc):
    return fx.Vector(
        rocdl.mfma_f32_16x16x32_fp8_fp8_(
            ir.VectorType.get([4], fx.Float32.ir_type),
            a.ir_value(),
            b.ir_value(),
            fx.Vector(acc).ir_value(),
            0,
            0,
            rocdl._blgp_attr(0),
        )
    )


def _fma(a, b, c):
    return fx.Float32(
        llvm.call_intrinsic(
            fx.Float32.ir_type,
            "llvm.fma.f32",
            [
                fx.Float32(a).ir_value(),
                fx.Float32(b).ir_value(),
                fx.Float32(c).ir_value(),
            ],
            [],
            [],
        )
    )


def _as_f32(bits):
    return fx.Vector.from_elements([fx.Int32(bits)], fx.Int32).bitcast(fx.Float32)[0]


def _compute(w, x, acc, queue, scales, nbase):
    """64 MFMAs: 16x16 blocks (N-block nbase + nbl, M-block mb), nbl, mb < 4; each block's partial is
    retired during the block two after it. acc[nb] holds the 4 M-blocks of N-block nb (4 values each).

    queue = the two unretired (partial, (nb, mb), scale), oldest first; returns the stage's last two.
    """
    queue = list(queue)
    for nbl in range(4):
        for mb in range(4):
            part, (pnb, pmb), scale = queue.pop(0)
            p = fx.Vector.filled(4, 0.0, fx.Float32)
            total = [acc[pnb][4 * pmb + e] for e in range(4)]
            e = 0
            for step in range(4):
                p = _mfma(_operand(w[nbl], step), _operand(x[mb], step), p)
                for _ in range(_RETIRE[step]):
                    total[e] = _fma(part[e], scale, total[e])
                    e += 1
            acc[pnb] = fx.Vector.from_elements(
                [
                    total[i - 4 * pmb] if 4 * pmb <= i < 4 * pmb + 4 else acc[pnb][i]
                    for i in range(16)
                ],
                fx.Float32,
            )
            queue.append((p, (nbase + nbl, mb), scales[mb]))
    return queue


def _load_xs(resource, voffset, soffset):
    """Activation scales of the lane's four tokens (M-blocks 0-3, 16 tokens apart)."""
    return [
        fx.Float32(
            rocdl.raw_ptr_buffer_load(
                fx.Float32.ir_type,
                resource,
                fx.Int32(voffset).ir_value(),
                fx.Int32(soffset + 64 * mb).ir_value(),
                aux=0,
            )
        )
        for mb in range(4)
    ]


def _write_b64(address, words):
    llvm.inline_asm(
        ir.Type.parse("!llvm.void"),
        [address.ir_value(), words.ir_value()],
        "ds_write_b64 $0, $1",
        "v,v,~{memory}",
        has_side_effects=True,
    )


def _read_b64(address, immediate=0):
    return fx.Vector(
        llvm.inline_asm(
            ir.VectorType.get([2], fx.Int32.ir_type),
            [address.ir_value()],
            f"ds_read_b64 $0, $1 offset:{immediate}",
            "=v,v,~{memory}",
            has_side_effects=True,
        )
    )


def _store_c(acc, shared, region, lane, y_res, y_row, n_lane, N):
    """BF16 RNE of the wave's 64 x 128 C^T, 32 tokens at a time through a wave-private 8 KiB LDS region.

    LDS row t (token) holds 128 N as 32 8-byte units, unit u at u ^ (t % 16): conflict-free b64 writes of
    16 rows and b64 reads along a row. Each store then covers two full 256-byte token rows.
    """
    r16, g = lane & 15, lane >> 4
    u, h = lane & 31, lane >> 5
    # Columns past N go past every buffer bound.
    store_lane = (n_lane + u * 4 < N).select((h * N + u * 4) * 2, fx.Int32(_DROP))
    read_base = shared + region + h * 256 + ((u ^ h) << 3)
    for half in range(2):
        for mbl in range(2):
            mb = 2 * half + mbl
            # Token row t = 16 mbl + lane % 16 of this half; unit 4 nb + g holds N 16 nb + 4 g .. + 3.
            row = shared + region + (16 * mbl + r16) * 256
            for nb in range(8):
                words = _pack_bf16(
                    fx.Vector.from_elements(
                        [acc[nb][4 * mb + e] for e in range(4)], fx.Float32
                    )
                )
                _write_b64(row + (((4 * nb + g) ^ r16) << 3), words.bitcast(fx.Int32))
        parts = [_read_b64(read_base ^ ((i & 7) << 4), i * 512) for i in range(16)]
        _wait(lgkmcnt=0)
        rocdl.sched_barrier(0)
        for i in range(16):
            rocdl.raw_ptr_buffer_store(
                parts[i].ir_value(),
                y_res,
                fx.Int32(store_lane).ir_value(),
                fx.Int32((y_row + half * 32 + 2 * i) * (N * 2) + n_lane * 2).ir_value(),
            )
        # The second half rewrites the region after these reads (in order within the wave).
        rocdl.sched_barrier(0)


def _store_partial(acc, p_res, slot, wave, lane):
    """FP32 partial of a split tail piece: per (slot, wave) 32 KiB, (nb, mb) chunks of 64 lanes x 16 bytes."""
    voffset = slot * 262144 + wave * 32768 + lane * 16
    for nb in range(8):
        for mb in range(4):
            v = fx.Vector.from_elements(
                [acc[nb][4 * mb + e] for e in range(4)], fx.Float32
            )
            rocdl.raw_ptr_buffer_store(
                v.ir_value(),
                p_res,
                fx.Int32(voffset).ir_value(),
                fx.Int32((nb * 4 + mb) * 1024).ir_value(),
            )


def _decode(v, tiles_m, TN):
    """Grouped order: 8 M rows per group, M fastest inside a group (X and W tiles shared in L2)."""
    group = v // (8 * TN)
    rows = _min(fx.Int32(8), tiles_m - group * 8)
    within = v - group * (8 * TN)
    return group * 8 + within % rows, within // rows


@flyc.jit
def _body(
    X,
    W,
    XS,
    WS,
    Y,
    P,
    storage,
    M,
    tile_m,
    tile_n,
    kb_lo,
    kb_end,
    slot,
    N: fx.Constexpr[int],
    K: fx.Constexpr[int],
    LDA: fx.Constexpr[int],
    PRESHUFFLE: fx.Constexpr[bool],
    SPLIT: fx.Constexpr[bool],
    GROUP: fx.Constexpr[int],
):
    # FlyDSL keys its disk cache on helpers named in this function's own code, not in closures.
    _helpers = (
        _dma,
        _dma_x,
        _dma_w,
        _read_frag,
        _operand,
        _mfma,
        _fma,
        _as_f32,
        _compute,
        _load_xs,
        _store_c,
        _write_b64,
        _read_b64,
        _buffer,
        _min,
        _pack_bf16,
        _scalar_ready,
        _pin_i32,
        _read_address,
        _read_scalar_g2r,
        _schedule,
        _stage_end,
        _uniform,
        _wait,
        _store_partial,
    )
    KB = K // BK
    tid = fx.Int32(gpu.thread_id("x"))
    wave = _uniform(tid >> 6)
    lane = tid & 63
    q = wave & 3
    wm, wn = (q >> 1) + 2 * GROUP, q & 1
    r, h = lane & 31, lane >> 5

    m_rows = _min(fx.Int32(BM), M - tile_m * BM)
    n_rows = _min(fx.Int32(BN), N - tile_n * BN)
    # Tile-relative X/W rows; rows past M/N read zero.
    x_res = _buffer(
        fx.make_view(
            fx.get_iter(X) + fx.Int64(tile_m) * (BM * LDA), fx.make_layout(BM * LDA, 1)
        ),
        m_rows * LDA,
    )
    w_res = _buffer(
        fx.make_view(
            fx.get_iter(W) + fx.Int64(tile_n) * (BN * K), fx.make_layout(BN * K, 1)
        ),
        n_rows * K,
    )
    xs_res = _buffer(XS, M * (KB * 4))
    y_res = _buffer(
        fx.make_view(
            fx.get_iter(Y) + fx.Int64(tile_m) * (BM * N), fx.make_layout(BM * N, 1)
        ),
        m_rows * (N * 2),
    )

    # Loop-invariant lane addresses: lane = 16 g + r16 reads row r16 of each 16-row block, K chunks 2 g + j.
    shared = fx.Int32(fx.ptrtoint(fx.get_iter(storage)))
    r16, g = lane & 15, lane >> 4
    x_read = [
        _pin_i32(
            shared
            + GROUP * SLOT
            + ((q >> 1) * 64 + r16) * 128
            + (((2 * g + j) ^ (r16 & 7)) << 4)
        )
        for j in range(2)
    ]
    w_read = [
        _pin_i32(shared + (wn * 64 + r16) * 128 + (((2 * g + j) ^ (r16 & 7)) << 4))
        for j in range(2)
    ]
    if fx.const_expr(PRESHUFFLE):
        # Packet group 4 wn + nbl at 2 KiB, chunk 2 g + j at 256 bytes, row r16 at 16 bytes.
        w_read = [
            _pin_i32(shared + wn * 8192 + (2 * g + j) * 256 + r16 * 16)
            for j in range(2)
        ]
    # DMA lane: row 2 i + h, 16-byte chunk (lane & 31) >> 2 of LDS row rho holds K chunk ^ (rho & 7),
    # and rho & 7 = 2 q + h for every instruction i = q + 4 t of this wave.
    dma_lane = _pin_i32(
        h * K + ((((lane & 31) >> 2) ^ ((q << 1) | h)) << 4) + ((lane & 3) << 2)
    )
    x_lane = dma_lane
    if fx.const_expr(LDA != K):
        x_lane = _pin_i32(
            h * LDA + ((((lane & 31) >> 2) ^ ((q << 1) | h)) << 4) + ((lane & 3) << 2)
        )
    # Preshuffled W: every DMA reads 256 contiguous bytes.
    w_lane = _pin_i32(lane * 4) if PRESHUFFLE else dma_lane
    xs_lane = _pin_i32((tile_m * BM + wm * 64 + r16) * 4)
    ws_row = _min(tile_n * 2 + wn, fx.Int32((N + 127) // 128 - 1)) * KB

    @flyc.jit
    def kblock(
        c0,
        c1,
        c2,
        c3,
        c4,
        c5,
        c6,
        c7,
        part0,
        scale0,
        part1,
        scale1,
        xs0,
        xs1,
        xs2,
        xs3,
        ws_bits,
        kb,
        LAST: fx.Constexpr[bool],
    ):
        acc = [fx.Vector(c) for c in (c0, c1, c2, c3, c4, c5, c6, c7)]
        kb = fx.Int32(kb)
        following = _min(kb + 1, fx.Int32(KB - 1))
        # M1: X and W N-blocks 0-3; G0: DMA P2(kb), G1: DMA X0(kb+1).
        rocdl.s_setprio(0)
        x = [_read_frag(x_read, mb * 2048) for mb in range(4)]
        w = [_read_frag(w_read, P1_SLOT + nbl * 2048) for nbl in range(4)]
        # This K-block's scales (loaded one K-block ahead) for both compute stages.
        ws = _as_f32(ws_bits)
        scales = tuple(fx.Float32(s) * ws for s in (xs0, xs1, xs2, xs3))
        if fx.const_expr(GROUP == 0):
            _dma_w(w_res, storage, q, w_lane, 1, kb, K, PRESHUFFLE)
        else:
            if fx.const_expr(not LAST):
                _dma_x(x_res, storage, q, x_lane, 0, kb + 1, LDA)
        _wait(lgkmcnt=0)
        _stage_end()
        # C1: N-blocks 0-3.
        rocdl.s_setprio(3)
        queue = _compute(
            w,
            x,
            acc,
            [
                (fx.Vector(part0), (7, 2), fx.Float32(scale0)),
                (fx.Vector(part1), (7, 3), fx.Float32(scale1)),
            ],
            scales,
            0,
        )
        _schedule(16, 0)
        rocdl.sched_barrier(0)
        if fx.const_expr(GROUP == 0):
            # P2(kb) for M2 of both groups.
            _wait(vmcnt=0)
        _stage_end()
        # M2: W N-blocks 4-7; G0: DMA P1(kb+1), G1: DMA X1(kb+1); scales of K-block kb+1.
        rocdl.s_setprio(0)
        w = [_read_frag(w_read, P2_SLOT + nbl * 2048) for nbl in range(4)]
        ws_request = _read_scalar_g2r(WS, ws_row + following)
        if fx.const_expr(not LAST):
            if fx.const_expr(GROUP == 0):
                _dma_w(w_res, storage, q, w_lane, 0, kb + 1, K, PRESHUFFLE)
            else:
                _dma_x(x_res, storage, q, x_lane, 1, kb + 1, LDA)
                # X0(kb+1) for G0's next M1; X1(kb+1) may stay in flight.
                _wait(vmcnt=16)
        xs_next = _load_xs(xs_res, xs_lane, following * (M * 4))
        _wait(lgkmcnt=0)
        ws_next = _scalar_ready(ws_request)
        _stage_end()
        # C2: N-blocks 4-7.
        rocdl.s_setprio(3)
        queue = _compute(w, x, acc, queue, scales, 4)
        _schedule(16, 1)
        rocdl.sched_barrier(0)
        _wait(vmcnt=0)
        if fx.const_expr(not (LAST and GROUP == 1)):
            # G1 skips its last barrier: G0 has no partner stage left.
            _stage_end()
        return (
            *acc,
            queue[0][0],
            queue[0][2],
            queue[1][0],
            queue[1][2],
            *xs_next,
            ws_next,
        )

    # Prologue: G0 loads P1(0), G1 X0(0) and X1(0); both the scales of K-block 0.
    if fx.const_expr(GROUP == 0):
        _dma_w(w_res, storage, q, w_lane, 0, kb_lo, K, PRESHUFFLE)
    else:
        _dma_x(x_res, storage, q, x_lane, 0, kb_lo, LDA)
        _dma_x(x_res, storage, q, x_lane, 1, kb_lo, LDA)
    ws_request = _read_scalar_g2r(WS, ws_row + kb_lo)
    xs_first = _load_xs(xs_res, xs_lane, kb_lo * (M * 4))
    _wait(vmcnt=0, lgkmcnt=0)
    ws_first = _scalar_ready(ws_request)
    _stage_end()
    if fx.const_expr(GROUP == 1):
        # G1 falls one stage behind.
        _stage_end()
    zero = fx.Vector.filled(16, 0.0, fx.Float32)
    zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
    state = (
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero4,
        fx.Float32(0.0),
        zero4,
        fx.Float32(0.0),
        *xs_first,
        ws_first,
    )
    for kb in range(kb_lo, kb_end - 1, fx.Int32(1)):
        state = kblock(*state, kb, False)
    state = kblock(*state, kb_end - 1, True)

    acc = [fx.Vector(c) for c in state[:8]]
    # The last two blocks (N-block 7, M-blocks 2 and 3) are still unretired.
    p2, s2, p3, s3 = (
        fx.Vector(state[8]),
        fx.Float32(state[9]),
        fx.Vector(state[10]),
        fx.Float32(state[11]),
    )
    acc[7] = fx.Vector.from_elements(
        [
            (
                _fma(p2[i - 8], s2, acc[7][i])
                if 8 <= i < 12
                else _fma(p3[i - 12], s3, acc[7][i]) if i >= 12 else acc[7][i]
            )
            for i in range(16)
        ],
        fx.Float32,
    )
    # Wave-private LDS regions: G0 (waves 0-3) reuses X0/X1, read for the last time two stages ago;
    # G1 reuses P1/P2, whose last readers finished before G1's last compute stage.
    if fx.const_expr(SPLIT):
        _store_partial(acc, _buffer(P, (slot + 1) * 262144), slot, wave, lane)
    else:
        _store_c(
            acc, shared, wave * 8192, lane, y_res, wm * 64, tile_n * BN + wn * 128, N
        )


@flyc.kernel(known_block_size=[THREADS, 1, 1])
def _gemm_kernel(
    X: fx.Tensor,
    W: fx.Tensor,
    XS: fx.Tensor,
    WS: fx.Tensor,
    Y: fx.Tensor,
    P: fx.Tensor,
    M: fx.Int32,
    TILES_M: fx.Int32,
    FULL: fx.Int32,
    S: fx.Int32,
    N: fx.Constexpr[int],
    K: fx.Constexpr[int],
    LDA: fx.Constexpr[int],
    XCD: fx.Constexpr[bool],
    PRESHUFFLE: fx.Constexpr[bool],
):
    storage = (
        fx.SharedAllocator()
        .allocate(fx.Array[fx.Int8, LDS_BYTES, 16])
        .peek()
        .view(fx.make_layout(LDS_BYTES, 1))
    )
    body = _body
    decode = _decode
    block = fx.Int32(gpu.block_id("x"))
    TN = (N + BN - 1) // BN
    KB = K // BK
    group = _uniform(fx.Int32(gpu.thread_id("x")) >> 8)
    if block < FULL:
        v = block
        if fx.const_expr(XCD):
            # Block b runs on XCD b % 8: give each XCD a contiguous chunk of the grouped order.
            v = (block & 7) * (FULL >> 3) + (block >> 3)
        tile_m, tile_n = decode(v, TILES_M, TN)
        if group != 0:
            body(
                X,
                W,
                XS,
                WS,
                Y,
                P,
                storage,
                M,
                tile_m,
                tile_n,
                fx.Int32(0),
                fx.Int32(KB),
                fx.Int32(0),
                N,
                K,
                LDA,
                PRESHUFFLE,
                False,
                1,
            )
        else:
            body(
                X,
                W,
                XS,
                WS,
                Y,
                P,
                storage,
                M,
                tile_m,
                tile_n,
                fx.Int32(0),
                fx.Int32(KB),
                fx.Int32(0),
                N,
                K,
                LDA,
                PRESHUFFLE,
                False,
                0,
            )
    else:
        # Tail tiles FULL.. split into S K-pieces; piece p writes partial slot p.
        p = block - FULL
        t = p // S
        i = p - t * S
        tile_m, tile_n = decode(FULL + t, TILES_M, TN)
        kb_lo = i * KB // S
        kb_end = (i + 1) * KB // S
        if group != 0:
            body(
                X,
                W,
                XS,
                WS,
                Y,
                P,
                storage,
                M,
                tile_m,
                tile_n,
                kb_lo,
                kb_end,
                p,
                N,
                K,
                LDA,
                PRESHUFFLE,
                True,
                1,
            )
        else:
            body(
                X,
                W,
                XS,
                WS,
                Y,
                P,
                storage,
                M,
                tile_m,
                tile_n,
                kb_lo,
                kb_end,
                p,
                N,
                K,
                LDA,
                PRESHUFFLE,
                True,
                0,
            )


@flyc.kernel(known_block_size=[64, 1, 1])
def _reduce_kernel(
    P: fx.Tensor,
    Y: fx.Tensor,
    M: fx.Int32,
    TILES_M: fx.Int32,
    FULL: fx.Int32,
    S: fx.Constexpr[int],
    N: fx.Constexpr[int],
):
    """Sum the S partials of one (tail tile, wave region, N-block) in piece order; direct BF16 stores."""
    _helpers = (_buffer, _decode, _min, _pack_bf16)
    block = fx.Int32(gpu.block_id("x"))
    lane = fx.Int32(gpu.thread_id("x"))
    t, w, nb = block >> 6, (block >> 3) & 7, block & 7
    TN = (N + BN - 1) // BN
    tile_m, tile_n = _decode(FULL + t, TILES_M, TN)
    q = w & 3
    wm, wn = (q >> 1) + 2 * (w >> 2), q & 1
    p_res = _buffer(P, (t + 1) * S * 262144)
    vt = ir.VectorType.get([4], fx.Float32.ir_type)
    base = t * S * 262144 + w * 32768 + lane * 16
    parts = [
        [
            fx.Vector(
                rocdl.raw_ptr_buffer_load(
                    vt,
                    p_res,
                    fx.Int32(base + i * 262144).ir_value(),
                    fx.Int32((nb * 4 + mb) * 1024).ir_value(),
                    aux=0,
                )
            )
            for mb in fx.range_constexpr(4)
        ]
        for i in fx.range_constexpr(S)
    ]
    state = []
    for mb in fx.range_constexpr(4):
        total = parts[0][mb]
        for i in fx.range_constexpr(1, S):
            total = total + parts[i][mb]
        state.append(total)
    m_rows = _min(fx.Int32(BM), M - tile_m * BM)
    y_res = _buffer(
        fx.make_view(
            fx.get_iter(Y) + fx.Int64(tile_m) * (BM * N), fx.make_layout(BM * N, 1)
        ),
        m_rows * (N * 2),
    )
    # Lane l holds token l % 16 of each M-block and N 4 (l / 16) .. + 3 of the N-block.
    n = tile_n * BN + wn * 128 + nb * 16 + (lane >> 4) * 4
    voffset = (n < N).select(((wm * 64 + (lane & 15)) * N + n) * 2, fx.Int32(_DROP))
    for mb in fx.range_constexpr(4):
        words = _pack_bf16(fx.Vector(state[mb]))
        rocdl.raw_ptr_buffer_store(
            words.bitcast(fx.Int32).ir_value(),
            y_res,
            fx.Int32(voffset).ir_value(),
            fx.Int32(mb * 16 * N * 2).ir_value(),
        )


@flyc.jit
def _launch(
    X: fx.Tensor,
    W: fx.Tensor,
    XS: fx.Tensor,
    WS: fx.Tensor,
    Y: fx.Tensor,
    P: fx.Tensor,
    M: fx.Int32,
    TILES_M: fx.Int32,
    FULL: fx.Int32,
    S: fx.Int32,
    GRID: fx.Int32,
    N: fx.Constexpr[int],
    K: fx.Constexpr[int],
    LDA: fx.Constexpr[int],
    XCD: fx.Constexpr[bool],
    PRESHUFFLE: fx.Constexpr[bool],
    stream: fx.Stream,
):
    _gemm_kernel(
        X,
        W,
        XS,
        WS,
        Y,
        P,
        M,
        TILES_M,
        FULL,
        S,
        N,
        K,
        LDA,
        XCD,
        PRESHUFFLE,
        value_attrs={
            "rocdl.waves_per_eu": 2,
            "llvm.target_features": ir.Attribute.parse(
                '#llvm.target_features<["-packed-fp32-ops"]>'
            ),
        },
    ).launch(grid=(GRID, 1, 1), block=(THREADS, 1, 1), stream=stream)


@flyc.jit
def _launch_reduce(
    P: fx.Tensor,
    Y: fx.Tensor,
    M: fx.Int32,
    TILES_M: fx.Int32,
    FULL: fx.Int32,
    GRID: fx.Int32,
    S: fx.Constexpr[int],
    N: fx.Constexpr[int],
    stream: fx.Stream,
):
    _reduce_kernel(P, Y, M, TILES_M, FULL, S, N).launch(
        grid=(GRID, 1, 1), block=(64, 1, 1), stream=stream
    )


_compiled = {}
_cus = {}
_partials = {}
# Bytes of one split piece's FP32 partial (8 waves x 64 x 128 values).
PIECE_BYTES = 262144


def _plan(tiles, kb, cus):
    """(full, tail, S): whole rounds of tiles, then `tail` tiles split into S K-pieces filling the last round.

    Only worth it for a sparse last round and pieces of >= 5 K-blocks (fixed cost: prologue, partial
    store, reduce).
    """
    tail = tiles % cus
    if tail == 0 or 2 * tail > cus:
        return tiles, 0, 1
    s = min(cus // tail, 8, kb // 5)
    return (tiles - tail, tail, s) if s >= 4 else (tiles, 0, 1)


def _partial_buffer(device, cus):
    # Sized for the largest split (a full round of pieces) once, so later calls never reallocate
    # (CUDA-graph capture safe).
    buf = _partials.get(device)
    if buf is None:
        buf = _partials[device] = torch.empty(
            cus * PIECE_BYTES // 4, dtype=torch.float32, device=device
        )
    return buf


def gemm_a8w8_blockscale_942(x, w, x_scale_t, w_scale, out=None, w_preshuffled=None):
    """Y = dequant(x) @ dequant(w)^T in BF16.

    x [M, K]: FP8 e4m3fnuz, row-major with unit column stride and a row stride lda = x.stride(0) >= K,
    lda % 16 == 0 (e.g. the activation view of a packed FP8 + scale all-gather). w [N, K]: FP8 e4m3fnuz,
    row-major or aiter shuffle_weight(w, layout=(16, 16)) (w_preshuffled; default: w.is_shuffled).
    x_scale_t: [K / 128, M] FP32 contiguous (token-contiguous activation scales; aiter's transposed
    x_scale). w_scale: [ceil(N / 128), K / 128] FP32 contiguous. K % 128 == 0, N % 32 == 0.
    """
    m, k = x.shape
    n = w.shape[0]
    if w_preshuffled is None:
        w_preshuffled = bool(getattr(w, "is_shuffled", False))
    if k % BK or n % 32 or w.shape[1] != k:
        raise ValueError("needs K % 128 == 0, N % 32 == 0 and matching K")
    if x_scale_t.numel() != (k // BK) * m or w_scale.numel() != ((n + 127) // 128) * (
        k // BK
    ):
        raise ValueError("scale shapes do not match")
    for t in (w, x_scale_t, w_scale):
        if not t.is_contiguous():
            raise ValueError("inputs must be contiguous")
    lda = x.stride(0) if m > 1 else k
    if x.stride(1) != 1 or lda < k or lda % 16:
        raise ValueError(
            "x needs unit column stride and a row stride >= K, multiple of 16"
        )
    if out is None:
        out = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
    if m == 0:
        return out
    device = x.device
    cus = _cus.get(device)
    if cus is None:
        cus = _cus[device] = torch.cuda.get_device_properties(
            device
        ).multi_processor_count
    tiles_m = (m + BM - 1) // BM
    tiles = tiles_m * ((n + BN - 1) // BN)
    full, tail, s = _plan(tiles, k // BK, cus)
    xcd = full % 8 == 0
    partials = _partial_buffer(device, cus)
    stream = torch.cuda.current_stream(device)
    # X as flat bytes from its first element through the last row's K bytes.
    x_flat = x.view(torch.uint8).as_strided(((m - 1) * lda + k,), (1,))
    args = (
        x_flat,
        w.view(torch.uint8).view(-1),
        x_scale_t.view(-1),
        w_scale.view(-1),
        out.view(-1),
        partials,
        m,
        tiles_m,
        full,
        s,
        full + tail * s,
        n,
        k,
        lda,
        xcd,
        bool(w_preshuffled),
        stream,
    )
    key = (device, n, k, lda, xcd, bool(w_preshuffled))
    compiled = _compiled.get(key)
    if compiled is None:
        # flyc.compile also launches this first call.
        started = time.perf_counter()
        _compiled[key] = flyc.compile(_launch, *args)
        logging.getLogger(__name__).info(
            "Initialized PyHIP FP8 GEMM M=%d N=%d K=%d lda=%d xcd=%s in %.3f s",
            m,
            n,
            k,
            lda,
            xcd,
            time.perf_counter() - started,
        )
    else:
        compiled(*args)
    if tail:
        args = (partials, out.view(-1), m, tiles_m, full, tail * 64, s, n, stream)
        key = ("reduce", device, n, s)
        compiled = _compiled.get(key)
        if compiled is None:
            started = time.perf_counter()
            _compiled[key] = flyc.compile(_launch_reduce, *args)
            logging.getLogger(__name__).info(
                "Initialized PyHIP FP8 split reduction N=%d split=%d in %.3f s",
                n,
                s,
                time.perf_counter() - started,
            )
        else:
            compiled(*args)
    return out
