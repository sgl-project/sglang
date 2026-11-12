# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Portions adapted from vLLM (https://github.com/vllm-project/vllm, Apache-2.0) at commit
# 642c18d64c4cb6db3ded8fbe1786718cd4005fcb (PR #61867):
# vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/gemm/splitk.py (SplitKGemm) and gemm/projection.py,
# which adapt FlashInfer v0.7.0.post1 flashinfer/gemm/kernels/dense_bf16_gemm_sm100_splitk.py (Apache-2.0).
# The same SplitKGemm ships in sgl-nvidia-mega as sgl_nvidia_mega/kimi_k3/cutedsl/gemm/splitk.py.
# FlashInfer NOTICE: FlashInfer / Copyright 2025-2026 NVIDIA / Copyright 2023-2026 FlashInfer community (https://flashinfer.ai/)
# Modified by NVIDIA in 2026: the conv epilogue on SK3B.
# The parent kernel SplitKTmemGemm64 is the megakernel plugin's gemm/splitk_tmem64.py (NVIDIA, Apache-2.0), based on earlier
# NVIDIA work.
# SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B_CONV: the draft conv kernel projection [1792, 7168] (feature = side * 896 + tap * 448 + group) on SK3B
# (speculative/dflash_sk3b.py) with the SIDE-0 grouped conv in its epilogue, for the dedicated-mailbox (flags 32) forms.
#   issue_a: tile bidx = tap-0 rows [side 896 + 64 (bidx % 7), + 64) in tile rows 0..63 and the tap-1 rows (+ 448) in 64..127 (two {64, 64} boxes);
#   owner_of pairs tile rows j and 64 + j; the reduce-scatter sums are untouched (projection output == the unfused SK3B output).
#   finish: BF16 staging as SK3B's finish; every owner stores its projection rows (both taps) to C [rows, 1792]; side-0 owners compute, per 8-channel
#   chunk of an 8-token block (x[t - 1] carried in registers, 0 at the block's first token as the Triton mask),
#   out = fma(f32(base0[c]) + f32(d0[t, g]), f32(x[t, c]), (f32(base1[c]) + f32(d1[t, g])) * f32(x[t - 1, c])) -> BF16 RN into HC [rows, 7168]:
#   the Triton _dflash_grouped_conv_kernel's FP32 expression as compiled (its PTX: add.f32x2 x2, mul.f32x2, fma.rn.f32x2). An FP32 emulation of
#   this form equals the Triton output bit for bit (checked on GPU); the other contraction fma(c1, x[t - 1], c0 * x[t]) does not.
#   PREFETCH (DMB forms): every epilogue thread waits on the PDL dependency and issues its x loads (rows clamped into [0, rows - 8], owner index
#   clamped, so non-owner / side-1 / ragged-block threads read valid memory and discard it) before the MMA-end wait; the reduce protocol is SK3B's
#   reduce_dmb, copied unchanged around it.
# Sync: no new inter-CTA edge (the parent's reduce-scatter; the DMB protocol is SK3B's); the owner reads its own staged tile after the parent's
#   named barrier 3; x is read only by threads that executed griddepcontrol.wait (PTX 9.7.15.13); HC / C are generic stores read by successors
#   after their own wait.
# BALANCED mapping (used for policies with share >= 32, i.e. the N64D 104-128-row forms): tile bidx holds groups
#   [32 bidx, +32) of BOTH sides (rows 0-31 side-0 tap 0, 32-63 side-0 tap 1, 64-95 side-1 tap 0, 96-127 side-1 tap 1; four {64, 32} boxes of a 32-row-box
#   weight map) and owner k gets q = share / 4 groups with all four rows, so every owner CTA does q groups of the side-0 conv (the 64-row mapping above
#   leaves the side-1 tiles' owners idle). Bit-equal on GPU; faster at 112 / 128 rows in the drafter chain rig and not at 96, so 96 keeps the 64-row
#   mapping. Same arithmetic, same reduce; only which CTA computes which feature row and conv group.
# PACKED: the conv arithmetic of one token x 8 channels as packed
#   FP32 pairs (add.rn.f32x2 x2, mul.rn.f32x2, fma.rn.f32x2 per channel pair, as the Triton kernel's own PTX) in one inline-PTX block: the same
#   per-lane IEEE RN operations in the same order as the scalar form, so the output bits are unchanged by construction; half the FP issue slots.
from functools import cache

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Uint32
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

from sglang.srt.speculative import dflash_sk3b as sk  # noqa: E402
from sgl_nvidia_mega.kimi_k3.cutedsl.common.host import architecture  # noqa: E402
from sgl_nvidia_mega.kimi_k3.cutedsl.common.host import stream as current_stream  # noqa: E402
from sgl_nvidia_mega.kimi_k3.cutedsl.common.primitives import asm, at, bf16x2, bits_float, float_bits, load_vector, named_barrier, store_words, unpack_bf16  # noqa: E402
from sgl_nvidia_mega.kimi_k3.cutedsl.gemm import splitk_tmem64 as tm  # noqa: E402
from sgl_nvidia_mega.kimi_k3.cutedsl.gemm.projection import _DSL_COMPILE, matrix_view, set_oversized, smem_overflow, weight_map  # noqa: E402
from sgl_nvidia_mega.kimi_k3.cutedsl.gemm.splitk import SplitKTactic, _arm_cluster, _fence_tcgen05, _wait_cluster  # noqa: E402

GROUPS, GS, TAPS, BLOCK = 448, 16, 2, 8
SIDE = TAPS * GROUPS  # 896 features per side
TILES_PER_SIDE = GROUPS // 64  # 7
PREFETCH = 1  # DMB forms: x prefetch before the MMA-end wait; 0 = the non-prefetch form (the loads after the reduce)
PACKED = 1  # 1 = packed f32x2 conv arithmetic; 0 = the scalar form


@asm((Uint32,) * 4, pure=True)
def _conv8_pk(b0, b1, d0, d1, cur, prev):
    """BF16x2 words of out[e] = fma(b0[e] + d0, cur[e], (b1[e] + d1) * prev[e]) for e = 0..7 (channel pairs (2q, 2q + 1) as f32x2 lanes,
    word q = (low: e = 2q, high: e = 2q + 1))."""
    code = "{ .reg .b64 dd0, dd1, bp, c0, c1, xp, pp, m, r; .reg .f32 plo, phi; mov.b64 dd0, {$20, $20}; mov.b64 dd1, {$21, $21}; "
    for q in range(4):
        e = 2 * q
        code += f"mov.b64 bp, {{${4 + e}, ${5 + e}}}; add.rn.f32x2 c0, bp, dd0; "
        code += f"mov.b64 bp, {{${12 + e}, ${13 + e}}}; add.rn.f32x2 c1, bp, dd1; "
        code += f"mov.b64 xp, {{${22 + e}, ${23 + e}}}; mov.b64 pp, {{${30 + e}, ${31 + e}}}; "
        code += "mul.rn.f32x2 m, c1, pp; fma.rn.f32x2 r, c0, xp, m; "
        code += f"mov.b64 {{plo, phi}}, r; cvt.rn.bf16x2.f32 ${q}, phi, plo; "
    code += "}"
    operands = tuple(Float32(v) for v in (*b0, *b1, d0, d1, *cur, *prev))
    return code, ",".join(["=r"] * 4 + ["f"] * 34), operands


@asm(Float32, pure=True)
def _add_rn(a, b):
    return "add.rn.f32 $0, $1, $2;", "=f,f,f", (Float32(a), Float32(b))


@asm(Float32, pure=True)
def _mul_rn(a, b):
    return "mul.rn.f32 $0, $1, $2;", "=f,f,f", (Float32(a), Float32(b))


@asm(Float32, pure=True)
def _fma_rn(a, b, c):
    return "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f", (Float32(a), Float32(b), Float32(c))


@asm(Float32)
def _ld_shared_bf16f(address):
    code = "{ .reg .u32 r; ld.shared.u16 r, [$1]; shl.b32 r, r, 16; mov.b32 $0, r; }"
    return code, "=f,r,~{memory}", (Int32(address),)


@asm((Uint32,) * 2)
def _ld_shared_v2(address):
    code = "ld.shared.v2.b32 {$0, $1}, [$2];"
    return code, "=r,=r,r,~{memory}", (Int32(address),)


@asm(Int32, pure=True)
def _min_s32(a, b):
    return "min.s32 $0, $1, $2;", "=r,r,r", (Int32(a), Int32(b))


class ConvFinish:
    """Mixin before a TM64-family kernel class: kernel args + (base [2, 7168] of side 0, HC [rows, 7168]); out = (C, x, base, HC)."""

    def conv_init(self, prefetch=False, balanced=False, packed=False):
        assert self.cta_m == 128 and self.share % 16 == 0
        self.packed = bool(packed)
        self.prefetch = bool(prefetch)
        self.balanced = bool(balanced)
        assert not self.prefetch or self.dmb, "the prefetch form overrides the dedicated-mailbox reduce only"
        assert not self.balanced or self.prefetch, "the balanced mapping exists in the prefetch form only"

    @cute.experimental.jit
    def __call__(
        self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, bias: cute.Tensor,
        a_map: cutlass.Int64, base: cute.Tensor, hc: cute.Tensor, stream: _cuda.CUstream,
    ):  # fmt: skip
        self.kernel(a, b, c, bias, a_map, base, hc).launch(
            grid=(
                cute.ceil_div(a.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(self.smem_bytes),
            stream=stream,
            use_pdl=True,
        )

    @cute.experimental.kernel
    def kernel(
        self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, mS: cute.Tensor,
        a_map: cutlass.Int64, mBase: cute.Tensor, mHc: cute.Tensor,
    ):  # fmt: skip
        self.body(mA, mB, mS, a_map, (mC, mB, mBase, mHc))

    def issue_a(self, dst, barrier, a_map, k0, row0):
        if self.balanced:
            g0 = row0 // 128 * 32
            for qd in range(4):  # side-0 tap 0, side-0 tap 1, side-1 tap 0, side-1 tap 1 of groups [g0, g0 + 32)
                tm._tma_box(dst + qd * 32 * 128, barrier, a_map, k0, (qd // 2) * SIDE + (qd % 2) * GROUPS + g0)
        else:
            bidx = row0 // 128
            side = bidx // TILES_PER_SIDE
            f0 = side * SIDE + (bidx - side * TILES_PER_SIDE) * 64
            tm._tma_box(dst, barrier, a_map, k0, f0)
            tm._tma_box(dst + 64 * 128, barrier, a_map, k0, f0 + GROUPS)

    def owner_of(self, row, row0):
        if self.balanced:
            q = self.share // 4
            j = row % 32
            return j // q, (row // 32) * q + j % q
        hs = self.share // 2
        j = row % 64
        return j // hs, (row // 64) * hs + j % hs

    @cute.experimental.jit
    def finish(
        self, epi_tid: cutlass.Int32, totals: cute.Tensor, tile: cutlass.Int32, out,
        row0: cutlass.Int32, col0: cutlass.Int32, l_idx: cutlass.Int32,
    ):  # fmt: skip
        """The non-prefetch form: staging, projection store, then the side-0 conv with its x loads here."""
        mC, mX, mBase, mHc = out
        share: cutlass.Constexpr = self.share
        hs: cutlass.Constexpr = self.share // 2
        n: cutlass.Constexpr = self.n
        rr, g = epi_tid % share, epi_tid // share
        width: cutlass.Constexpr = n * share // 128
        for t in cutlass.range_constexpr(width):
            tm._st_shared_bf16(tile + ((width * g + t) * share + rr) * 2, totals[t])
        named_barrier(3, 128)
        bidx = row0 // 128
        side = bidx // TILES_PER_SIDE
        k = (row0 - bidx * 128) // share
        g0 = (bidx - side * TILES_PER_SIDE) * 64 + k * hs  # this owner's first group
        f0 = side * SIDE + g0  # its tap-0 feature
        columns = mC.layout.shape[1]
        dch: cutlass.Constexpr = hs // 8
        items: cutlass.Constexpr = n * 2 * dch
        for i in cutlass.range_constexpr((items + 127) // 128):
            idx = epi_tid + i * 128
            token = idx // (2 * dch)
            r = idx % (2 * dch)
            tp = r // dch
            c8 = r % dch
            if (idx < items) & (col0 + token < columns):
                words = tm._ld_shared_v4(tile + (token * share + tp * hs + 8 * c8) * 2)
                store_words(at(mC, f0 + tp * GROUPS + 8 * c8, col0 + token, l_idx), words)
        if side == 0:
            cute.arch.griddepcontrol_wait()  # x is producer data: read only by threads that waited
            cpt: cutlass.Constexpr = 2 * hs
            blocks: cutlass.Constexpr = n // BLOCK
            citems: cutlass.Constexpr = blocks * cpt
            cc = epi_tid % cpt
            gl = cc // 2
            ch0 = (g0 + gl) * GS + (cc % 2) * 8
            bw0 = load_vector(at(mBase, 0, ch0, 0))
            bw1 = load_vector(at(mBase, 1, ch0, 0))
            for i in cutlass.range_constexpr((citems + 127) // 128):
                idx = epi_tid + i * 128
                blk = idx // cpt
                tb = col0 + blk * BLOCK
                if (idx < citems) & (tb < columns):
                    xs = [load_vector(at(mX, tb + j, ch0, l_idx)) for j in range(BLOCK)]
                    self._conv_block(xs, bw0, bw1, tile, blk, share, hs, gl, mHc, tb, ch0, l_idx)

    def _conv_block(self, xs, bw0, bw1, tile, blk, share, hs, gl, mHc, tb, ch0, l_idx):
        """One 8-token block of one 8-channel chunk (straight-line code, traced inline; no dynamic control flow)."""
        b0 = []
        b1 = []
        for q in range(4):
            lo, hi = unpack_bf16(bw0[q])
            b0.extend((lo, hi))
            lo, hi = unpack_bf16(bw1[q])
            b1.extend((lo, hi))
        prev = [Float32(0.0)] * 8
        for j in range(BLOCK):
            t = blk * BLOCK + j
            d0 = _ld_shared_bf16f(tile + (t * share + gl) * 2)
            d1 = _ld_shared_bf16f(tile + (t * share + hs + gl) * 2)
            cur = []
            for q in range(4):
                lo, hi = unpack_bf16(xs[j][q])
                cur.extend((lo, hi))
            if self.packed:
                words = _conv8_pk(b0, b1, d0, d1, cur, prev)
                store_words(at(mHc, tb + j, ch0, l_idx), words)
                prev = cur
                continue
            res = []
            for e in range(8):
                c0, c1 = _add_rn(b0[e], d0), _add_rn(b1[e], d1)
                # the Triton kernel's: fma(c0, x[t], c1 * x[t - 1]) (its PTX: mul.f32x2 then fma.rn.f32x2; bit-equal on GPU)
                res.append(_fma_rn(c0, cur[e], _mul_rn(c1, prev[e])))
            words = tuple(bf16x2(res[2 * q], res[2 * q + 1]) for q in range(4))
            store_words(at(mHc, tb + j, ch0, l_idx), words)
            prev = cur

    @cute.experimental.jit
    def reduce_dmb(
        self, epi_tid: cutlass.Int32, bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr,
        mbox: Int32, bar_reduce, split_rank: cutlass.Int32, out,
        row0: cutlass.Int32, col0: cutlass.Int32, l_idx: cutlass.Int32,
    ):  # fmt: skip
        """dflash_sk3b.SplitKStream3b.reduce_dmb's protocol, unchanged; with self.prefetch the side-0 conv's x / base loads are issued before the
        MMA-end wait and the finish is inlined (so the loaded words stay in this function's registers); without it, the parent's path."""
        mC, mX, mBase, mHc = out
        split, share = self.split_k, self.share
        quads: cutlass.Constexpr = self.n // 4
        slot_bytes = quads * share * 16
        hs: cutlass.Constexpr = self.share // 4 if self.balanced else self.share // 2  # conv groups per owner
        n: cutlass.Constexpr = self.n
        cpt: cutlass.Constexpr = 2 * hs  # 8-channel chunks per token
        blocks: cutlass.Constexpr = n // BLOCK
        citems: cutlass.Constexpr = blocks * cpt
        iters: cutlass.Constexpr = (citems + 127) // 128
        columns = mC.layout.shape[1]
        bidx = row0 // 128
        side = bidx // TILES_PER_SIDE
        kq = _min_s32(split_rank, self.owners - 1)
        if cutlass.const_expr(self.balanced):
            g0 = bidx * 32 + kq * hs  # this owner's first group (clamped for non-owners)
        else:
            g0 = (bidx - side * TILES_PER_SIDE) * 64 + kq * hs  # this owner's first group (clamped for non-owners)
        cc = epi_tid % cpt
        gl = cc // 2
        ch0 = (g0 + gl) * GS + (cc % 2) * 8
        if (epi_tid == 0) & (split_rank < self.owners):
            _arm_cluster(bar_reduce, (split - 1) * slot_bytes)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        _fence_tcgen05()
        base = Int32(cute.make_tensor(tmem_base_ptr, cute.make_layout(1))[0])
        quadrant = epi_tid // 32
        xpf = []
        bw0 = None
        bw1 = None
        if cutlass.const_expr(self.prefetch):
            # base_kernel[0] is read BEFORE this thread's PDL wait: correct only because it is a load-time model parameter that no kernel
            # writes; if it is ever made per-step, move these two loads after griddepcontrol_wait.
            bw0 = load_vector(at(mBase, 0, ch0, 0))
            bw1 = load_vector(at(mBase, 1, ch0, 0))
            cute.arch.griddepcontrol_wait()  # x is producer data
            for i in cutlass.range_constexpr(iters):
                tb = _min_s32(col0 + (epi_tid + i * 128) // cpt * BLOCK, columns - BLOCK)
                xpf.append([load_vector(at(mX, tb + j, ch0, l_idx)) for j in range(BLOCK)])
        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        _fence_tcgen05()
        values = tm._load_row(base + (quadrant * 32 << 16), self.n)
        _fence_tcgen05(before=True)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        owner, r = self.owner_of(epi_tid, row0)
        slot = mbox + (split_rank * quads * share + r) * 16  # at the owner's address
        mailbox = cute.make_ptr(Float32, slot, cute.AddressSpace.smem, assumed_align=16)
        if owner == split_rank:
            for i in cutlass.range_constexpr(quads):
                bits = tuple(float_bits(values[4 * i + j]) for j in range(4))
                tm._st_shared_v4(slot + i * share * 16, bits)
        else:
            for i in cutlass.range_constexpr(quads):
                quad = tuple(values[4 * i + j] for j in range(4))
                tm.push_float(mailbox + i * share * 4, quad, bar_reduce, owner)
        if split_rank < self.owners:
            named_barrier(3, 128)  # this CTA's own slot
            _wait_cluster(bar_reduce, Int32(0))
            width: cutlass.Constexpr = self.n * share // 128
            rr, g = epi_tid % share, epi_tid // share
            totals = cute.make_rmem_tensor(width, Float32)
            for s in cutlass.range_constexpr(split):
                for v in cutlass.range_constexpr(width // 4):
                    offset = ((s * quads + width // 4 * g + v) * share + rr) * 16
                    words = tm._ld_shared_v4(mbox + offset)
                    for j in cutlass.range_constexpr(4):
                        part = bits_float(words[j])
                        if cutlass.const_expr(s == 0):
                            totals[4 * v + j] = part
                        else:
                            totals[4 * v + j] = totals[4 * v + j] + part
            tile = mbox + self.tile_offset
            if cutlass.const_expr(not self.prefetch):
                self.finish(epi_tid, totals, tile, out, row0 + split_rank * share, col0, l_idx)
            elif cutlass.const_expr(self.balanced):
                for t in cutlass.range_constexpr(width):
                    tm._st_shared_bf16(tile + ((width * g + t) * share + rr) * 2, totals[t])
                named_barrier(3, 128)
                chunk: cutlass.Constexpr = hs if hs < 8 else 8  # features per store (q = 4: 8-byte stores)
                nch: cutlass.Constexpr = hs // chunk
                pitems: cutlass.Constexpr = n * 4 * nch
                for i in cutlass.range_constexpr((pitems + 127) // 128):
                    idx = epi_tid + i * 128
                    token = idx // (4 * nch)
                    r3 = idx % (4 * nch)
                    qd = r3 // nch
                    c8 = r3 % nch
                    if (idx < pitems) & (col0 + token < columns):
                        addr = tile + (token * share + qd * hs + chunk * c8) * 2
                        if cutlass.const_expr(chunk == 4):
                            pw = _ld_shared_v2(addr)
                        else:
                            pw = tm._ld_shared_v4(addr)
                        store_words(at(mC, (qd // 2) * SIDE + (qd % 2) * GROUPS + g0 + chunk * c8, col0 + token, l_idx), pw)
                for i in cutlass.range_constexpr(iters):  # every owner: the side-0 conv of its hs groups
                    idx = epi_tid + i * 128
                    blk = idx // cpt
                    tb = col0 + blk * BLOCK
                    if (idx < citems) & (tb < columns):
                        self._conv_block(xpf[i], bw0, bw1, tile, blk, share, hs, gl, mHc, tb, ch0, l_idx)
            else:
                for t in cutlass.range_constexpr(width):
                    tm._st_shared_bf16(tile + ((width * g + t) * share + rr) * 2, totals[t])
                named_barrier(3, 128)
                f0 = side * SIDE + g0  # this owner's tap-0 feature (kq == split_rank for owners)
                dch: cutlass.Constexpr = hs // 8
                items: cutlass.Constexpr = n * 2 * dch
                for i in cutlass.range_constexpr((items + 127) // 128):
                    idx = epi_tid + i * 128
                    token = idx // (2 * dch)
                    r2 = idx % (2 * dch)
                    tp = r2 // dch
                    c8 = r2 % dch
                    if (idx < items) & (col0 + token < columns):
                        words = tm._ld_shared_v4(tile + (token * share + tp * hs + 8 * c8) * 2)
                        store_words(at(mC, f0 + tp * GROUPS + 8 * c8, col0 + token, l_idx), words)
                if side == 0:
                    for i in cutlass.range_constexpr(iters):
                        idx = epi_tid + i * 128
                        blk = idx // cpt
                        tb = col0 + blk * BLOCK
                        if (idx < citems) & (tb < columns):
                            self._conv_block(xpf[i], bw0, bw1, tile, blk, share, hs, gl, mHc, tb, ch0, l_idx)


class SK3BConv(ConvFinish, sk.SplitKStream3b):
    def __init__(self, *, prefetch=False, balanced=False, packed=False, **kw):
        sk.SplitKStream3b.__init__(self, **kw)
        self.conv_init(prefetch, balanced, packed)


@cache
def compile_sk3b_conv(n, split, slots, wstages, owners, k, arch, flags=0, prefetch=False, balanced=False, packed=False):
    def fake(leading, dtype=cutlass.BFloat16):
        shape = tuple(cute.sym_int() for _ in range(3))
        stride = (1, cute.sym_int(64), 0) if leading == 0 else (cute.sym_int(64), 1, 0)
        return make_fake_tensor(dtype, shape, stride, assumed_align=16)

    y = fake(0)
    kw = dict(xlsu=bool(flags & 1), bulk=bool(flags & 2), ov=bool(flags & 4), dmb=bool(flags & 32))
    kernel = SK3BConv(tactic=SplitKTactic(128, n, split, slots), k=k, wstages=wstages, owners=owners, prefetch=prefetch,
                      balanced=balanced, packed=packed, **kw)
    args = (fake(1), fake(1), y, y, cutlass.Int64(0), fake(1), fake(1), make_fake_stream())
    compiled = _DSL_COMPILE(kernel, *args, options=f"--gpu-arch={arch}")
    overflow = smem_overflow(compiled)
    if overflow is not None:
        raise ValueError(f"SK3B-conv {(128, n, split, slots, wstages)} owners {owners} uses more SMEM than its launch allocates ({overflow} B)")
    return compiled, kernel.smem_bytes


class SK3BConvProjection:
    """weight [1792, 7168] BF16 (the conv kernel projection); ``policies`` as SK3BProjection; run(x [rows, 7168], delta [rows, 1792], base0 [2, 7168],
    hc [rows, 7168]) -> delta = x W^T (SK3B arithmetic), hc = grouped conv side 0 of x with base0 + delta side 0."""

    def __init__(self, weight, policies, prefetch=None, balanced=None, packed=None):
        assert tuple(weight.shape) == (2 * SIDE, 7168) and weight.dtype == torch.bfloat16 and weight.stride(1) == 1
        self.weight, self.device, self.policies = weight, weight.device, dict(policies)
        self.prefetch = bool(PREFETCH) if prefetch is None else bool(prefetch)
        self.balanced = balanced  # None = per policy (share >= 32 in the prefetch form); True / False forces it (tests)
        self.packed = bool(PACKED) if packed is None else bool(packed)
        self.weight_view = matrix_view(weight)
        self.kernels, self.map, self.map32, self.row_map = {}, None, None, {}

    def prepare(self):
        arch = architecture(self.device)
        with torch.accelerator.device_index(self.device.index):
            for rows, pol in self.policies.items():
                n, split, slots, wstages, owners = pol[:5]
                flags = int(pol[5]) if len(pol) > 5 else 0
                pf = self.prefetch and bool(flags & 32)
                bal = pf and ((128 // owners) >= 32 if self.balanced is None else bool(self.balanced))
                code, smem = compile_sk3b_conv(n, split, slots, wstages, owners, 7168, arch, flags, pf, bal, self.packed)
                sk.allow_nonportable(code, self.device)
                if smem > sk.NORMAL_SMEM:
                    set_oversized(code, self.device)
                self.kernels[rows] = code
                self.row_map[rows] = 32 if bal else 64
            if 64 in self.row_map.values():
                self.map = weight_map(self.weight, 64)
            if 32 in self.row_map.values():
                self.map32 = weight_map(self.weight, 32)
        return self

    def tensor_maps(self):
        """The device tensors holding this projection's TMA weight descriptors (64-row and / or 32-row boxes)."""
        return [m for m in (self.map, self.map32) if m is not None]

    def run(self, x, delta, base0, hc):
        b = x.shape[0]
        ok = (b in self.kernels and tuple(x.shape) == (b, 7168) and tuple(delta.shape) == (b, 2 * SIDE) and tuple(base0.shape) == (TAPS, 7168)
              and tuple(hc.shape) == (b, 7168) and x.stride(1) == 1 and hc.stride(1) == 1 and base0.stride(1) == 1 and delta.is_contiguous()
              and base0.dtype == torch.bfloat16)
        ok = ok and b % 8 == 0  # whole 8-token blocks (the conv loop loads a block's rows unguarded)
        if not ok:
            raise ValueError(f"SK3B-conv x {tuple(x.shape)} delta {tuple(delta.shape)} base {tuple(base0.shape)} hc {tuple(hc.shape)}: prepared {sorted(self.kernels)}")
        y = matrix_view(delta, output=True)
        a_map = self.map32 if self.row_map[b] == 32 else self.map
        self.kernels[b](self.weight_view, matrix_view(x), y, y, cutlass.Int64(a_map.data_ptr()), matrix_view(base0), matrix_view(hc),
                        current_stream(self.device))
        return hc
