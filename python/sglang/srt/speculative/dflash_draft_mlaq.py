# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Portions adapted from vLLM (https://github.com/vllm-project/vllm, Apache-2.0) at commit
# 642c18d64c4cb6db3ded8fbe1786718cd4005fcb (PR #61867):
# vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/mla/query.py (MlaQuery.tiled, MlaQuery.insert),
# also shipped in sgl-nvidia-mega as sgl_nvidia_mega/kimi_k3/cutedsl/mla/query.py.
# Modified by NVIDIA in 2026: the DFlash drafter's MLA query block (RoPE, FP8 KV row, k / v outputs, wide-SM schedules).
"""DFlash drafter: the MLA query block of one draft layer in one kernel (DraftMlaQuery).

Replaces the served five-launch chain behind the fused qkv_a GEMM (q_a RMSNorm, kv_a RMSNorm, q_b split-K GEMM,
W_UK bmm, RoPE + q concat + FP8 KV-row store) with one clustered CuTe kernel:

  inputs  qkv [rows, >= 2112] BF16 (q_a 1536 | kv_a 512 | k_pe 64), read after the PDL wait;
          static weights (q_a / kv_a RMSNorm gammas, Q_B [heads * 192, 1536], W_UK [heads, 128, 512]) staged before it;
  outputs q [rows, heads, 576] BF16 (W_UK-absorbed nope 512 | rotated rope 64, unscaled, as the served concat kernel),
          the FP8 KV row (normed kv_a 512 | rotated k_pe 64; plain e4m3 cast of the BF16 values, k_scale 1) at slots[row],
          and the same row as BF16 k [rows, 576] plus its latent part v [rows, 512] (the extend K / V the draft attention reads).

Rounding points are the served chain's (BF16 after each norm, after Q_B, after W_UK, after RoPE); RoPE is GPT-J (interleaved
pairs) in FP32 with the served contraction (x cos - y sin as fma(x, cos, -(y sin)), y cos + x sin as fma(y, cos, x sin)).
Accumulation order differs from the served kernels (numerics-changing at the ulp level).

The schedule is the tiled MlaQuery of the K3 megakernel plugin (vLLM's, see the header) with additions based on earlier
NVIDIA work: Q_B rows bulk-copied to shared memory before the PDL wait (``stage_qb``) and, for at
most 8 rows, the RMSNorm skipping the 16-row tile's dead upper rows (``half_tile``). The cluster / DSMEM exchange (Q_B
nope rows of 6 producer CTAs to every peer, one mbarrier) is vLLM MlaQuery's, unchanged.
"""

from __future__ import annotations

import logging

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int64

from sgl_nvidia_mega.kimi_k3.cutedsl.common.primitives import (
    alloc,
    at,
    bf16x2,
    copy_evict,
    copy_g2s,
    copy_s2cluster,
    fma,
    fma2,
    fp8_bits,
    launch,
    ldmatrix,
    load_bf16x8,
    load_words,
    mma_into,
    mul2,
    prefetch_l2,
    query_norm_rsqrt,
    smem_words,
    store_bf16_pair,
    store_shared_words,
    unpack_bf16,
    view,
    wait,
    warp_sum,
    zeros,
)

logger = logging.getLogger(__name__)

KP = 1536 + 32  # padded Q_B / x row (elements): 3136 B, conflict-free 16 B fragments
WIDE_SM_ARCHS = ("sm_107a",)  # wide-SM schedules (architecture() strings)
# Argument order: qkv, w_qa, w_kva, w_qb, w_uk, cache [pages, page, 576] u8, slots [rows] i64, eps, q_out [rows, heads, 576] bf16,
# positions [rows] i64, cos_sin [max_pos, 64] f32 (cos | sin halves), k_out [rows, 576] bf16, v_out [rows, 512] bf16, stream.
EPS, OUT, POS, COS_SIN, K_OUT, V_OUT = 7, 8, 9, 10, 11, 12
# Wide-SM schedule, 9-64 rows (the FUSED_NORM form): the RMSNorm pass only computes each row's rsqrt factor; x * rsqrt * gamma (BF16)
# is applied to the Q_B A fragments as they load (the same packed operations per element: bitwise to the write-back form).
FUSED_NORM_ROWS = range(9, 65)
# Wide-SM schedule, 6 heads, 17-32 rows (the WIDE_TOKENS form): two 16-request tiles (96 CTAs) instead of one of 32. Same
# arithmetic per row (bitwise).
# Wide-SM schedule, 6 heads, 97-128 rows (draft batches of 13-16 requests, reached only with SGLANG_OPT_DFLASH_DRAFT_MLAQ_MAX_ROWS > 96): 48-request tiles,
# so the grid is 3 tiles x 6 clusters of 8 = 18 clusters instead of 24. Same arithmetic per row (bitwise to 32-request tiles).
WIDE_TOKENS = {6: {**{rows: 16 for rows in range(17, 33)}, **{rows: 48 for rows in range(97, 129)}}}


@cute.jit
def rotate_pair(x, y, cos, sin):
    """GPT-J RoPE of one (x, y) pair in FP32, contracted as the served Triton kernel (rope.cuh's NVCC form)."""
    return fma(x, cos, -(y * sin)), fma(y, cos, x * sin)


class DraftMlaQuery:
    def __init__(
        self,
        heads: int,
        batch: int,
        *,
        arch: str = "sm_100a",
        tokens: int = 0,
        fused_norm: bool | None = None,
    ):
        self.heads, self.batch = heads, batch
        rubin = arch in WIDE_SM_ARCHS
        self.stage_qb = rubin
        self.half_tile = rubin and batch <= 8
        self.fused_norm = (rubin and batch in FUSED_NORM_ROWS) if fused_norm is None else bool(fused_norm)
        self.cluster = 8
        self.rows, self.latent = 192 // self.cluster, 512 // self.cluster
        default = 16 if batch <= 16 else 32 if batch <= 32 or heads < 8 else 64
        if rubin:
            default = WIDE_TOKENS.get(heads, {}).get(batch, default)
        self.tokens = tokens or default
        # CTA j of a tile inserts request j's KV row: the tile's CTAs must cover it; 16-row MMA tiles.
        assert self.tokens <= 8 * heads and self.tokens % 16 == 0, (heads, self.tokens)
        self.grid = (self.cluster, heads, (batch + self.tokens - 1) // self.tokens)
        self.options = dict(cluster=self.cluster, preferred_smem_carveout=100 if batch > 1 else None)

    def form(self) -> str:
        return (f"DraftMlaQuery heads {self.heads} batch {self.batch} tokens {self.tokens} grid {self.grid} stage_qb {self.stage_qb} "
                f"half_tile {self.half_tile} fused_norm {self.fused_norm}")

    @cute.jit
    def __call__(self, *args):
        launch(self.tiled, args, self.grid, 256, **self.options)

    @cute.jit
    def live(self, token):
        return token < self.batch

    @cute.jit
    def insert(self, args, ss, tok0, j, valid, tid):
        """RMSNorm request tok0 + j's latent KV row and store it with its rotated k_pe (FP8) at its slot."""
        qkv, cache, slots, eps = args[0], args[5], args[6], args[EPS]
        w_kva = args[2]
        positions, cos_sin, k_out, v_out = args[POS], args[COS_SIN], args[K_OUT], args[V_OUT]
        lane, warp = tid % 32, tid // 32
        slot = Int64(-1)
        if valid:
            slot = slots[tok0 + j]
        if slot >= 0:
            xx = cute.make_rmem_tensor(8, Float32)
            total = Float32(0)
            if tid < 64:
                values = load_bf16x8(at(qkv, tok0 + j, 1536 + tid * 8), mutable=True)
                for i in cutlass.range_constexpr(8):
                    xx[i] = values[i]
                    total = fma(values[i], values[i], total)
            total = warp_sum(total)
            cute.arch.sync_threads()
            if (lane == 0) & (warp < 2):
                ss[warp, 0] = total
            cute.arch.sync_threads()
            page, offset = slot // cache.shape[1], slot % cache.shape[1]
            if tid < 64:
                mean = fma(ss[0, 0] + ss[1, 0], Float32(1 / 512), eps)
                rr = cute.math.rsqrt(mean, fastmath=True)
                weights = load_bf16x8(w_kva.iterator + tid * 8)
                normed = [Float32(BFloat16(xx[i] * rr * weights[i])) for i in range(8)]
                for i in cutlass.range_constexpr(8):
                    cache[page, offset, tid * 8 + i] = fp8_bits(normed[i])
                for i in cutlass.range_constexpr(0, 8, 2):  # exact: the values are BF16 already
                    store_bf16_pair(at(k_out, tok0 + j, tid * 8 + i), normed[i], normed[i + 1])
                    store_bf16_pair(at(v_out, tok0 + j, tid * 8 + i), normed[i], normed[i + 1])
            elif tid < 72:
                c = tid - 64  # k_pe elements 8c..8c+7 = GPT-J pairs 4c..4c+3
                values = load_bf16x8(at(qkv, tok0 + j, 2048 + c * 8), mutable=True)
                position = positions[tok0 + j]
                for q in cutlass.range_constexpr(4):
                    x, y = values[2 * q], values[2 * q + 1]
                    k = c * 4 + q
                    x, y = rotate_pair(x, y, cos_sin[position, k], cos_sin[position, 32 + k])
                    cache[page, offset, 512 + c * 8 + 2 * q] = fp8_bits(Float32(BFloat16(x)))
                    cache[page, offset, 512 + c * 8 + 2 * q + 1] = fp8_bits(Float32(BFloat16(y)))
                    store_bf16_pair(at(k_out, tok0 + j, 512 + c * 8 + 2 * q), x, y)

    @cute.kernel
    def tiled(self, args):
        """T requests per CTA. Q_B rows staged before the PDL wait (wide-SM schedule: bulk copies to shared memory); in-place RMSNorm of
        the staged x rows, split-K Q_B MMAs over the warps, nope rows gathered from the six producing CTAs by bulk DSMEM
        copies (into the dead x rows), rope rows rotated and stored, then W_UK MMAs (warp w owns latent columns 8w..8w+7)."""
        qkv, w_qa, w_qb, w_uk = args[0], args[1], args[3], args[4]
        eps, output = args[EPS], args[OUT]
        tid, _, _ = cute.arch.thread_idx()
        part, head, tok0 = cute.arch.block_idx()
        tok0 *= self.tokens
        smem = cutlass.utils.SmemAllocator()
        T, R, L = self.tokens, self.rows, self.latent
        MT, senders = T // 16, (128 + R - 1) // R
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        uk_head = head
        xq = alloc(smem, BFloat16, (T, KP), 128)
        uk = alloc(smem, BFloat16, (128, L + 8), 128)  # 144 B rows for ldmatrix
        qa = alloc(smem, BFloat16, 1536)
        ss = alloc(smem, Float32, (8, 2))
        rrs = alloc(smem, Float32, T) if self.fused_norm else None  # rsqrt per row
        barq = smem.allocate(cutlass.Uint64, 8)
        if cutlass.const_expr(self.stage_qb):
            qbs = alloc(smem, BFloat16, (R, KP), 128)  # padded rows as x
            barb = smem.allocate(cutlass.Uint64, 8)
        # After the Q_B MMAs the x rows hold the split-K partials, then our nope rows and the six producers' (peers write
        # them only after our cluster arrival).
        red = view(xq.iterator, 0, Float32, (8, T * R // 32, 32), (T * R, 32, 1))
        qblk = view(xq.iterator, 8 * T * R * 4, BFloat16, (T, R), (R, 1))
        qn = view(xq.iterator, 9 * T * R * 4, BFloat16, (senders, T, R), (T * R, R, 1))
        if tid == 0:
            cute.arch.mbarrier_init(barq, 1)
            if cutlass.const_expr(self.stage_qb):
                cute.arch.mbarrier_init(barb, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.mbarrier_arrive_and_expect_tx(barq, senders * T * R * 2)
            if cutlass.const_expr(self.stage_qb):  # one bulk copy per Q_B row
                cute.arch.mbarrier_arrive_and_expect_tx(barb, R * 1536 * 2)
                for r in cutlass.range_constexpr(R):
                    src = w_qb.iterator + (head * 192 + part * R + r) * 1536
                    copy_evict(qbs.iterator + r * KP, src, size=3072, barrier=barb)
        # Split-K Q_B: warp w owns k = 192w..192w+191 (8 values per lane and 32-wide chunk; the x fragments use the same
        # permutation). L2 now, registers later (wide-SM schedule: from the staged rows).
        rows = head * 192 + part * R + group
        qb_at = [[at(w_qb, rows + 8 * n, (6 * warp + cc) * 32 + 8 * quarter) for n in range(3)] for cc in range(6)]
        if cutlass.const_expr(not self.stage_qb):
            for chunk in qb_at:
                for address in chunk:
                    prefetch_l2(address)
        for c in range(tid, 128 * L // 8, 256):
            p, j = c // (L // 8), c % (L // 8)
            dst = uk.iterator + p * (L + 8) + j * 8
            copy_evict(dst, at(w_uk, uk_head, p, part * L + j * 8))
        if tid < 192:
            copy_evict(qa.iterator + tid * 8, w_qa.iterator + tid * 8)
        cute.arch.cp_async_commit_group()
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        for c in range(tid, T * 192, 256):
            t, j = c // 192, c % 192
            if self.live(tok0 + t):
                copy_g2s(xq.iterator + t * KP + j * 8, at(qkv, tok0 + t, j * 8))
        cute.arch.cp_async_commit_group()
        # The KV insert overlaps the staging of x (CTA j of the tile owns request j).
        j = head * self.cluster + part
        self.insert(args, ss, tok0, j, (j < T) & self.live(tok0 + j), tid)
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        # Packed FP32 pairs: the same IEEE ops in the same order as the scalar form.
        gamma = []
        for i in cutlass.range_constexpr(6):
            gamma += list(load_bf16x8(qa.iterator + 256 * i + 8 * lane, shared=True))
        for pair in cutlass.range_constexpr(T // 16):  # requests warp + 8k, two at once
            xrows = [xq.iterator + (warp + 16 * pair + 8 * h) * KP for h in range(2)]
            values: list = []
            rr: list = []
            rows_n = 2 - int(self.half_tile)  # half_tile: rows warp + 8 are dead
            for h in cutlass.range_constexpr(rows_n):
                values.append([])
                for i in cutlass.range_constexpr(6):
                    for word in smem_words(xrows[h] + 256 * i + 8 * lane):
                        values[h] += list(unpack_bf16(word))
            for h in cutlass.range_constexpr(rows_n):
                zero = Float32(0), Float32(0)
                chains = [zero, zero]  # (0, 1), (2, 3): element e feeds chain e % 4
                for e in cutlass.range_constexpr(0, 48, 2):
                    x = values[h][e], values[h][e + 1]
                    chains[e % 4 // 2] = fma2(x, x, chains[e % 4 // 2])
                (c0, c1), (c2, c3) = chains
                rr.append(query_norm_rsqrt(warp_sum((c0 + c1) + (c2 + c3)), eps))
                if cutlass.const_expr(self.fused_norm):  # noqa: SIM102
                    if lane == 0:
                        rrs[warp + 16 * pair + 8 * h] = rr[h]  # type: ignore[index]
            for h in cutlass.range_constexpr(0 if self.fused_norm else rows_n):
                for i in cutlass.range_constexpr(6):
                    words = []
                    for e in cutlass.range_constexpr(8 * i, 8 * i + 8, 2):
                        x = values[h][e], values[h][e + 1]
                        y = mul2(mul2(x, (rr[h], rr[h])), (gamma[e], gamma[e + 1]))
                        words.append(bf16x2(*y))
                    store_shared_words(xrows[h] + 256 * i + 8 * lane, tuple(words))
        if cutlass.const_expr(self.stage_qb):
            wait(barb)
            qbr = [
                [smem_words(qbs.iterator + (group + 8 * n) * KP + (6 * warp + cc) * 32 + 8 * quarter) for n in range(3)]
                for cc in range(6)
            ]
        else:
            qbr = [[load_words(a, count=4) for a in chunk] for chunk in qb_at]
        cute.arch.sync_threads()
        acc = zeros((MT * 3, 4), Float32)
        for cc in cutlass.range_constexpr(6):
            k = (6 * warp + cc) * 32 + 8 * quarter
            z = qbr[cc]
            for m in cutlass.range_constexpr(MT):
                x, y = (load_words(xq.iterator + r * KP + k, count=4, shared=True) for r in (16 * m + group, 16 * m + group + 8))
                if cutlass.const_expr(self.fused_norm):  # x * rsqrt * gamma here
                    g = load_bf16x8(qa.iterator + k, shared=True)
                    x, y = (
                        self.scaled(words, self.rr_of(rrs, r), g)
                        for words, r in ((x, 16 * m + group), (y, 16 * m + group + 8))
                    )
                for n in cutlass.range_constexpr(3):
                    for s in cutlass.range_constexpr(2):
                        a = (x[2 * s], y[2 * s], x[2 * s + 1], y[2 * s + 1])
                        b = (z[n][2 * s], z[n][2 * s + 1])
                        mma_into(acc[m * 3 + n, None], a, b, bf16=True)
        cute.arch.sync_threads()
        cute.arch.cluster_arrive_relaxed()  # x is consumed: peers may fill qn
        for i in cutlass.range_constexpr(MT * 12):
            red[warp, i, lane] = acc[i // 4, i % 4]
        cute.arch.sync_threads()
        self.rope_rows(args, red, qblk, tok0, tid)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()
        cute.arch.cluster_wait()
        if (part < senders) & (tid < self.cluster):
            copy_s2cluster(qn.iterator + part * T * R, qblk.iterator, barq, tid, size=T * R * 2)
        wait(barq)
        cute.arch.cluster_arrive_relaxed()  # peers may exit once qblk is delivered
        b_frags = [ldmatrix(uk.iterator + (32 * kp + lane) * (L + 8) + 8 * warp, trans=True) for kp in range(4)]
        for m in cutlass.range_constexpr(MT):
            out = zeros(4, Float32)
            t = 16 * m + lane % 8 + 8 * ((lane // 8) % 2)
            for ks in cutlass.range_constexpr(8):
                p = 16 * ks + 8 * (lane // 16)
                a = ldmatrix(qn.iterator + ((p // R) * T + t) * R + p % R)
                b = b_frags[ks // 2][2 * (ks % 2) : 2 * (ks % 2) + 2]
                mma_into(out, a, b, bf16=True)
            for h in cutlass.range_constexpr(2):  # rows group, group + 8: one BF16 pair each
                row = 16 * m + group + 8 * h
                if self.live(tok0 + row):
                    col = L * part + 8 * warp + 2 * quarter
                    store_bf16_pair(at(output, tok0 + row, head, col), out[2 * h], out[2 * h + 1])
        cute.arch.cluster_wait()

    @cute.jit
    def rr_of(self, rrs, r):
        return rrs[r]

    @cute.jit
    def scaled(self, words, rr, g):
        """Four BF16-pair ``words`` of one x row scaled as the write-back norm stores them: BF16(x * rr * gamma) per element,
        packed pairs in the same order."""
        out = []
        for w in cutlass.range_constexpr(4):
            x = unpack_bf16(words[w])
            y = mul2(mul2(x, (rr, rr)), (g[2 * w], g[2 * w + 1]))
            out.append(bf16x2(*y))
        return tuple(out)

    @cute.jit
    def rope_rows(self, args, red, qblk, tok0, tid):
        """The Q_B row epilogue: each thread sums (warp order) and rounds an adjacent (even, odd) pair of rows of one request;
        nope pairs go to ``qblk``, rope pairs are rotated at the request's position (FP32) and stored as BF16."""
        part, head, _ = cute.arch.block_idx()
        output, positions, cos_sin = args[OUT], args[POS], args[COS_SIN]
        T, R = self.tokens, self.rows
        for e in range(tid, T * R // 2, 256):
            pair, ln = e // 32, e % 32
            i = 2 * pair  # elements i, i + 1: the same tile and token, rows r, r + 1
            v0, v1 = Float32(0), Float32(0)
            for w in cutlass.range_constexpr(8):
                v0 += red[w, i, ln]
            for w in cutlass.range_constexpr(8):
                v1 += red[w, i + 1, ln]
            v0, v1 = Float32(BFloat16(v0)), Float32(BFloat16(v1))
            tile, c = i // 4, i % 4
            t = 16 * (tile // 3) + ln // 4 + 8 * (c // 2)
            r = 8 * (tile % 3) + 2 * (ln % 4)
            p = R * part + r
            if p < 128:
                qblk[t, r] = BFloat16(v0)
                qblk[t, r + 1] = BFloat16(v1)
            elif self.live(tok0 + t):
                o0, o1 = v0, v1  # branch-local names (no join of a value rebound inside a dynamic branch)
                k = (p - 128) // 2
                position = positions[tok0 + t]
                o0, o1 = rotate_pair(v0, v1, cos_sin[position, k], cos_sin[position, 32 + k])
                store_bf16_pair(at(output, tok0 + t, head, 512 + p - 128), o0, o1)


def compile_draft_mlaq(
    heads, batch, *, cache_shape, max_pos, arch, pitch=2112, tokens=0, fused_norm=None, index_dtypes=None
):
    """Compile DraftMlaQuery for static shapes (qkv rows of ``pitch`` elements, a [pages, page, 576] FP8 pool, a [max_pos, 64]
    FP32 cos / sin cache). Returns (op, compiled); call compiled(qkv, w_qa, w_kva, w_qb, w_uk, cache, slots, eps, q_out,
    positions, cos_sin, k_out, v_out, stream) with cute tensors (the plugin's to_cute)."""
    import torch
    from cutlass.cute.runtime import make_fake_stream
    from cutlass.cutlass_dsl import CompileCallable
    from sgl_nvidia_mega.kimi_k3.cutedsl.common.host import fake

    op = DraftMlaQuery(heads, batch, arch=arch, tokens=tokens, fused_norm=fused_norm)
    slot_dtype, pos_dtype = index_dtypes or (torch.int64, torch.int64)
    index = lambda n, dt: fake((n,), (1,), dt, dt.itemsize)  # noqa: E731
    args = (
        fake((batch, 2112), (pitch, 1)),
        fake((1536,)),
        fake((512,)),
        fake((heads * 192, 1536)),
        fake((heads, 128, 512)),
        fake(tuple(cache_shape), None, torch.uint8),
        index(batch, slot_dtype),
        Float32(1e-6),
        fake((batch, heads, 576)),
        index(batch, pos_dtype),
        fake((max_pos, 64), None, torch.float32, 4),
        fake((batch, 576)),
        fake((batch, 512)),
        make_fake_stream(),
    )
    return op, CompileCallable()(op, *args, options=f"--gpu-arch={arch}")


# ------------------------------------------------------------------------------------------- serving (SGLANG_OPT_DFLASH_DRAFT_MLAQ)
MAX_ROWS = 64  # draft B8 x block 8; larger batches keep the served chain
# SGLANG_OPT_DFLASH_DRAFT_MLAQ_MAX_ROWS raises the cap for wider draft batches (9-16 requests x block 8): the same kernel (non-fused RMSNorm
# form above 64 rows), unit-checked at 72-128 rows; above the cap the served chain runs.
ROW_CAP_LIMIT = 128
_COMPILED: dict = {}


class DraftMlaQueryLayer:
    """One DFlash MLA draft layer's query block through DraftMlaQuery. ``run`` returns (q [T, heads, 576], k [T, 1, 576],
    v [T, 1, 512]) after storing the layer's FP8 KV rows at ``loc``, or None when this call keeps the served chain."""

    def __init__(self, attn, kv_rows, max_rows=MAX_ROWS):
        import torch
        from sgl_nvidia_mega.kimi_k3.cutedsl.common.host import architecture

        self.heads = attn.num_local_heads
        self.w_qa = attn.q_a_layernorm.weight
        self.w_kva = attn.kv_a_layernorm.weight
        self.w_qb = attn.q_b_proj.weight
        self.w_uk = attn.w_kc.contiguous()  # [heads, 128, 512] row-major (w_kc itself is stored [heads, 512, 128])
        self.eps = float(attn.q_a_layernorm.variance_epsilon)
        self.cos_sin = attn.rotary_emb.cos_sin_cache
        self.kv_rows = kv_rows  # this layer's [slots, 1, 576] uint8 pool rows
        self.max_rows = int(max_rows)
        device = self.w_qb.device
        self.arch = architecture(device)
        self.calls = 0

    def _code(self, rows, pitch, slot_dtype, pos_dtype):
        import torch
        from sgl_nvidia_mega.kimi_k3.cutedsl.common.host import prepare_shared_carveout

        key = (self.heads, rows, pitch, tuple(self.kv_rows.shape), tuple(self.cos_sin.shape), slot_dtype, pos_dtype, self.arch)
        code = _COMPILED.get(key)
        if code is None:
            if torch.cuda.is_current_stream_capturing():
                logger.warning("DFLASH DraftMlaQuery compiling rows %d pitch %d (%s / %s) during graph capture", rows, pitch, slot_dtype, pos_dtype)
            op, code = compile_draft_mlaq(
                self.heads, rows, cache_shape=self.kv_rows.shape, max_pos=self.cos_sin.shape[0], arch=self.arch,
                pitch=pitch, index_dtypes=(slot_dtype, pos_dtype),
            )
            code = prepare_shared_carveout(code, self.w_qb.device.index or 0, 100)
            _COMPILED[key] = code
            logger.info("DFLASH DraftMlaQuery compiled: %s pitch %d", op.form(), pitch)
        return code

    def run(self, qkv, positions, loc):
        import torch
        from sgl_nvidia_mega.kimi_k3.cutedsl.common.host import stream, to_cute

        rows = qkv.shape[0]
        if not (0 < rows <= self.max_rows) or qkv.shape[1] < 2112 or qkv.stride(1) != 1 or positions.shape != (rows,) or loc.shape != (rows,):
            return None
        code = self._code(rows, qkv.stride(0), loc.dtype, positions.dtype)
        q = torch.empty(rows, self.heads, 576, dtype=torch.bfloat16, device=qkv.device)
        k = torch.empty(rows, 576, dtype=torch.bfloat16, device=qkv.device)
        v = torch.empty(rows, 512, dtype=torch.bfloat16, device=qkv.device)
        code(
            to_cute(qkv[:, :2112]), to_cute(self.w_qa), to_cute(self.w_kva), to_cute(self.w_qb), to_cute(self.w_uk),
            to_cute(self.kv_rows), to_cute(loc, loc.element_size()), Float32(self.eps), to_cute(q),
            to_cute(positions, positions.element_size()), to_cute(self.cos_sin, 4), to_cute(k), to_cute(v), stream(qkv.device),
        )
        self.calls += 1
        return q, k.unsqueeze(1), v.unsqueeze(1)


def draft_mlaq_eligible(attn) -> str:
    """'' when DraftMlaQuery can serve this DFlashMLAAttention, else the reason it cannot."""
    import torch
    from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import _mla_fused_kv_write_folds_rope

    checks = (
        (attn.q_lora_rank == 1536, f"q_lora_rank {attn.q_lora_rank}"),
        (attn.kv_lora_rank == 512 and attn.qk_rope_head_dim == 64 and attn.qk_nope_head_dim == 128, "MLA dims"),
        (getattr(attn, "dcp_size", 1) == 1, "DCP"),
        (attn.w_kc is not None and attn.w_kc.dtype == torch.bfloat16, "w_kc dtype"),
        (attn.q_b_proj.weight.dtype == torch.bfloat16 and tuple(attn.q_b_proj.weight.shape) == (attn.num_local_heads * 192, 1536), "q_b weight"),
        (2 <= attn.num_local_heads and 16 <= 8 * attn.num_local_heads, "heads"),
        (_mla_fused_kv_write_folds_rope(attn.rotary_emb), "RoPE form (needs GPT-J, rotary 64, FP32 cos/sin)"),
        (float(attn.q_a_layernorm.variance_epsilon) == float(attn.kv_a_layernorm.variance_epsilon), "norm eps"),
        (not hasattr(attn, "prepare_attention_output_gate"), "attention output gate"),
    )
    return "; ".join(why for ok, why in checks if not ok)


def draft_mlaq_max_rows(value=None) -> int:
    """The row cap: SGLANG_OPT_DFLASH_DRAFT_MLAQ_MAX_ROWS (default 64); a multiple of 8 in [8, ROW_CAP_LIMIT], else ValueError."""
    if value is None:
        from sglang.srt.environ import envs

        value = envs.SGLANG_OPT_DFLASH_DRAFT_MLAQ_MAX_ROWS.get()
    cap = int(value)
    if cap % 8 or not 8 <= cap <= ROW_CAP_LIMIT:
        raise ValueError(f"SGLANG_OPT_DFLASH_DRAFT_MLAQ_MAX_ROWS={value}: needs a multiple of 8 in [8, {ROW_CAP_LIMIT}]")
    return cap


def draft_mlaq_wide_min_rows(value=None) -> int:
    """SGLANG_OPT_DFLASH_DRAFT_MLAQ_WIDE_MIN_ROWS: 0 (off) or a multiple of 8 in [16, ROW_CAP_LIMIT], else ValueError."""
    if value is None:
        from sglang.srt.environ import envs

        value = envs.SGLANG_OPT_DFLASH_DRAFT_MLAQ_WIDE_MIN_ROWS.get()
    lo = int(value)
    if lo and (lo % 8 or not 16 <= lo <= ROW_CAP_LIMIT):
        raise ValueError(f"SGLANG_OPT_DFLASH_DRAFT_MLAQ_WIDE_MIN_ROWS={value}: needs 0 or a multiple of 8 in [16, {ROW_CAP_LIMIT}]")
    return lo


def install_draft_mlaq(draft_model, pool, rows=None, max_rows=None) -> int:
    """SGLANG_OPT_DFLASH_DRAFT_MLAQ: give each eligible MLA draft layer a DraftMlaQueryLayer over its rows of ``pool`` (only when the
    Triton backend would store them through the fused plain FP8 write) and compile the row counts in ``rows`` (int64 indices, qkv
    pitch 2112; default 8, 16, ..., the row cap) before any graph capture. Calls on another backend, a row count above the cap
    (draft_mlaq_max_rows) or pool form keep the served chain.
    Returns the number of armed layers."""
    import torch
    from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import _mla_fused_kv_write_rows
    from sglang.srt.models.dflash import DFlashMLAAttention

    cap = draft_mlaq_max_rows(max_rows)
    wide_min = draft_mlaq_wide_min_rows()
    rows = tuple(range(8, cap + 1, 8)) if rows is None else tuple(rows)
    if wide_min:  # rows >= wide_min run DraftMlaQueryWide: no DraftMlaQuery compile for them
        rows = tuple(n for n in rows if n < wide_min)
    armed, skipped, wide_form = 0, [], ""
    for layer in getattr(draft_model, "layers", []):
        attn = getattr(layer, "self_attn", None)
        if not isinstance(attn, DFlashMLAAttention):
            continue
        why = draft_mlaq_eligible(attn)
        kv_rows = None
        if not why:
            kv_rows = _mla_fused_kv_write_rows(
                pool=pool, layer_id=attn.attn_mqa.layer_id, attention_backend="triton", dcp_size=attn.dcp_size, k_scale=attn.attn_mqa.k_scale
            )
            why = "" if kv_rows is not None else "pool form (needs a plain FP8 MLATokenToKVPool, no k_scale)"
        if why:
            skipped.append(f"layer {attn.layer_id}: {why}")
            continue
        attn._draft_mlaq = DraftMlaQueryLayer(attn, kv_rows, cap)
        for n in rows:
            attn._draft_mlaq._code(n, 2112, torch.int64, torch.int64)
        if wide_min:
            from sglang.srt.speculative.dflash_draft_mlaq_wide import DraftMlaQueryRouted, DraftMlaQueryWide

            wide = DraftMlaQueryWide(attn, kv_rows, ROW_CAP_LIMIT)
            wide_form = wide.form()
            attn._draft_mlaq = DraftMlaQueryRouted(attn._draft_mlaq, wide, wide_min - 8)
        armed += 1
    logger.info(
        "DFLASH DraftMlaQuery armed on %d MLA draft layers, rows <= %d%s", armed, cap, (" (skipped: " + "; ".join(skipped) + ")") if skipped else ""
    )
    if wide_min and armed:
        logger.info("DFLASH DraftMlaQueryWide armed on %d MLA draft layers, rows %d-%d: %s", armed, wide_min, ROW_CAP_LIMIT, wide_form)
    return armed
