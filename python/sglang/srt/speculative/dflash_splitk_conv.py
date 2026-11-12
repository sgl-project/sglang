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
# Modified by NVIDIA in 2026: SplitKGemm with its act_fuse pair load and a conv epilogue, for the DFlash draft conv kernel projection.
"""SGLANG_OPT_DFLASH_DRAFT_SPLITK_WCONV (default off).

The draft conv kernel projection [1792, 7168] (feature = side * 896 + tap * 448 + group) at the SERVED split-K / OV-wide tactic of its row
count (8 rows: (128, 8, 7, 5); 16 rows: OV-wide (64, 16, 4, 11)) with the SIDE-0 grouped conv in the epilogue, so the separate
dflash_grouped_conv launch after the projection goes away at 8 / 16 rows (SK3B_CONV covers 24-64 rows).

  Weight load: SplitKGemm's act_fuse pair load (weight_warp_pair, unchanged): CTA row bidx loads, per 64-K atom, a half-height box of the
  tap-0 rows [side * 896 + j * h, + h) into tile rows [0, h) and the tap-1 rows (+ 448) into [h, 2h) (h = mma_m / 2, j = bidx % (448 / h),
  side = bidx // (448 / h)). Each output row's dot product sees the same K slices on the same cluster ranks, reduced at rank 0 in the same
  rank order and cast once (SplitKGemm.epilog_warp, unchanged up to the cast) -> the projection output is bitwise the served split-K /
  OV-wide output; only which CTA computes which feature row changes.
  Epilogue (owner = cluster rank 0, as SplitKGemm): the BF16 tile is staged in SMEM as act_fuse does (same TMEM load, same cast, named
  barrier 1); every owner stores both halves of its tile to C [rows, 1792]; side-0 owners then compute, per 8-channel chunk of an 8-token
  block (x[t - 1] = 0 at the block's first token, as the Triton kernel's mask),
  out = fma(f32(base0[c]) + f32(d0[t, g]), f32(x[t, c]), (f32(base1[c]) + f32(d1[t, g])) * f32(x[t - 1, c])) -> BF16 RN into HC [rows, 7168],
  with dflash_sk3b_conv._conv8_pk (the packed f32x2 form of the Triton kernel's own PTX sequence; bit-equal, checked on GPU).
Sync: no new inter-CTA edge (SplitKGemm's DSMEM reduce, unchanged). The staged tile is read only after named barrier 1 among the 128
  epilogue threads (act_fuse's precedent, served as SGLANG_OPT_DFLASH_DRAFT_SPLITK_ACT); x is producer data, read only by threads that
  executed griddepcontrol.wait (PTX 9.7.15.13); base0 is a load-time parameter; C / HC are generic stores, fenced generic -> async after the
  stores when proxy_fences (the projection's proxy fence), read by successors after their own wait.
PREFETCH (chosen over the plain form in an A/B on the target GPU): every epilogue thread waits on
  the PDL dependency and loads its conv chunk's x block and base0 words BEFORE the MMA-end wait (indices are valid on every CTA, so side-1 and
  non-owner CTAs load and discard, as SK3B_CONV's prefetch form); the conv then runs from registers after the reduce. Same arithmetic, same stores.
"""

from functools import cache

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Float32, Int32
from cutlass.cute import experimental as cute_ext
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

from sglang.srt.speculative.dflash_sk3b_conv import _conv8_pk, _min_s32  # noqa: E402
from sgl_nvidia_mega.kimi_k3.cutedsl.common.host import architecture  # noqa: E402
from sgl_nvidia_mega.kimi_k3.cutedsl.common.host import stream as current_stream  # noqa: E402
from sgl_nvidia_mega.kimi_k3.cutedsl.common.primitives import at, bf16x2, load_vector, store_words, unpack_bf16  # noqa: E402
from sgl_nvidia_mega.kimi_k3.cutedsl.gemm import splitk as skm  # noqa: E402
from sgl_nvidia_mega.kimi_k3.cutedsl.gemm.projection import _DSL_COMPILE, matrix_view, set_oversized, smem_overflow, weight_map  # noqa: E402
from sgl_nvidia_mega.kimi_k3.cutedsl.gemm.splitk import OWNER_RANK, SplitKGemm, SplitKTactic  # noqa: E402

GROUPS, GS, TAPS, BLOCK = 448, 16, 2, 8
SIDE = TAPS * GROUPS  # 896 features per side
PREFETCH = 1  # 1 = the prefetch form (default; the form the wiring runs); 0 = the plain form


class SplitKConvGemm(SplitKGemm):
    """SplitKGemm (fenced_trigger 0, the draft projections' setting) over the conv kernel projection with the side-0 conv epilogue."""

    def __init__(self, *, tactic: SplitKTactic, proxy_fences: bool = True, oversized: bool = False, prefetch: bool = False):
        SplitKGemm.__init__(self, tactic=tactic, fenced_trigger=0, proxy_fences=proxy_fences, act_fuse=False, oversized=oversized)
        assert tactic.mma_m in (64, 128) and tactic.mma_n % BLOCK == 0, tactic
        self.half = tactic.mma_m // 2
        assert GROUPS % self.half == 0 and self.half % 8 == 0
        self.tps = GROUPS // self.half  # CTA rows per side
        self.prefetch = bool(prefetch)
        assert (2 * self.half) * (tactic.mma_n // BLOCK) <= self.epilog_threads, "prefetch / conv: one chunk-block item per epilogue thread"

    @cute.experimental.jit
    def __call__(
        self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, bias: cute.Tensor,
        a_map: cutlass.Int64, base: cute.Tensor, hc: cute.Tensor, stream: _cuda.CUstream,
    ):  # fmt: skip
        self.kernel(a, b, c, bias, a_map, base, hc).launch(
            grid=(
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(self.smem_alloc),
            stream=stream,
            use_pdl=True,
        )

    @cute.experimental.kernel
    def kernel(
        self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, mBias: cute.Tensor,
        a_map: cutlass.Int64, mBase: cute.Tensor, mHc: cute.Tensor,
    ):  # fmt: skip
        """SplitKGemm.kernel with act_fuse's pair load (tap-0 / tap-1 boxes) and epilog_warp_conv; split > 1 only."""
        stages = self.num_ab_stage
        ab_dtype = mA.element_type
        majors = (utils.LayoutEnum.from_tensor(t).mma_major_mode() for t in (mA, mB))
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype, ab_dtype, *majors, self.acc_dtype, self.cta_group,
            self.mma_tiler_mn,
        )  # fmt: skip

        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        block_idx = cute.arch.block_idx()
        bidx = block_idx[0]
        split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        n_idx = block_idx[1] // self.split_k
        l_idx = block_idx[2]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        shape = (tiled_mma, mnk_tiler, ab_dtype, stages)
        sA = skm._smem(ab_dtype, None, 1024, sm100_utils.make_smem_layout_a(*shape))
        sB = skm._smem(ab_dtype, None, 1024, sm100_utils.make_smem_layout_b(*shape))
        acc_layout = cute_ext.make_tmem_layout_acc(
            tiled_mma, self.mma_tiler_mn, acc_stage=1
        )
        bar_full = skm._smem(cutlass.Int64, stages, 8).iterator
        bar_empty = skm._smem(cutlass.Int64, stages, 8).iterator
        bar_tma_epilog = skm._smem(cutlass.Int64, 1, 8).iterator  # FlashInfer's bias
        bar_mma_epilog = skm._smem(cutlass.Int64, 1, 8).iterator
        bar_tmem_alloc = skm._smem(cutlass.Int64, 1, 8).iterator
        tmem_base_ptr = skm._smem(cutlass.Int32, 1, 4).iterator
        mailbox = skm._smem(cutlass.Float32, self.mailbox_elements, 128)
        bar_reduce = skm._smem(cutlass.Int64, 1, 8).iterator
        stage_d = skm._smem(
            cutlass.BFloat16, None, 16,
            cute.make_layout((self.cta_m, self.cta_n), stride=(1, self.cta_m)),
        )  # fmt: skip

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, 2)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_tma_epilog, 32)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)
                cute.arch.mbarrier_init(bar_reduce, 1)

        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()  # peers' barriers before their stores

        k_tile_count = cute.size(mA, mode=[1]) // self.cta_k // self.split_k
        k_tile_start = split_rank * k_tile_count

        cute.arch.cluster_wait()

        side = bidx // self.tps
        j0 = bidx - side * self.tps
        f0 = side * SIDE + j0 * self.half  # this CTA row's first tap-0 feature
        tiles = (k_tile_start, k_tile_count)
        if warp_idx == 0:
            tap1 = f0 + GROUPS
            self.weight_warp_pair(bar_full, bar_empty, sA, a_map, f0, tap1, *tiles)
        elif warp_idx == 1:
            b_tile = cute.local_tile(mB, (self.cta_n, self.cta_k), (n_idx, None, l_idx))
            b_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")
            self.activation_warp(bar_full, bar_empty, b_tile, sB, b_map, *tiles)
        elif warp_idx == 2:
            k_blocks = self.cta_k // cute.size(tiled_mma.shape_mnk, mode=[2])
            self.mma_warp(
                bar_full, bar_empty, bar_mma_epilog, bar_tmem_alloc, tiled_mma, sA, sB,
                tmem_base_ptr, acc_layout, k_blocks, k_tile_count,
            )  # fmt: skip
        elif warp_idx >= 4:
            self.epilog_warp_conv(
                bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr, acc_layout, cute.arch.thread_idx()[0] - 128,
                mC.element_type, utils.LayoutEnum.from_tensor(mC), mailbox, bar_reduce, split_rank, stage_d,
                (mC, mB, mBase, mHc), side, j0, f0, n_idx * self.cta_n, l_idx,
            )  # fmt: skip

    @cute.experimental.jit
    def epilog_warp_conv(
        self, bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr,
        acc_layout: cutlass.Constexpr, epi_tid: cutlass.Int32,
        c_dtype: cutlass.Constexpr, d_layout: cutlass.Constexpr, mailbox, bar_reduce,
        split_rank: cutlass.Int32, stage_d: cute.Tensor, out, side: cutlass.Int32, j0: cutlass.Int32,
        f0: cutlass.Int32, col0: cutlass.Int32, l_idx: cutlass.Int32,
    ):  # fmt: skip
        """SplitKGemm.epilog_warp_act up to the staged BF16 tile (same TMEM load, rank-order reduction, cast, named barrier 1); then the
        projection store (both halves) and, on side-0 tiles, the conv."""
        mC, mX, mBase, mHc = out
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        skm._fence_tcgen05()

        acc_view = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr),
            acc_layout,
        )[((None, None), 0, 0, 0)]
        epi_tile = (self.cta_m, self.cta_n)
        load_op = sm100_utils.get_tmem_load_op(
            (self.cta_m, self.cta_n, self.cta_k), d_layout, c_dtype, self.acc_dtype,
            epi_tile, False,
        )  # fmt: skip
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(load_op, acc_view)
        sD_epi = cute.flat_divide(stage_d, epi_tile)

        rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, sD_epi, epi_tid)
        rmem = cute.AddressSpace.rmem
        rAcc = cute_ext.allocate(self.acc_dtype, rmem, rmem_layout, alignment=32)
        rD = cute_ext.allocate(c_dtype, rmem, rmem_layout, alignment=32)
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)

        half: cutlass.Constexpr = self.half
        n: cutlass.Constexpr = self.cta_n
        cpt: cutlass.Constexpr = 2 * half  # 8-channel chunks per token (half groups x 16 channels)
        cc = epi_tid % cpt
        gl = cc // 2
        ch0 = (j0 * half + gl) * GS + (cc % 2) * 8
        blk = _min_s32(epi_tid // cpt, n // BLOCK - 1)  # clamped: threads past the last item load a valid block and discard it
        tb = col0 + blk * BLOCK
        xpf = []
        bw0 = None
        bw1 = None
        if cutlass.const_expr(self.prefetch):
            cute.arch.griddepcontrol_wait()  # x is producer data: read only by threads that waited
            bw0 = load_vector(at(mBase, 0, ch0, 0))
            bw1 = load_vector(at(mBase, 1, ch0, 0))
            xpf = [load_vector(at(mX, tb + j, ch0, l_idx)) for j in range(BLOCK)]
        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        skm._fence_tcgen05()
        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
        cute.arch.fence_view_async_tmem_load()
        skm._fence_tcgen05(before=True)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)

        vpt = cutlass.const_expr(cute.size(rmem_layout))
        vpp = cutlass.const_expr(self.epilog_threads * vpt)
        if split_rank != OWNER_RANK:
            for i in cutlass.range_constexpr(0, vpt, 4):
                words = (rAcc[i], rAcc[i + 1], rAcc[i + 2], rAcc[i + 3])
                slot = mailbox.iterator + (split_rank - Int32(1)) * vpp
                slot = slot + epi_tid * vpt + i
                skm._store_shared_remote_v4(*words, slot, bar_reduce, Int32(OWNER_RANK))
        else:
            if epi_tid == 0:
                expected = self.expected_transaction_bytes
                if cutlass.const_expr(self.k23_arm):
                    skm._arm_cluster(bar_reduce, expected)
                else:
                    cute.arch.mbarrier_arrive_and_expect_tx(bar_reduce, expected)
            skm._wait_cluster(bar_reduce, Int32(0))
            for peer in cutlass.range_constexpr(self.split_k - 1):
                for i in cutlass.range_constexpr(vpt):
                    rAcc[i] = rAcc[i] + mailbox[peer * vpp + epi_tid * vpt + i]

        if split_rank == OWNER_RANK:
            rD.store(rAcc.load().to(c_dtype))
            cute_ext.partition_and_copy(thr_t2r, rD, sD_epi[None, None, 0, 0])
            # The 128 epilogue threads (warps 4-7) only: every staged value before any read of it.
            cute.arch.barrier(barrier_id=1, number_of_threads=self.epilog_threads)
            columns = mC.layout.shape[1]
            # (1) the projection tile, both halves, 8 features (16 B) per item: tile rows [0, h) -> features f0 + r, [h, 2h) -> f0 + 448 + r - h.
            dch: cutlass.Constexpr = half // 8
            items: cutlass.Constexpr = n * 2 * dch
            for i in cutlass.range_constexpr((items + 127) // 128):
                idx = epi_tid + i * 128
                token = idx // (2 * dch)
                r = idx % (2 * dch)
                tp = r // dch
                c8 = r % dch
                if (idx < items) & (col0 + token < columns):
                    vals = [stage_d[tp * half + 8 * c8 + e, token].to(Float32) for e in range(8)]
                    words = tuple(bf16x2(vals[2 * q], vals[2 * q + 1]) for q in range(4))
                    store_words(at(mC, f0 + tp * GROUPS + 8 * c8, col0 + token, l_idx), words)
            # (2) side 0: the grouped conv of x with base0 + this tile's deltas (tap 0 = tile row gl, tap 1 = tile row h + gl).
            if side == 0:
                citems: cutlass.Constexpr = (n // BLOCK) * cpt
                if cutlass.const_expr(self.prefetch):
                    if (epi_tid < citems) & (tb < columns):
                        self._conv_block(xpf, bw0, bw1, stage_d, blk, gl, mHc, tb, ch0, l_idx)
                else:
                    cute.arch.griddepcontrol_wait()  # x is producer data: read only by threads that waited
                    pw0 = load_vector(at(mBase, 0, ch0, 0))
                    pw1 = load_vector(at(mBase, 1, ch0, 0))
                    if (epi_tid < citems) & (tb < columns):
                        xs = [load_vector(at(mX, tb + j, ch0, l_idx)) for j in range(BLOCK)]
                        self._conv_block(xs, pw0, pw1, stage_d, blk, gl, mHc, tb, ch0, l_idx)
            if cutlass.const_expr(self.proxy_fences):
                skm._fence_proxy_async_global()  # proxy fence: these generic stores -> successors' TMA reads

    def _conv_block(self, xs, bw0, bw1, stage_d, blk, gl, mHc, tb, ch0, l_idx):
        """One 8-token block of one 8-channel chunk (straight-line, traced inline)."""
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
            d0 = stage_d[gl, t].to(Float32)
            d1 = stage_d[self.half + gl, t].to(Float32)
            cur = []
            for q in range(4):
                lo, hi = unpack_bf16(xs[j][q])
                cur.extend((lo, hi))
            words = _conv8_pk(b0, b1, d0, d1, cur, prev)
            store_words(at(mHc, tb + j, ch0, l_idx), words)
            prev = cur


@cache
def compile_splitk_conv(tactic, arch, oversized=False, prefetch=False):
    def fake(leading, dtype=cutlass.BFloat16):
        shape = tuple(cute.sym_int() for _ in range(3))
        stride = (1, cute.sym_int(64), 0) if leading == 0 else (cute.sym_int(64), 1, 0)
        return make_fake_tensor(dtype, shape, stride, assumed_align=16)

    y = fake(0)
    kernel = SplitKConvGemm(tactic=SplitKTactic(*tactic), proxy_fences=True, oversized=bool(oversized), prefetch=bool(prefetch))
    args = (fake(1), fake(1), y, y, cutlass.Int64(0), fake(1), fake(1), make_fake_stream())
    compiled = _DSL_COMPILE(kernel, *args, options=f"--gpu-arch={arch}")
    overflow = smem_overflow(compiled)
    if overflow is not None:  # SMEM overflow check: fail closed before any launch
        raise ValueError(f"split-K conv tactic {tuple(tactic)} uses more SMEM than its launch allocates (allocated, used = {overflow} B)")
    return compiled


class SplitKConvProjection:
    """weight [1792, 7168] BF16 (the conv kernel projection); ``policies`` {rows: (mma_m, mma_n, split, stages)} = the served split-K / OV-wide
    tactic at that row count, ``oversized_rows`` the rows whose tactic launches in the extended shared-memory mode (OV-wide).
    run(x [rows, 7168], delta [rows, 1792], base0 [2, 7168], hc [rows, 7168]): delta = the served projection output (bitwise), hc = grouped
    conv side 0 of x with base0 + delta side 0 (bitwise to the Triton conv)."""

    def __init__(self, weight, policies, oversized_rows=(), prefetch=None):
        assert tuple(weight.shape) == (2 * SIDE, 7168) and weight.dtype == torch.bfloat16 and weight.stride(1) == 1
        self.weight, self.device = weight, weight.device
        self.policies = {int(r): tuple(t) for r, t in dict(policies).items()}
        for rows, (m, n, split, stages) in self.policies.items():
            if rows % BLOCK or n < rows or n % BLOCK or m not in (64, 128) or (7168 // 128) % split or split < 2:
                raise ValueError(f"split-K conv tactic {(m, n, split, stages)} at {rows} rows is not supported")
        self.oversized_rows = frozenset(int(r) for r in oversized_rows)
        self.prefetch = bool(PREFETCH) if prefetch is None else bool(prefetch)
        self.weight_view = matrix_view(weight)
        self.kernels, self.maps = {}, {}

    def prepare(self):
        arch = architecture(self.device)
        with torch.accelerator.device_index(self.device.index):
            for rows, tactic in self.policies.items():
                over = rows in self.oversized_rows
                code = compile_splitk_conv(tactic, arch, over, self.prefetch)
                if over:
                    set_oversized(code, self.device)
                self.kernels[rows] = code
                half = tactic[0] // 2
                if half not in self.maps:
                    self.maps[half] = weight_map(self.weight, half)
        return self

    def tensor_maps(self):
        return list(self.maps.values())

    def run(self, x, delta, base0, hc):
        b = x.shape[0]
        ok = (b in self.kernels and tuple(x.shape) == (b, 7168) and tuple(delta.shape) == (b, 2 * SIDE) and tuple(base0.shape) == (TAPS, 7168)
              and tuple(hc.shape) == (b, 7168) and x.stride(1) == 1 and hc.stride(1) == 1 and base0.stride(1) == 1 and delta.is_contiguous()
              and base0.dtype == torch.bfloat16 and b % BLOCK == 0)
        if not ok:
            raise ValueError(f"split-K conv x {tuple(x.shape)} delta {tuple(delta.shape)} base {tuple(base0.shape)} hc {tuple(hc.shape)}: "
                             f"prepared {sorted(self.kernels)}")
        y = matrix_view(delta, output=True)
        a_map = self.maps[self.policies[b][0] // 2]
        self.kernels[b](self.weight_view, matrix_view(x), y, y, cutlass.Int64(a_map.data_ptr()), matrix_view(base0), matrix_view(hc),
                        current_stream(self.device))
        return hc
