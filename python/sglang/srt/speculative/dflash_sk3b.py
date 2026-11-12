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
# Modified by NVIDIA in 2026: SK3B, a streaming split-K for the DFlash drafter's K = 7168 GEMMs.
# The parent kernel SplitKTmemGemm64 is the megakernel plugin's gemm/splitk_tmem64.py (NVIDIA, Apache-2.0), based on earlier
# NVIDIA work.
# SK3B: a streaming split-K for the drafter's K = 7168 BF16 GEMMs at 96-128 activation rows.
# Built as a subclass of the megakernel plugin's SplitKTmemGemm64 (TM64): the x warp (x_warp64), the PDL trigger (fenced
# trigger mode 2, warp 3), the cluster DSMEM reduce-scatter (reduce_scatter64) and owner_of are the parent's, unchanged.
# What changes (in TM64, x waits until EVERY weight tile is staged, because x's slots are the staging ring and the x warp
# waits for its drain):
#   (1) no tensor-memory staging: every weight K tile (128 rows x 64, 16 KB) streams through its OWN shared-memory ring of `wstages` stages,
#       from kernel entry (before the PDL wait) to the last tile; the MMA reads A from shared memory (_mma_smem64, as TM64's A slots);
#       ring reuse = MMA tcgen05.commit -> w_empty -> weight warp (SplitKGemm's own ring / CUTLASS PipelineTmaUmma form).
#   (2) x has its own `slots` ring (x_warp64, the parent's), started right after the PDL wait WITHOUT waiting for any weight drain.
#   (3) tensor memory: only the accumulator (128 columns), not 512.
#   (4) owners default 8 (share 16), finish() per share; split up to 16 in a NON-PORTABLE cluster
#       (CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED).
#   The mailboxes (split x share x N x 4 B) + the owner's tile live in the weight ring alone (asserted), so no contiguity assumption.
# Options (policy flags):
# xlsu = x loaded by the 4 epilogue warps with cp.async.cg 16 B (LSU path, swizzled SW128 stores, zero-fill past the rows,
#   commit groups of one tile, wait_group(slots - 1) -> fence.proxy.async.shared::cta -> 128-thread arrive) instead of the TMA x warp, so x
#   does not queue behind the weight TMA boxes. The mailboxes may span ring + x slots (contiguous allocations, TM64's own assumption).
# bulk = the reduce-scatter's transport: each epilogue thread stages its accumulator row in a local SMEM chunk per owner ([owner][quads]
#   [share][4], the owner's mailbox slot layout; its own rows straight into its own slot), fence.proxy.async.shared::cta + named barrier,
#   arm + cluster barrier (as TM64), then ONE cp.async.bulk shared::cta -> shared::cluster per (sender, owner) completing on the owner's
#   bar_reduce; owners sum in rank order and finish exactly as TM64; a second cluster barrier keeps every sender's staging alive until all
#   owners received.
# ov = allow the plugin's extended shared-memory mode, so the ring can hold a whole rank's weight slice (one burst).
# flags 32 = DMB, a DEDICATED mailbox + owner tile in their own SMEM (not over the ring / x slots), so the epilogue cluster barrier goes:
#   owners arm bar_reduce at cluster scope (splitk._arm_cluster) when their epilogue starts, every rank pushes its rows (TM64's st.async
#   push, unchanged) as soon as its own MMAs end, owners wait, sum in rank order, finish; non-owners exit after pushing (as TM64).
#   Without it, the epilogue cluster barrier holds every push until the slowest rank's MMAs end.
# flags 512 = next-weight L2 prefetch (SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B_PF): the 'bias' tensor slot carries the NEXT GEMM's
#   weight [M', K] (K-major view) and, after its drain (every own weight tile landed and consumed), the weight warp issues
#   cp.async.bulk.prefetch.L2.global for its CTA's PF_CHUNK-byte chunks of it (chunk c to CTA c mod #CTAs). A cache hint only: no output
#   changes, nothing waits on it (no completion, no ordering edge).
# Numerics: split 8 at N96 = the same K slices, K16 MMA order and rank-order FP32 sum as TM64 (128, 96, 8, 8) (bitwise to cuBLAS at
#   96 rows) -> expected bitwise there; other splits are numerics-changing (FP64 error ratio is the check).
# Host pieces: compile helper, non-portable attribute, occupancy query, projection wrapper.
from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

from sgl_nvidia_mega.kimi_k3.cutedsl.common.host import architecture, resolve_kernel
from sgl_nvidia_mega.kimi_k3.cutedsl.common.host import stream as current_stream
from sgl_nvidia_mega.kimi_k3.cutedsl.common.primitives import asm, at, bits_float, float_bits, named_barrier, store_words
from sgl_nvidia_mega.kimi_k3.cutedsl.gemm import splitk_tmem64 as tm
from sgl_nvidia_mega.kimi_k3.cutedsl.gemm.projection import _DSL_COMPILE, matrix_view, set_oversized, smem_overflow, weight_map
from sgl_nvidia_mega.kimi_k3.cutedsl.gemm.splitk import (
    SplitKGemm,
    SplitKTactic,
    _acquire_tensor_map,
    _fence_tcgen05,
    _fenced_launch_dependents,
    _smem,
    _wait_cluster,
    _arm_cluster,
    large_smem_alloc,
)

_NONPORT = getattr(cuda.CUfunction_attribute, "CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED", None)
NORMAL_SMEM = 232448  # sharedMemPerBlockOptin (227 KiB)


@asm()
def _cp_async16(dst, src, src_bytes):
    """16 B global -> shared through L2 (cp.async.cg), zero-filled past ``src_bytes`` (0 or 16)."""
    return "cp.async.cg.shared.global [$0], [$1], 16, $2;", "r,l,r,~{memory}", (Int32(dst), Int64(src), Int32(src_bytes))


@asm()
def _cp_async_commit():
    return "cp.async.commit_group;", "~{memory}", ()


@asm()
def _cp_async_wait(n):
    """cp.async.wait_group n (n: a trace-time int)."""
    return f"cp.async.wait_group {int(n)};", "~{memory}", ()


@asm()
def _bulk_s2c(dst, src, nbytes, bar, peer):
    """``nbytes`` from this CTA's shared ``src`` to shared ``dst`` of cluster rank ``peer`` (the same offset there), completing on
    ``bar`` (an mbarrier at the same offset in the peer)."""
    code = "{ .reg .b32 d, b; mapa.shared::cluster.u32 d, $0, $4; mapa.shared::cluster.u32 b, $3, $4; "
    code += "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [d], [$1], $2, [b]; }"
    return code, "r,r,r,r,r,~{memory}", (Int32(dst), Int32(src), Int32(nbytes), bar, Int32(peer))


@asm()
def _prefetch_l2(addr, nbytes):
    """``nbytes`` (a multiple of 16) of global memory at ``addr`` into L2 (bulk prefetch; no completion)."""
    return "cp.async.bulk.prefetch.L2.global [$0], $1;", "l,r,~{memory}", (Int64(addr), Int32(nbytes))


PF_CHUNK = 16384


@asm()
def _fence_async_smem():
    """generic-proxy shared writes (cp.async, completed) -> async-proxy reads (tcgen05.mma)."""
    return "fence.proxy.async.shared::cta;", "~{memory}", ()


class SplitKStream3b(tm.SplitKTmemGemm64):
    """``tactic`` (128, N, split, x slots), ``wstages`` weight ring stages (16 KB each), ``owners`` ranks reducing 128 / owners rows."""

    def __init__(self, *, tactic, k, wstages, owners=8, xlsu=False, bulk=False, ov=False, dmb=False, pf=False):
        SplitKGemm.__init__(self, tactic=tactic)
        # The plugin's SplitKTmemGemm64.__init__ is bypassed (SplitKGemm's is called), so the TM64 options a newer plugin adds there
        # (MBOX, L2PF, XSEP, LEAN, TX576; its reduce_scatter64 reads self.mbox) are set here to OFF: with them off the inherited
        # methods (x_warp64, reduce_scatter64) run the base code paths. A plugin without them ignores the attributes.
        self.mbox, self.l2pf, self.xsep, self.x_extra, self.lean, self.tx576 = 0, 0, 0, 0, 0, 0
        self.k23_arm, self.mbox_base = False, None
        self.pf = bool(pf)  # next-weight L2 prefetch after the drain
        self.fenced_trigger = 2  # fenced trigger mode 2 only (warp 3: wait, fence, trigger)
        self.promote, self.acc_slots = 0, 1
        assert tactic.mma_m == 128 and tactic.mma_n in tm.N_TILES, "M128, N in N_TILES"
        assert 128 % owners == 0 and tactic.mma_n * (128 // owners) // 128 % 4 == 0, "N x share / 128 % 4"
        self.n = tactic.mma_n
        self.acc_columns = tm.acc_columns(self.n)
        self.acc_base = self.acc_columns
        self.tmem_cols = 128 if self.acc_columns <= 128 else 256  # power of two >= the accumulator
        self.b_tile = self.n * 64 * 2
        self.idesc = 0x08000490 | (self.n >> 3) << 17
        self.cta_k = 64
        self.k_total = k // self.cta_k
        assert self.k_total * self.cta_k == k
        split = self.split_k
        assert 2 <= split <= 16, "cluster (1, split, 1): non-portable above 8"
        self.k_tiles = -(-self.k_total // split)
        self.extra = self.k_total % split
        self.owners, self.share = owners, 128 // owners
        assert split >= owners and self.share % 4 == 0
        self.tmem_tiles, self.smem_tiles = 0, 0
        self.wstages = int(wstages)
        self.num_ab_stage = self.wstages  # the parent's drain() runs over the weight ring
        self.slots = tactic.ab_stages  # x's slots
        # every rank (short ones included) fills every ring stage once: the drain's phase logic stays in SplitKGemm's tested regime
        assert 1 <= self.wstages <= self.k_total // split and 1 <= self.slots <= self.k_total // split
        self.xlsu, self.bulk, self.ov, self.dmb = bool(xlsu), bool(bulk), bool(ov), bool(dmb)
        assert not (self.dmb and self.bulk), "dmb uses the st.async push"
        self.mailbox_bytes = split * self.share * self.n * 4
        self.tile_offset = (self.mailbox_bytes + 1023) // 1024 * 1024
        data = self.wstages * tm.A_STAGE + self.slots * self.b_tile
        self.slot_bytes = self.n // 4 * self.share * 16  # one owner's mailbox slot (= one staging chunk)
        self.stage_offset = (self.tile_offset + self.n * self.share * 4 + 1023) // 1024 * 1024
        need = self.stage_offset + owners * self.slot_bytes if self.bulk else self.tile_offset + self.n * self.share * 4
        self.dmb_bytes = (self.tile_offset + self.n * self.share * 4 + 1023) // 1024 * 1024 if self.dmb else 0
        assert self.dmb or data >= need, f"mailboxes + tile (+ bulk staging) {need} inside ring + x slots {data} (contiguous, as TM64)"
        used = data + self.dmb_bytes + (2 * self.wstages + 2 * self.k_tiles + 3) * 8 + 4
        self.smem_bytes = (used + 2048 + 1023) // 1024 * 1024
        cap = large_smem_alloc() if self.ov else NORMAL_SMEM
        assert self.smem_bytes <= cap, f"SMEM {self.smem_bytes} > {cap}"

    @cute.experimental.jit
    def body(
        self, mA: cute.Tensor, mB: cute.Tensor, mS: cute.Tensor, a_map: cutlass.Int64,
        out,
    ):  # fmt: skip
        stages, k_tiles = self.wstages, self.k_tiles
        ab_dtype = mA.element_type
        majors = (utils.LayoutEnum.from_tensor(t).mma_major_mode() for t in (mA, mB))
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype, ab_dtype, *majors, self.acc_dtype, self.cta_group,
            self.mma_tiler_mn,
        )  # fmt: skip
        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        bidx, by, l_idx = cute.arch.block_idx()
        split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        n_idx = by // self.split_k
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        base_tiles = self.k_total // self.split_k
        tiles = base_tiles + Int32(split_rank < self.extra)
        over = Int32(split_rank > self.extra) * (split_rank - self.extra)
        first = split_rank * (base_tiles + 1) - over

        # Shared memory: [weight ring][x's slots]; the mailboxes + the owner's tile reuse the weight ring after the cluster barrier.
        b_shape = (tiled_mma, mnk_tiler, ab_dtype, self.slots)
        b_layout = sm100_utils.make_smem_layout_b(*b_shape)
        w_ring = _smem(cutlass.Uint8, stages * tm.A_STAGE, 1024)
        data = Int32(w_ring.iterator.toint())
        sB = _smem(ab_dtype, None, 1024, b_layout)
        xbase = Int32(sB.iterator.toint())
        mbox = data
        if cutlass.const_expr(self.dmb):
            mb = _smem(cutlass.Uint8, self.dmb_bytes, 1024)
            mbox = Int32(mb.iterator.toint())
        w_full = _smem(cutlass.Int64, stages, 8).iterator
        w_empty = _smem(cutlass.Int64, stages, 8).iterator
        b_full = _smem(cutlass.Int64, k_tiles, 8).iterator
        b_empty = _smem(cutlass.Int64, k_tiles, 8).iterator
        bar_mma_epilog = _smem(cutlass.Int64, 1, 8).iterator
        bar_tmem_alloc = _smem(cutlass.Int64, 1, 8).iterator
        bar_reduce = _smem(cutlass.Int64, 1, 8).iterator
        tmem_base_ptr = _smem(cutlass.Int32, 1, 4).iterator
        self.accs = None

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(w_full + i, 1)
                    cute.arch.mbarrier_init(w_empty + i, 1)
                for i in range(k_tiles):
                    cute.arch.mbarrier_init(b_full + i, 128 if self.xlsu else 1)
                    cute.arch.mbarrier_init(b_empty + i, 1)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)
                cute.arch.mbarrier_init(bar_reduce, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()  # every barrier before any DSMEM traffic
        cute.arch.cluster_wait()
        if warp_idx == 0:
            pf_base, pf_chunks, pf_cta, pf_ctas = Int64(0), Int32(0), Int32(0), Int32(1)
            if cutlass.const_expr(self.pf):
                gx, gy, _ = cute.arch.grid_dim()
                pf_base = mS.iterator.toint()
                pf_chunks = Int32(mS.layout.shape[0]) * Int32(mS.layout.shape[1]) * 2 // PF_CHUNK
                pf_cta = bidx + gx * by
                pf_ctas = gx * gy
            self.weight_warp3b(w_full, w_empty, data, a_map, bidx * self.cta_m, first, tiles, pf_base, pf_chunks, pf_cta, pf_ctas)
            if cutlass.const_expr(not self.dmb):
                tm._cluster_phases()
            if cutlass.const_expr(self.bulk):
                tm._cluster_phases()
        elif warp_idx == 1:
            if cutlass.const_expr(not self.xlsu):
                b_tile = cute.local_tile(mB, (self.cta_n, self.cta_k), (n_idx, None, l_idx))
                b_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")
                self.x_warp64(b_full, b_empty, b_tile, sB, b_map, first, tiles)  # the parent's: PDL wait, async-proxy fence, own slots
            if cutlass.const_expr(not self.dmb):
                tm._cluster_phases()
            if cutlass.const_expr(self.bulk):
                tm._cluster_phases()
        elif warp_idx == 2:
            self.mma_warp3b(
                w_full, w_empty, b_full, b_empty, bar_mma_epilog, bar_tmem_alloc,
                data, xbase, tmem_base_ptr, tiles,
            )  # fmt: skip
            if cutlass.const_expr(not self.dmb):
                tm._cluster_phases()
            if cutlass.const_expr(self.bulk):
                tm._cluster_phases()
        elif warp_idx == 3:  # fenced trigger mode 2: wait, (MMA warp holds its TMEM), fence, trigger
            cute.arch.griddepcontrol_wait()
            named_barrier(tm.TRIGGER_BARRIER, 64)
            _fenced_launch_dependents()
            if cutlass.const_expr(not self.dmb):
                tm._cluster_phases()
            if cutlass.const_expr(self.bulk):
                tm._cluster_phases()
        else:
            if cutlass.const_expr(self.xlsu):
                self.x_lsu(cute.arch.thread_idx()[0] - 128, b_full, b_empty, mB, xbase, first, tiles, n_idx)
            if cutlass.const_expr(self.dmb):
                self.reduce_dmb(
                    cute.arch.thread_idx()[0] - 128, bar_mma_epilog, bar_tmem_alloc,
                    tmem_base_ptr, mbox, bar_reduce, split_rank, out,
                    bidx * self.cta_m, n_idx * self.cta_n, l_idx,
                )  # fmt: skip
            elif cutlass.const_expr(self.bulk):
                self.reduce_bulk(
                    cute.arch.thread_idx()[0] - 128, bar_mma_epilog, bar_tmem_alloc,
                    tmem_base_ptr, data, bar_reduce, split_rank, out,
                    bidx * self.cta_m, n_idx * self.cta_n, l_idx,
                )  # fmt: skip
            else:
                self.reduce_scatter64(
                    cute.arch.thread_idx()[0] - 128, bar_mma_epilog, bar_tmem_alloc,
                    tmem_base_ptr, data, bar_reduce, split_rank, out,
                    bidx * self.cta_m, n_idx * self.cta_n, l_idx,
                )  # fmt: skip

    @cute.experimental.jit
    def x_lsu(
        self, epi_tid: cutlass.Int32, b_full, b_empty, mB: cute.Tensor, xbase: Int32, first: cutlass.Int32,
        tiles: cutlass.Int32, n_idx: cutlass.Int32,
    ):  # fmt: skip
        """x's K tiles by the 128 epilogue threads after the PDL wait: thread (row rb + 16 i, 16-byte chunk c) of each N x 64 tile, stored
        at row * 128 + ((c ^ row % 8) * 16) of its slot (the SW128 K-major layout TMA writes); `slots` tiles in flight as commit groups;
        a tile is published (fence.proxy.async, arrive on b_full, count 128) once its group completes; slot reuse waits on b_empty."""
        cute.arch.griddepcontrol_wait()
        rows_total = mB.layout.shape[0]
        rstride = mB.layout.stride[0]
        gbase = mB.iterator.toint()
        c = epi_tid % 8
        rb = epi_tid // 8
        swz = (c ^ (rb % 8)) * 16  # r % 8 == rb % 8 for r = rb + 16 i
        row0 = n_idx * self.n
        slots = self.slots
        for k_tile in cutlass.range(tiles, unroll=1):
            slot = k_tile % slots
            if k_tile >= slots:
                cute.arch.mbarrier_wait(b_empty + k_tile - slots, 0)
            k0 = (first + k_tile) * self.cta_k + c * 8
            dst0 = xbase + slot * self.b_tile + swz
            for i in cutlass.range_constexpr(self.n // 16):
                r = rb + 16 * i
                ok = Int32(row0 + r < rows_total)
                src = gbase + (Int64((row0 + r) * ok) * Int64(rstride) + Int64(k0)) * 2
                _cp_async16(dst0 + r * 128, src, ok * 16)
            _cp_async_commit()
            if k_tile >= slots - 1:
                _cp_async_wait(slots - 1)
                _fence_async_smem()
                cute.arch.mbarrier_arrive(b_full + k_tile - (slots - 1))
        _cp_async_wait(0)
        _fence_async_smem()
        for j in cutlass.range(slots - 1, unroll=1):
            cute.arch.mbarrier_arrive(b_full + tiles - (slots - 1) + j)

    @cute.experimental.jit
    def weight_warp3b(
        self, w_full, w_empty, wring: Int32, a_map: cutlass.Int64, row0: cutlass.Int32,
        first: cutlass.Int32, tiles: cutlass.Int32, pf_base: Int64, pf_chunks: Int32, pf_cta: Int32, pf_ctas: Int32,
    ):  # fmt: skip
        """Every weight K tile of this rank through the ring, from kernel entry (no PDL wait: weights are not written upstream); one
        evict_first {64, 128} box per tile; then the drain (every stage released by the MMAs) before the cluster phase."""
        stages = self.wstages
        _acquire_tensor_map(a_map)
        empty_phase = cutlass.Int32(1)
        for k_tile in cutlass.range(tiles, unroll=1):
            stage = k_tile % stages
            cute.arch.mbarrier_wait(w_empty + stage, empty_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(w_full + stage, tm.A_STAGE)
                k0 = (first + k_tile) * self.cta_k
                self.issue_a(wring + stage * tm.A_STAGE, w_full + stage, a_map, k0, row0)
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1
        self.drain(w_empty, empty_phase, tiles)
        if cutlass.const_expr(self.pf):
            self.prefetch_next(pf_base, pf_chunks, pf_cta, pf_ctas)

    @cute.experimental.jit
    def prefetch_next(self, pf_base: Int64, pf_chunks: Int32, pf_cta: Int32, pf_ctas: Int32):
        """Next-weight prefetch: this CTA's 16 KiB chunks (c = pf_cta + j * pf_ctas) of the next weight into L2, issued by one lane."""
        iters = (pf_chunks + pf_ctas - 1) // pf_ctas
        with cute.arch.elect_one():
            for j in cutlass.range(iters, unroll=1):
                c = pf_cta + j * pf_ctas
                if c < pf_chunks:
                    _prefetch_l2(pf_base + Int64(c) * PF_CHUNK, PF_CHUNK)

    @cute.experimental.jit
    def mma_warp3b(
        self, w_full, w_empty, b_full, b_empty, bar_mma_epilog, bar_tmem_alloc,
        wring: Int32, xbase: Int32, tmem_base_ptr, tiles: cutlass.Int32,
    ):  # fmt: skip
        """Tile k's four M128 N K16 MMAs (A = weight ring stage k % wstages, B = x slot k % slots, both shared memory) once both landed,
        in K order; each tile's commit releases its ring stage and (if a later x tile reuses it) its x slot."""
        stages, k_blocks = self.wstages, self.cta_k // 16
        a_step, b_step = tm.A_STAGE // 16, self.b_tile // 16  # descriptor units
        cute.arch.alloc_tmem(self.tmem_cols, tmem_base_ptr, is_two_cta=False)
        _fence_tcgen05(before=True)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        named_barrier(tm.TRIGGER_BARRIER, 64, arrive=True)  # warp 3 triggers the dependents only after this allocation
        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)
        _fence_tcgen05()
        base = Int32(cute.make_tensor(tmem_base_ptr, cute.make_layout(1))[0])
        w_desc = ((wring >> 4) & 0x3FFF) | 0x10000
        x_desc = ((xbase >> 4) & 0x3FFF) | 0x10000
        for k in cutlass.range_constexpr(self.k_tiles):
            if k < tiles:
                stage, phase = k % stages, (k // stages) % 2
                cute.arch.mbarrier_wait(w_full + stage, phase)
                cute.arch.mbarrier_wait(b_full + k, 0)
                _fence_tcgen05()
                with cute.arch.elect_one():
                    for kb in cutlass.range_constexpr(k_blocks):
                        a = w_desc + stage * a_step + 2 * kb
                        b = x_desc + (k % self.slots) * b_step + 2 * kb
                        accumulate = Int32(0 if k == 0 and kb == 0 else 1)
                        tm._mma_smem64(base, a, b, accumulate, idesc=self.idesc)
                    tcgen05.commit(w_empty + stage, None, self.cta_group)
                    if cutlass.const_expr(k + self.slots < self.k_tiles):
                        tcgen05.commit(b_empty + k, None, self.cta_group)
        with cute.arch.elect_one():
            tcgen05.commit(bar_mma_epilog, None, self.cta_group)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)  # the release lands in phase 1
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 1)
        _fence_tcgen05()
        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        cute.arch.dealloc_tmem(tmem_ptr, self.tmem_cols, is_two_cta=False)

    @cute.experimental.jit
    def reduce_dmb(
        self, epi_tid: cutlass.Int32, bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr,
        mbox: Int32, bar_reduce, split_rank: cutlass.Int32, out,
        row0: cutlass.Int32, col0: cutlass.Int32, l_idx: cutlass.Int32,
    ):  # fmt: skip
        """TM64's reduce_scatter64 (no promote) on a DEDICATED mailbox ``mbox`` (+ the owner tile at tile_offset): owners arm bar_reduce at
        cluster scope first; each thread pushes its accumulator row as soon as this CTA's MMAs end (no cluster barrier); owners wait, sum
        in rank order, finish."""
        split, share = self.split_k, self.share
        quads: cutlass.Constexpr = self.n // 4
        slot_bytes = quads * share * 16
        if (epi_tid == 0) & (split_rank < self.owners):
            _arm_cluster(bar_reduce, (split - 1) * slot_bytes)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        _fence_tcgen05()
        base = Int32(cute.make_tensor(tmem_base_ptr, cute.make_layout(1))[0])
        quadrant = epi_tid // 32
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
            row = row0 + split_rank * share
            self.finish(epi_tid, totals, mbox + self.tile_offset, out, row, col0, l_idx)

    @cute.experimental.jit
    def reduce_bulk(
        self, epi_tid: cutlass.Int32, bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr,
        data: Int32, bar_reduce, split_rank: cutlass.Int32, out,
        row0: cutlass.Int32, col0: cutlass.Int32, l_idx: cutlass.Int32,
    ):  # fmt: skip
        """TM64's reduce_scatter64 (no promote) with a bulk transport: accumulator row ``epi_tid`` -> this CTA's staging chunk of its
        owner (own rows: straight into its own mailbox slot), then one cp.async.bulk per (this rank, owner) into slot ``split_rank`` of the
        owner's mailbox; owners sum the slots in rank order and ``finish``; a second cluster barrier before exit."""
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        _fence_tcgen05()
        base = Int32(cute.make_tensor(tmem_base_ptr, cute.make_layout(1))[0])
        quadrant = epi_tid // 32  # warps 4-7 read TMEM lanes 32 (warp % 4) ..
        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        _fence_tcgen05()
        values = tm._load_row(base + (quadrant * 32 << 16), self.n)
        _fence_tcgen05(before=True)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)

        split, share = self.split_k, self.share
        quads: cutlass.Constexpr = self.n // 4
        slot_bytes = self.slot_bytes
        stage = data + self.stage_offset
        owner, r = self.owner_of(epi_tid, row0)
        own = Int32(owner == split_rank)
        to_stage = stage + owner * slot_bytes + r * 16
        to_slot = data + split_rank * slot_bytes + r * 16
        dst = to_stage + own * (to_slot - to_stage)
        for i in cutlass.range_constexpr(quads):
            bits = tuple(float_bits(values[4 * i + j]) for j in range(4))
            tm._st_shared_v4(dst + i * share * 16, bits)
        _fence_async_smem()  # these generic stores -> the bulk copies' (async proxy) reads
        named_barrier(3, 128)
        if (epi_tid == 0) & (split_rank < self.owners):
            cute.arch.mbarrier_arrive_and_expect_tx(bar_reduce, (split - 1) * slot_bytes)
        tm._cluster_arrive()  # this CTA's MMAs are done: its slots may take partials
        tm._cluster_wait()
        if (epi_tid < self.owners) & (epi_tid != split_rank):
            _bulk_s2c(data + split_rank * slot_bytes, stage + epi_tid * slot_bytes, slot_bytes, bar_reduce, epi_tid)
        if split_rank < self.owners:
            _wait_cluster(bar_reduce, Int32(0))
            tm._cluster_arrive()  # everything this owner needs has landed: the senders' staging may go
            width: cutlass.Constexpr = self.n * share // 128
            rr, g = epi_tid % share, epi_tid // share
            totals = cute.make_rmem_tensor(width, Float32)
            for s in cutlass.range_constexpr(split):
                for v in cutlass.range_constexpr(width // 4):
                    offset = ((s * quads + width // 4 * g + v) * share + rr) * 16
                    words = tm._ld_shared_v4(data + offset)
                    for j in cutlass.range_constexpr(4):
                        part = bits_float(words[j])
                        if cutlass.const_expr(s == 0):
                            totals[4 * v + j] = part
                        else:
                            totals[4 * v + j] = totals[4 * v + j] + part
            row = row0 + split_rank * share
            tile = data + self.tile_offset
            self.finish(epi_tid, totals, tile, out, row, col0, l_idx)
        else:
            tm._cluster_arrive()
        tm._cluster_wait()  # no CTA leaves while a bulk copy may still read its staging

    @cute.experimental.jit
    def finish(
        self, epi_tid: cutlass.Int32, totals: cute.Tensor, tile: cutlass.Int32, out,
        row0: cutlass.Int32, col0: cutlass.Int32, l_idx: cutlass.Int32,
    ):  # fmt: skip
        """The share-wise finish (self.share for SHARE): rounded to BF16, transposed through ``tile`` [N tokens][share], stored one
        token per thread as share / 8 16-byte rows of C (M-major), guarded by token < N, token < the activation rows, rows < M."""
        mC = out[0]
        share: cutlass.Constexpr = self.share
        rr, g = epi_tid % share, epi_tid // share
        width: cutlass.Constexpr = self.n * share // 128
        for t in cutlass.range_constexpr(width):
            tm._st_shared_bf16(tile + ((width * g + t) * share + rr) * 2, totals[t])
        named_barrier(3, 128)
        rows, columns = mC.layout.shape[0], mC.layout.shape[1]
        token = epi_tid
        if (token < self.n) & (col0 + token < columns):
            for c in cutlass.range_constexpr(share // 8):
                if row0 + 8 * c < rows:
                    words = tm._ld_shared_v4(tile + (token * share + 8 * c) * 2)
                    store_words(at(mC, row0 + 8 * c, col0 + token, l_idx), words)


@cache
def compile_sk3b(n, split, slots, wstages, owners, k, arch, flags=0):
    """(compiled, smem_bytes) for SK3B (128, n, split, slots) with ``wstages`` ring stages; plugin SMEM overflow check, as TM64."""

    def fake(leading, dtype=cutlass.BFloat16):
        shape = tuple(cute.sym_int() for _ in range(3))
        stride = (1, cute.sym_int(64), 0) if leading == 0 else (cute.sym_int(64), 1, 0)
        return make_fake_tensor(dtype, shape, stride, assumed_align=16)

    y = fake(0)
    kernel = SplitKStream3b(tactic=SplitKTactic(128, n, split, slots), k=k, wstages=wstages, owners=owners,
                            xlsu=bool(flags & 1), bulk=bool(flags & 2), ov=bool(flags & 4), dmb=bool(flags & 32),
                            pf=bool(flags & 512))
    bias = fake(1) if flags & 512 else y  # next-weight L2 prefetch: the next weight's K-major view rides the bias slot
    args = (fake(1), fake(1), y, bias, cutlass.Int64(0), make_fake_stream())
    compiled = _DSL_COMPILE(kernel, *args, options=f"--gpu-arch={arch}")
    overflow = smem_overflow(compiled)
    if overflow is not None:
        raise ValueError(f"SK3B tactic {(128, n, split, slots, wstages)} owners {owners} uses more SMEM than its launch allocates ({overflow} B)")
    return compiled, kernel.smem_bytes


def pf_policy(pol) -> bool:
    """The policy carries flags 512 (next-weight L2 prefetch)."""
    return len(pol) > 5 and bool(int(pol[5]) & 512)


def prefetch_target(weight):
    """Next-weight L2 prefetch: the next GEMM's weight, checked: contiguous BF16, a whole number of 16 KiB chunks."""
    if weight is None or weight.dtype != torch.bfloat16 or weight.dim() != 2 or not weight.is_contiguous():
        raise ValueError("SK3B prefetch policy needs the next weight: a contiguous 2-D BF16 tensor")
    if weight.numel() * 2 % PF_CHUNK:
        raise ValueError(f"SK3B prefetch target {tuple(weight.shape)}: not a whole number of {PF_CHUNK} B chunks")
    return weight


_ALLOWED = {}


def allow_nonportable(code, device):
    """NON_PORTABLE_CLUSTER_SIZE_ALLOWED = 1 on the compiled function (once per code object and device); returns it."""
    key = (id(code), device.index)
    if key not in _ALLOWED:
        _, function = resolve_kernel(code, device)
        attr = _NONPORT if _NONPORT is not None else cuda.CUfunction_attribute(14)
        (status,) = cuda.cuFuncSetAttribute(function, attr, 1)
        if status != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"non-portable cluster size attribute: {status}")
        _ALLOWED[key] = function
    return _ALLOWED[key]


def active_clusters(function, split, smem, m_tiles):
    """cuOccupancyMaxActiveClusters for this launch shape (diagnostic; None when the query fails)."""
    try:
        cfg = cuda.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = m_tiles, split, 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = 256, 1, 1
        cfg.sharedMemBytes = smem
        attr = cuda.CUlaunchAttribute()
        attr.id = cuda.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        attr.value.clusterDim.x, attr.value.clusterDim.y, attr.value.clusterDim.z = 1, split, 1
        cfg.attrs = [attr]
        cfg.numAttrs = 1
        status, count = cuda.cuOccupancyMaxActiveClusters(function, cfg)
        return int(count) if status == cuda.CUresult.CUDA_SUCCESS else f"err {status}"
    except Exception as ex:  # noqa: BLE001
        return f"query failed {type(ex).__name__}: {str(ex)[:80]}"


class SK3BProjection:
    """weight [M, K] BF16; ``policies`` {rows: (N tile, split, x slots, wstages, owners[, flags])}, flags = 1 xlsu | 2 bulk | 4 ov;
    run(x [rows, K], out [rows, M])."""

    def __init__(self, weight, policies):
        assert weight.dim() == 2 and weight.dtype == torch.bfloat16 and weight.stride(1) == 1
        self.weight, self.device, self.policies = weight, weight.device, dict(policies)
        self.weight_view = matrix_view(weight)
        self.kernels, self.info, self.map = {}, {}, None

    def prepare(self):
        arch = architecture(self.device)
        k = self.weight.shape[1]
        with torch.accelerator.device_index(self.device.index):
            for rows, pol in self.policies.items():
                n, split, slots, wstages, owners = pol[:5]
                flags = int(pol[5]) if len(pol) > 5 else 0
                code, smem = compile_sk3b(n, split, slots, wstages, owners, k, arch, flags)
                function = allow_nonportable(code, self.device)
                if smem > NORMAL_SMEM:
                    set_oversized(code, self.device)
                self.kernels[rows] = code
                self.info[rows] = dict(smem=smem, function=function, split=split)
            self.map = weight_map(self.weight, 128)
        return self

    def clusters(self, rows):
        i = self.info[rows]
        return active_clusters(i["function"], i["split"], i["smem"], -(-self.weight.shape[0] // 128))

    def run(self, x, out, prefetch=None):
        """``prefetch``: the next GEMM's weight [M', K] (required by a policy with flags 512, ignored otherwise)."""
        b = x.shape[0]
        if b not in self.kernels or x.shape[1] != self.weight.shape[1] or out.shape != (b, self.weight.shape[0]):
            raise ValueError(f"SK3B input {tuple(x.shape)} out {tuple(out.shape)}: prepared {sorted(self.kernels)}")
        y = matrix_view(out, output=True)
        bias = y
        if pf_policy(self.policies[b]):
            bias = matrix_view(prefetch_target(prefetch))
        self.kernels[b](self.weight_view, matrix_view(x), y, bias, cutlass.Int64(self.map.data_ptr()), current_stream(self.device))
        return out
