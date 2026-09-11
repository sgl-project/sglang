# Copyright (c) 2026, SGLang Team.
"""End-to-end transposed SM120 paged-decode specialization.

This path keeps the packed query axis on the N=8 dimension of warp MMA:

    scores.T = K @ Q.T        # (64, 8)
    output.T = V.T @ P.T      # (256, 8)

The dataflow is isolated from the general SM120 kernel because its page-TMA
transport, column-wise online softmax, and transposed epilogue form one compile
specialization.
"""

import math
from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import warp
from cutlass.pipeline import PipelineAsync, PipelineState
from quack import layout_utils

from sglang.kernels.ops.attention.fa4_sm120.flash_fwd import (
    FlashAttentionForwardSm120,
)
from sglang.kernels.ops.attention.flash_attn.cute import utils
from sglang.kernels.ops.attention.flash_attn.cute.block_info import BlockInfo
from sglang.kernels.ops.attention.flash_attn.cute.named_barrier import NamedBarrierFwd
from sglang.kernels.ops.attention.flash_attn.cute.pack_gqa import PackGQA
from sglang.kernels.ops.attention.flash_attn.cute.seqlen_info import SeqlenInfoQK
from sglang.kernels.ops.attention.flash_attn.cute.utils import AuxData


class FlashAttentionForwardSm120DecodeTranspose(FlashAttentionForwardSm120):
    """M64N8 QK and transposed PV for qualified packed single-token decode."""

    # Paged TMA is part of this kernel's dataflow, not a runtime tuning knob.
    # Keeping it on the distinct class identity prevents a gather-compiled
    # specialization from being reused for the TMA tensor layout.
    paged_tma = True
    query_mma_n = 8
    query_in_regs = True

    def _uses_n_distributed_qk(self) -> bool:
        # Reuse the base kernel's four-consumer-warp dispatch. All four warps
        # participate in both transposed MMA phases.
        return True

    def _uses_split_pv_warps(self) -> bool:
        # Experimental mixed-HDV channel slices retain the same shared P
        # handoff even though they are outside the base kernel's qualified
        # (HDQ, HDV) configuration table.
        return True

    def _setup_attributes(self):
        super()._setup_attributes()
        # The tiled MMA is deliberately warp-local. The base kernel normally
        # derives its cooperative-group size from TiledMma.size, which would
        # expose only one consumer warp here. Four physical consumer warps
        # instead operate on disjoint K rows.
        self.num_qk_threads = self.num_threads
        self.num_mma_threads = self.num_threads
        self.num_Q_load_threads = self.num_threads
        self.num_epilogue_threads = self.num_threads

    def _get_tiled_mma(self):
        mma_op = warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16))
        tiled_mma_qk = cute.make_tiled_mma(
            mma_op,
            (1, 1, 1),
            permutation_mnk=(16, self.query_mma_n, 16),
        )
        # Each warp covers 64 value rows; the four disjoint SMEM views cover
        # HDV=256 without a CTA-level M permutation.
        tiled_mma_pv = cute.make_tiled_mma(
            mma_op,
            (1, 1, 1),
            permutation_mnk=(64, self.query_mma_n, 16),
        )
        return tiled_mma_qk, tiled_mma_pv

    @cute.jit
    def _gemm_n8(
        self,
        tiled_mma: cute.TiledMma,
        acc: cute.Tensor,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        tCsA: cute.Tensor,
        tCsB: cute.Tensor,
        smem_thr_copy_A: cute.TiledCopy,
        smem_thr_copy_B: cute.TiledCopy,
        B_in_regs: cutlass.Constexpr[bool] = False,
    ):
        """Issue an N=8 warp-MMA mainloop through the underlying MMA atom.

        CuTe DSL's tiled-MMA verifier rejects m16n8k16 when logical N is
        exactly eight because it compares the raw A value mode against the C
        value mode. The hardware atom itself has the correct native fragment
        contract, so keep tiling for partitioning/copies and issue GEMM through
        that atom.
        """
        mma_atom = cute.make_mma_atom(tiled_mma.op)
        tCrA_copy_view = smem_thr_copy_A.retile(tCrA)
        tCrB_copy_view = smem_thr_copy_B.retile(tCrB)
        cute.copy(
            smem_thr_copy_A,
            tCsA[None, None, 0],
            tCrA_copy_view[None, None, 0],
        )
        if const_expr(not B_in_regs):
            cute.copy(
                smem_thr_copy_B,
                tCsB[None, None, 0],
                tCrB_copy_view[None, None, 0],
            )
        for k in cutlass.range_constexpr(cute.size(tCsA.shape[2])):
            if k < cute.size(tCsA.shape[2]) - 1:
                cute.copy(
                    smem_thr_copy_A,
                    tCsA[None, None, k + 1],
                    tCrA_copy_view[None, None, k + 1],
                )
                if const_expr(not B_in_regs):
                    cute.copy(
                        smem_thr_copy_B,
                        tCsB[None, None, k + 1],
                        tCrB_copy_view[None, None, k + 1],
                    )
            cute.gemm(
                mma_atom,
                acc,
                tCrA[None, None, k],
                tCrB[None, None, k],
                acc,
            )

    @cute.jit
    def _rescale_transposed_o(
        self,
        acc_O: cute.Tensor,
        tiled_mma_pv: cute.TiledMma,
        tidx: Int32,
        sRowScale: cute.Tensor,
        scale_row: cutlass.Constexpr[int],
    ):
        lane_idx = tidx % cute.arch.WARP_SIZE
        thr_mma_pv = tiled_mma_pv.get_slice(lane_idx)
        acc_O_qd = layout_utils.reshape_acc_to_mn(acc_O, transpose=True)
        cO = cute.make_identity_tensor((64, self.query_mma_n))
        tOcO_qd = layout_utils.reshape_acc_to_mn(
            thr_mma_pv.partition_C(cO), transpose=True
        )
        for r in cutlass.range(cute.size(acc_O_qd, mode=[0]), unroll_full=True):
            query_row = tOcO_qd[r, 0][1]
            acc_O_qd[r, None].store(
                acc_O_qd[r, None].load() * sRowScale[scale_row, query_row]
            )

    @cute.jit
    def _compute_one_n_block_transposed(
        self,
        n_block: Int32,
        consumer_state: PipelineState,
        acc_O: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sP: cute.Tensor,
        sRowScale: cute.Tensor,
        pipeline_k: PipelineAsync,
        pipeline_v: PipelineAsync,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        smem_thr_copy_K: cute.TiledCopy,
        smem_thr_copy_Q: cute.TiledCopy,
        smem_thr_copy_V: cute.TiledCopy,
        smem_thr_copy_P: cute.TiledCopy,
        tSrK: cute.Tensor,
        tSrQ: cute.Tensor,
        tOrV: cute.Tensor,
        tOrP: cute.Tensor,
        tKsK: cute.Tensor,
        tQsQ: cute.Tensor,
        tVsV: cute.Tensor,
        tPsP: cute.Tensor,
        tidx: Int32,
        softmax_scale_log2: Float32,
        seqlen: SeqlenInfoQK,
        window_size_left: Optional[Int32],
        is_first_n_block: cutlass.Constexpr[bool] = False,
    ):
        num_qk_warps = const_expr(4)
        local_sum_base = const_expr(num_qk_warps)
        global_max_row = const_expr(2 * num_qk_warps)
        global_sum_row = const_expr(global_max_row + 1)
        old_o_scale_row = const_expr(global_max_row + 2)
        warp_scale_base = const_expr(global_max_row + 3)
        p_stage = consumer_state.index
        warp_idx = tidx // cute.arch.WARP_SIZE
        lane_idx = tidx % cute.arch.WARP_SIZE
        key_row_base = warp_idx * const_expr(16)

        k_wait_token = pipeline_k.consumer_try_wait(consumer_state)
        pipeline_k.consumer_wait(consumer_state, k_wait_token)

        thr_mma_qk = tiled_mma_qk.get_slice(lane_idx)
        acc_shape_S = thr_mma_qk.partition_shape_C((16, self.query_mma_n))
        acc_S = cute.make_rmem_tensor(acc_shape_S, Float32)
        acc_S.fill(0.0)
        self._gemm_n8(
            tiled_mma_qk,
            acc_S,
            tSrK,
            tSrQ,
            tKsK[None, None, None, p_stage],
            tQsQ,
            smem_thr_copy_K,
            smem_thr_copy_Q,
            B_in_regs=self.query_in_regs,
        )
        pipeline_k.consumer_release(consumer_state)

        acc_S_qk = layout_utils.reshape_acc_to_mn(acc_S, transpose=True)
        cS = cute.make_identity_tensor((16, self.query_mma_n))
        tScS_qk = layout_utils.reshape_acc_to_mn(
            thr_mma_qk.partition_C(cS), transpose=True
        )
        num_query_rows = cute.size(acc_S_qk, mode=[0])
        row_max_local = cute.make_rmem_tensor(num_query_rows, Float32)
        row_sum_local = cute.make_rmem_tensor(num_query_rows, Float32)
        for r in cutlass.range(num_query_rows, unroll_full=True):
            # The final paged tile can be only partially populated. The page
            # gather leaves invalid SMEM rows untouched. Keep the test outside
            # the unrolled fragment loop so complete 64-token tiles pay no
            # per-element coordinate/comparison cost.
            tile_start = n_block * self.tile_n
            local_window_start = (
                cutlass.max(
                    seqlen.seqlen_k - 1 - window_size_left,
                    0,
                )
                if const_expr(self.is_local and window_size_left is not None)
                else Int32(0)
            )
            if tile_start + self.tile_n > seqlen.seqlen_k or (
                const_expr(self.is_local and window_size_left is not None)
                and tile_start < local_window_start
            ):
                for c in cutlass.range(cute.size(acc_S_qk, mode=[1]), unroll_full=True):
                    key_row = tile_start + key_row_base + tScS_qk[r, c][0]
                    if key_row >= seqlen.seqlen_k or (
                        const_expr(self.is_local and window_size_left is not None)
                        and key_row < local_window_start
                    ):
                        acc_S_qk[r, c] = -Float32.inf

            acc_S_row = acc_S_qk[r, None].load()
            row_max = utils.fmax_reduce(acc_S_row)
            # In an m16n8 accumulator, lanes with the same low two lane
            # bits own the same pair of N/query columns. Reducing across M
            # (keys) therefore uses the strided lane group
            # {lane, lane^4, lane^8, lane^16}, not a contiguous width-4
            # group as in the ordinary row-wise QK layout.
            for offset in cutlass.range_constexpr(2, 5):
                row_max = utils.fmax(
                    row_max,
                    cute.arch.shuffle_sync_bfly(row_max, offset=1 << offset),
                )
            row_max_safe = 0.0 if row_max == -Float32.inf else row_max
            acc_S_row_exp = cute.math.exp2(
                (acc_S_row - row_max_safe) * softmax_scale_log2,
                fastmath=True,
            )
            row_sum = utils.fadd_reduce(acc_S_row_exp)
            for offset in cutlass.range_constexpr(2, 5):
                row_sum += cute.arch.shuffle_sync_bfly(row_sum, offset=1 << offset)
            row_max_local[r] = row_max
            row_sum_local[r] = row_sum
            acc_S_qk[r, None].store(acc_S_row_exp)

        keys_per_warp = const_expr(self.tile_n // num_qk_warps)
        if tScS_qk[0, 0][0] % keys_per_warp == 0:
            for r in cutlass.range(num_query_rows, unroll_full=True):
                query_row = tScS_qk[r, 0][1]
                sRowScale[warp_idx, query_row] = row_max_local[r]
                sRowScale[local_sum_base + warp_idx, query_row] = row_sum_local[r]
        cute.arch.fence_view_async_shared()

        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.PFull),
            number_of_threads=self.num_mma_threads,
        )
        if tidx < self.query_mma_n:
            query_row = tidx
            row_max = sRowScale[0, query_row]
            for warp_idx_it in cutlass.range_constexpr(1, num_qk_warps):
                row_max = utils.fmax(row_max, sRowScale[warp_idx_it, query_row])
            row_max_prev = (
                row_max
                if const_expr(is_first_n_block)
                else sRowScale[global_max_row, query_row]
            )
            row_max_new = (
                row_max
                if const_expr(is_first_n_block)
                else utils.fmax(row_max_prev, row_max)
            )
            row_max_new_safe = 0.0 if row_max_new == -Float32.inf else row_max_new
            old_o_scale = (
                1.0
                if const_expr(is_first_n_block)
                else cute.math.exp2(
                    (row_max_prev - row_max_new_safe) * softmax_scale_log2,
                    fastmath=True,
                )
            )
            row_sum_new = (
                0.0
                if const_expr(is_first_n_block)
                else sRowScale[global_sum_row, query_row] * old_o_scale
            )
            for warp_idx_it in cutlass.range_constexpr(num_qk_warps):
                warp_scale = cute.math.exp2(
                    (sRowScale[warp_idx_it, query_row] - row_max_new_safe)
                    * softmax_scale_log2,
                    fastmath=True,
                )
                sRowScale[warp_scale_base + warp_idx_it, query_row] = warp_scale
                row_sum_new += (
                    sRowScale[local_sum_base + warp_idx_it, query_row] * warp_scale
                )
            sRowScale[global_max_row, query_row] = row_max_new
            sRowScale[global_sum_row, query_row] = row_sum_new
            if const_expr(not is_first_n_block):
                sRowScale[old_o_scale_row, query_row] = old_o_scale
        cute.arch.fence_view_async_shared()

        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.PEmpty),
            number_of_threads=self.num_mma_threads,
        )
        for r in cutlass.range(num_query_rows, unroll_full=True):
            query_row = tScS_qk[r, 0][1]
            warp_scale = sRowScale[warp_scale_base + warp_idx, query_row]
            acc_S_qk[r, None].store(acc_S_qk[r, None].load() * warp_scale)
        for r in cutlass.range(num_query_rows, unroll_full=True):
            query_row = tScS_qk[r, 0][1]
            for c in cutlass.range(cute.size(acc_S_qk, mode=[1]), unroll_full=True):
                key_row = key_row_base + tScS_qk[r, c][0]
                sP[query_row, key_row, p_stage] = self.dtype(acc_S_qk[r, c])
        cute.arch.fence_view_async_shared()

        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.PFull),
            number_of_threads=self.num_mma_threads,
        )
        # acc_O is zero-initialized immediately before the first KV tile, so
        # multiplying it by that tile's compile-time unit scale is pure
        # overhead. Later tiles still rescale accumulated output before PV.
        if const_expr(not is_first_n_block):
            self._rescale_transposed_o(
                acc_O,
                tiled_mma_pv,
                tidx,
                sRowScale,
                old_o_scale_row,
            )

        v_wait_token = pipeline_v.consumer_try_wait(consumer_state)
        pipeline_v.consumer_wait(consumer_state, v_wait_token)

        self._gemm_n8(
            tiled_mma_pv,
            acc_O,
            tOrV,
            tOrP,
            tVsV[None, None, None, p_stage],
            tPsP[None, None, None, p_stage],
            smem_thr_copy_V,
            smem_thr_copy_P,
        )
        pipeline_v.consumer_release(consumer_state)

        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.PEmpty),
            number_of_threads=self.num_mma_threads,
        )
        consumer_state.advance()
        return consumer_state

    @cute.jit
    def _store_transposed_output(
        self,
        acc_O: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sLSE: cute.Tensor,
        tiled_mma_pv: cute.TiledMma,
        tidx: Int32,
        seqlen: SeqlenInfoQK,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
        split_idx: Int32,
    ):
        row_limit = seqlen.seqlen_q * self.qhead_per_kvhead
        warp_idx = tidx // cute.arch.WARP_SIZE
        lane_idx = tidx % cute.arch.WARP_SIZE
        thr_mma_pv = tiled_mma_pv.get_slice(lane_idx)
        acc_O_qd = layout_utils.reshape_acc_to_mn(acc_O, transpose=True)
        cO = cute.make_identity_tensor((64, self.query_mma_n))
        tOcO_qd = layout_utils.reshape_acc_to_mn(
            thr_mma_pv.partition_C(cO), transpose=True
        )
        mO_cur = (
            seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx, split_idx]
            if const_expr(self.is_split_kv)
            else seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, head_idx]
        )
        for r in cutlass.range(cute.size(acc_O_qd, mode=[0]), unroll_full=True):
            query_row_local = tOcO_qd[r, 0][1]
            query_row = m_block * self.tile_m + query_row_local
            if query_row < row_limit:
                for c in cutlass.range(cute.size(acc_O_qd, mode=[1]), unroll_full=True):
                    value_col = warp_idx * const_expr(64) + tOcO_qd[r, c][0]
                    mO_cur[query_row, value_col] = self.dtype(acc_O_qd[r, c])

        if const_expr(mLSE is not None):
            if tidx < self.query_mma_n:
                query_row = m_block * self.tile_m + tidx
                if query_row < row_limit:
                    mLSE_cur = (
                        seqlen.offset_batch_Q(mLSE, batch_idx, dim=2)[
                            None, head_idx, split_idx
                        ]
                        if const_expr(self.is_split_kv)
                        else seqlen.offset_batch_Q(mLSE, batch_idx, dim=2)[
                            None, head_idx
                        ]
                    )
                    mLSE_cur[query_row] = sLSE[tidx]

    @cute.jit
    def mma(
        self,
        mQ: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sO: cute.Tensor,
        sP: Optional[cute.Tensor],
        sRowScale: Optional[cute.Tensor],
        sLSE: Optional[cute.Tensor],
        learnable_sink: Optional[cute.Tensor],
        pipeline_k: PipelineAsync,
        pipeline_v: PipelineAsync,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tidx: Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        consumer_state: PipelineState,
        block_info: BlockInfo,
        seqlen: SeqlenInfoQK,
        n_block_min: Int32,
        n_block_max: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
        split_idx: Int32,
        is_qk_owner: cutlass.Constexpr[bool],
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
    ):
        assert self.paged_kv
        assert self.pack_gqa
        assert self.is_causal or self.is_local
        assert self.score_mod is None
        assert self.mask_mod is None
        assert self.tile_m == 16 and self.tile_n == 64
        assert self.tile_hdim == 256
        assert self.tile_hdimv in (64, 256)
        assert self.qhead_per_kvhead <= self.query_mma_n
        assert is_qk_owner

        mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
        warp_idx = tidx // cute.arch.WARP_SIZE
        lane_idx = tidx % cute.arch.WARP_SIZE
        thr_mma_qk = tiled_mma_qk.get_slice(lane_idx)
        thr_mma_pv = tiled_mma_pv.get_slice(lane_idx)
        acc_shape_O = thr_mma_pv.partition_shape_C((64, self.query_mma_n))
        acc_O = cute.make_rmem_tensor(acc_shape_O, Float32)
        acc_O.fill(0.0)

        smem_copy_atom_k_major = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self.dtype,
        )
        smem_copy_atom_v_major = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            self.dtype,
        )
        smem_thr_copy_K = utils.make_tiled_copy_A(
            smem_copy_atom_k_major, tiled_mma_qk
        ).get_slice(lane_idx)
        smem_thr_copy_Q = utils.make_tiled_copy_B(
            smem_copy_atom_k_major, tiled_mma_qk
        ).get_slice(lane_idx)
        smem_thr_copy_V = utils.make_tiled_copy_A(
            smem_copy_atom_v_major, tiled_mma_pv
        ).get_slice(lane_idx)
        smem_thr_copy_P = utils.make_tiled_copy_B(
            smem_copy_atom_k_major, tiled_mma_pv
        ).get_slice(lane_idx)

        sK_warp = cute.local_tile(
            sK,
            (16, self.tile_hdim, self._num_k_stages()),
            (warp_idx, 0, 0),
        )
        sQ_query = cute.local_tile(sQ, (self.query_mma_n, self.tile_hdim), (0, 0))
        sP_query = cute.local_tile(
            sP,
            (self.query_mma_n, self.tile_n, self._num_p_stages()),
            (0, 0, 0),
        )
        tSrK = thr_mma_qk.make_fragment_A(
            thr_mma_qk.partition_A(sK_warp[None, None, 0])
        )
        tSrQ = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sQ_query))
        sV_warp = cute.local_tile(
            sV,
            (64, self.tile_n, self._num_v_stages()),
            (warp_idx, 0, 0),
        )
        tOrV = thr_mma_pv.make_fragment_A(
            thr_mma_pv.partition_A(sV_warp[None, None, 0])
        )
        tOrP = thr_mma_pv.make_fragment_B(
            thr_mma_pv.partition_B(sP_query[None, None, 0])
        )
        tVsV = smem_thr_copy_V.partition_S(sV_warp)
        tKsK = smem_thr_copy_K.partition_S(sK_warp)
        tQsQ = smem_thr_copy_Q.partition_S(sQ_query)
        tPsP = smem_thr_copy_P.partition_S(sP_query)

        PackGQA(
            self.tile_m,
            self.tile_hdim,
            self.check_hdim_oob,
            self.qhead_per_kvhead,
        ).load_Q(
            mQ_cur,
            sQ,
            gmem_tiled_copy_Q,
            tidx,
            m_block,
            seqlen.seqlen_q,
        )
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier(
            barrier_id=1,
            number_of_threads=self.num_Q_load_threads,
        )

        if const_expr(self.query_in_regs):
            # Q.T is invariant across all KV tiles and its N=8 fragment is
            # small. Load it while the first K TMA is in flight, then reuse it
            # instead of reloading Q from SMEM per tile.
            tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
            for k in cutlass.range_constexpr(cute.size(tQsQ.shape[2])):
                cute.copy(
                    smem_thr_copy_Q,
                    tQsQ[None, None, k],
                    tSrQ_copy_view[None, None, k],
                )

        n_block = cutlass.max(n_block_max - 1, n_block_min)
        consumer_state = self._compute_one_n_block_transposed(
            n_block,
            consumer_state,
            acc_O,
            sQ,
            sK,
            sV,
            sP,
            sRowScale,
            pipeline_k,
            pipeline_v,
            tiled_mma_qk,
            tiled_mma_pv,
            smem_thr_copy_K,
            smem_thr_copy_Q,
            smem_thr_copy_V,
            smem_thr_copy_P,
            tSrK,
            tSrQ,
            tOrV,
            tOrP,
            tKsK,
            tQsQ,
            tVsV,
            tPsP,
            tidx,
            softmax_scale_log2,
            seqlen,
            block_info.window_size_left,
            is_first_n_block=True,
        )
        for n_tile in cutlass.range(n_block - n_block_min, unroll=1):
            consumer_state = self._compute_one_n_block_transposed(
                n_block - n_tile - 1,
                consumer_state,
                acc_O,
                sQ,
                sK,
                sV,
                sP,
                sRowScale,
                pipeline_k,
                pipeline_v,
                tiled_mma_qk,
                tiled_mma_pv,
                smem_thr_copy_K,
                smem_thr_copy_Q,
                smem_thr_copy_V,
                smem_thr_copy_P,
                tSrK,
                tSrQ,
                tOrV,
                tOrP,
                tKsK,
                tQsQ,
                tVsV,
                tPsP,
                tidx,
                softmax_scale_log2,
                seqlen,
                block_info.window_size_left,
            )

        global_max_row = const_expr(8)
        global_sum_row = const_expr(9)
        final_scale_row = const_expr(0)
        row_limit = seqlen.seqlen_q * self.qhead_per_kvhead
        if tidx < self.query_mma_n:
            query_row = tidx
            row_max = sRowScale[global_max_row, query_row]
            row_sum = sRowScale[global_sum_row, query_row]
            if query_row < row_limit:
                if const_expr(learnable_sink is not None):
                    if const_expr(not self.is_split_kv) or split_idx == 0:
                        q_head_idx = query_row + head_idx * self.qhead_per_kvhead
                        sink_val = Float32(learnable_sink[q_head_idx])
                        log2_e = math.log2(math.e)
                        if row_max == -Float32.inf:
                            row_max = sink_val * (log2_e / softmax_scale_log2)
                            row_sum = 1.0
                        else:
                            row_sum += cute.math.exp2(
                                sink_val * log2_e - row_max * softmax_scale_log2,
                                fastmath=True,
                            )
                row_sum_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
                sRowScale[final_scale_row, query_row] = cute.arch.rcp_approx(
                    row_sum if not row_sum_is_zero_or_nan else 1.0
                )
                sLSE[query_row] = (
                    (
                        row_max * softmax_scale_log2
                        + cute.math.log2(row_sum, fastmath=True)
                    )
                    * math.log(2.0)
                    if not row_sum_is_zero_or_nan
                    else -Float32.inf
                )
            else:
                sRowScale[final_scale_row, query_row] = 1.0
                sLSE[query_row] = -Float32.inf
        cute.arch.fence_view_async_shared()
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.PFull),
            number_of_threads=self.num_mma_threads,
        )
        self._rescale_transposed_o(
            acc_O,
            tiled_mma_pv,
            tidx,
            sRowScale,
            final_scale_row,
        )
        self._store_transposed_output(
            acc_O,
            mO,
            mLSE,
            sLSE,
            tiled_mma_pv,
            tidx,
            seqlen,
            m_block,
            head_idx,
            batch_idx,
            split_idx,
        )
