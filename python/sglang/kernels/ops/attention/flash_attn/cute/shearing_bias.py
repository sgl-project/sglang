# Copyright (c) 2026, Colfax International.

import math
from functools import partial
from typing import Callable, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

from sglang.kernels.ops.attention.flash_attn.cute.block_info import BlockInfo
from sglang.kernels.ops.attention.flash_attn.cute.copy_utils import tiled_copy_2d
from sglang.kernels.ops.attention.flash_attn.cute.cute_dsl_utils import (
    assume_tensor_aligned,
)
from sglang.kernels.ops.attention.flash_attn.cute.pack_gqa import pack_gqa_layout
from sglang.kernels.ops.attention.flash_attn.cute.seqlen_info import SeqlenInfoQK
from sglang.kernels.ops.attention.flash_attn.cute.tile_scheduler import (
    ParamsBase,
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
)
from sglang.kernels.ops.attention.flash_attn.cute.utils import get_batch_from_cu_tensor


class ShearingBias:
    def __init__(
        self,
        rel_extent: int = 512,
        is_causal: bool = True,
        is_local: bool = False,
        pack_gqa: bool = False,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        rows_per_cta: int = 4,
        tile_m: int = 128,
        max_m_blocks_leq_one: bool = False,
        use_pdl: bool = False,
        clamp_subtiles: bool = True,
    ):
        self.is_causal = is_causal
        self.is_local = is_local
        assert is_causal or is_local, "Doesn't make sense otherwise"
        self.pack_gqa = pack_gqa
        self.qhead_per_kvhead = qhead_per_kvhead
        if self.pack_gqa:
            assert (
                128 % self.qhead_per_kvhead == 0
            ), "pack_gqa only supported when qhead_per_kvhead divides 128"
        self.qhead_per_kvhead_packgqa = qhead_per_kvhead if self.pack_gqa else 1
        self.rel_extent = rel_extent
        assert rel_extent % 128 == 0
        self.rel_extent_padded = rel_extent + 256
        self.num_bias_blocks_padded = (self.rel_extent_padded) // 128
        # tuneable parameters
        assert rows_per_cta % 4 == 0
        self.rows_per_cta = rows_per_cta
        self.num_threads = self.rows_per_cta * 32
        self.cta_tiler = (self.rows_per_cta, self.rel_extent)
        self.cta_out_tiler = (self.rows_per_cta, self.rel_extent_padded)

        self.buffer_align_bytes = 1024

        self.max_m_blocks_leq_one = max_m_blocks_leq_one
        self.use_pdl = use_pdl

        # only used with block packed scheduling
        self.tile_m = tile_m
        # Shrink the subtile grid dim to the rows a block can actually hold
        # (decode blocks hold qhead_per_kvhead*seqlen_q rows, not tile_m).
        self.clamp_subtiles = clamp_subtiles

    @cute.jit
    def __call__(
        self,
        mPreBias: cute.Tensor,  # (b, s_q, h, rel_extent) or (total_q, h, rel_extent)
        mBias: cute.Tensor,  # (b, s_q, h, rel_extent_padded) or (total_q, h, rel_extent_padded)
        max_seqlen_q: Int32 | int,
        max_seqlen_k: Int32 | int,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mCuTotalMBlocks: Optional[cute.Tensor] = None,
        mBlocksToBatchIdx: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        assert mPreBias.element_type == mBias.element_type
        self.bias_dtype = mBias.element_type

        right_pad_value = -Float32.inf
        left_pad_value = (
            -Float32.inf if const_expr(window_size_left is not None) else 0.0
        )

        self.vec_size = 32 // self.bias_dtype.width
        self.cols_per_iter = 32 * self.vec_size
        assert self.vec_size <= 2
        assert 128 % self.cols_per_iter == 0

        max_seqlen_k = Int32(max_seqlen_k)
        if const_expr(window_size_left is not None):
            window_size_left = Int32(window_size_left)
        if const_expr(window_size_right is not None):
            window_size_right = Int32(window_size_right)

        mPreBias, mBias = [assume_tensor_aligned(t) for t in (mPreBias, mBias)]
        # (s_q, rel_extent, h, b) or (total_q, rel_extent, h)
        Q_layout_transpose = (
            [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        )
        mPreBias, mBias = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=Q_layout_transpose))
            for t in (mPreBias, mBias)
        ]

        if const_expr(self.pack_gqa):
            nheads_kv = mBias.shape[2] // self.qhead_per_kvhead
            mPreBias, mBias = [
                pack_gqa_layout(t, self.qhead_per_kvhead, nheads_kv, head_idx=2)
                for t in (mPreBias, mBias)
            ]

        # SMEM layouts
        prebias_tile_shape = (self.rows_per_cta, self.rel_extent)
        bias_tile_shape = (self.rows_per_cta, self.rel_extent_padded)

        sPreBias_layout = cute.make_ordered_layout(prebias_tile_shape, order=(1, 0))
        sBias_layout = cute.make_ordered_layout(bias_tile_shape, order=(1, 0))
        sPreBias_size = cute.cosize(sPreBias_layout)
        sBias_size = cute.cosize(sBias_layout)

        in_major_size = math.gcd(256, self.rel_extent)
        assert in_major_size % 128 == 0
        self.num_g2s_threads = self.num_threads if in_major_size == 256 else 128
        g2s_tiled_copy = tiled_copy_2d(
            self.bias_dtype,
            math.gcd(256, self.rel_extent),
            self.num_g2s_threads,
            is_async=True,
        )

        out_major_size = math.gcd(256, self.rel_extent_padded)
        assert out_major_size % 128 == 0
        self.num_s2g_threads = self.num_threads if out_major_size == 256 else 128
        s2g_tiled_copy = tiled_copy_2d(
            self.bias_dtype,
            math.gcd(256, self.rel_extent_padded),
            self.num_s2g_threads,
        )

        @cute.struct
        class SharedStorage:
            sPreBias: cute.struct.Align[
                cute.struct.MemRange[self.bias_dtype, sPreBias_size],
                self.buffer_align_bytes,
            ]
            sBias: cute.struct.Align[
                cute.struct.MemRange[self.bias_dtype, sBias_size],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        varlen_q = mCuSeqlensQ is not None or mSeqUsedQ is not None
        self.use_block_packed_scheduling = (
            mCuTotalMBlocks is not None
            and mCuSeqlensQ is not None
            and not self.max_m_blocks_leq_one
            # and False
        )

        if const_expr(varlen_q and not self.max_m_blocks_leq_one):
            if const_expr(self.use_block_packed_scheduling):
                TileScheduler = SingleTileScheduler
            else:
                TileScheduler = SingleTileVarlenScheduler
        else:
            TileScheduler = SingleTileScheduler

        batch_size = (
            cute.size(mPreBias.shape[3])
            if const_expr(mCuSeqlensQ is None)
            else cute.size(mCuSeqlensQ.shape[0] - 1)
        )
        eff_seqlen_q = (
            max_seqlen_q
            if const_expr(not self.pack_gqa)
            else max_seqlen_q * self.qhead_per_kvhead
        )
        total_q = (
            cute.size(mPreBias.shape[0])
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mPreBias.shape[0]) * cute.size(mPreBias.shape[3])
        )
        # same formula as in varlen scheduler -- only used with block packed scheduling
        total_blocks_max = (total_q + batch_size * (self.tile_m - 1)) // self.tile_m

        num_blocks_for_sched = (
            cute.ceil_div(eff_seqlen_q, self.rows_per_cta)
            if const_expr(not self.use_block_packed_scheduling)
            else total_blocks_max
        )
        if const_expr(not self.use_block_packed_scheduling):
            batch_size_for_sched = batch_size
        elif const_expr(self.clamp_subtiles):
            # A block covers at most min(tile_m, eff_seqlen_q) valid rows; subtiles
            # past that would fail the per-row seqlen guards and exit immediately.
            batch_size_for_sched = cute.ceil_div(
                min(self.tile_m, eff_seqlen_q), self.rows_per_cta
            )
        else:
            batch_size_for_sched = self.tile_m // self.rows_per_cta

        tile_sched_args = TileSchedulerArguments(
            num_blocks_for_sched,
            cute.size(mPreBias.shape[2]),
            batch_size_for_sched,
            1,
            1,
            1,
            1,
            total_q=total_q,
            tile_shape_mn=self.cta_tiler,
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead_packgqa,
            element_size=self.bias_dtype.width // 8,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        self.kernel(
            mPreBias,
            mBias,
            left_pad_value,
            right_pad_value,
            max_seqlen_k,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mCuTotalMBlocks,
            mBlocksToBatchIdx,
            sPreBias_layout,
            sBias_layout,
            window_size_left,
            window_size_right,
            g2s_tiled_copy,
            s2g_tiled_copy,
            SharedStorage,
            tile_sched_params,
        ).launch(
            grid=grid_dim,
            block=(self.num_threads, 1, 1),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mPreBias: cute.Tensor,
        mBias: cute.Tensor,
        left_pad_value: cutlass.Float32,
        right_pad_value: cutlass.Float32,
        max_seqlen_k: Int32,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mCuTotalMBlocks: Optional[cute.Tensor],
        mBlocksToBatchIdx: Optional[cute.Tensor],
        sPreBias_layout: cute.ComposedLayout | cute.Layout,
        sBias_layout: cute.ComposedLayout | cute.Layout,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        g2s_tiled_copy: cute.TiledCopy,
        s2g_tiled_copy: cute.TiledCopy,
        SharedStorage: cutlass.Constexpr[Callable],
        tile_sched_params: ParamsBase,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        sPreBias = storage.sPreBias.get_tensor(sPreBias_layout)
        sBias = storage.sBias.get_tensor(sBias_layout)

        TileSchedulerCls = partial(self.tile_scheduler_cls.create, tile_sched_params)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        # if pack_gqa, head_idx means head_idx_kv
        m_block, head_idx, batch_idx, _ = work_tile.tile_idx
        subtile_idx = batch_idx if const_expr(self.use_block_packed_scheduling) else 0

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

        is_valid_tile = work_tile.is_valid_tile
        if const_expr(self.use_block_packed_scheduling):
            batch_size = mCuTotalMBlocks.shape[0] - 1
            is_valid_tile = m_block < mCuTotalMBlocks[batch_size]

        if is_valid_tile:
            if const_expr(self.use_block_packed_scheduling):
                if const_expr(mBlocksToBatchIdx is not None):
                    batch_idx = mBlocksToBatchIdx[m_block]
                else:
                    batch_idx = get_batch_from_cu_tensor(m_block, mCuTotalMBlocks)
                # get local m_block for batch
                m_block -= mCuTotalMBlocks[batch_idx]
                m_block = m_block * (self.tile_m // self.rows_per_cta) + subtile_idx
            seqlen_info = SeqlenInfoQK.create(
                batch_idx=batch_idx,
                seqlen_q_static=(
                    mPreBias.shape[0]
                    if const_expr(not self.pack_gqa)
                    else mPreBias.shape[0][1]
                ),
                seqlen_k_static=max_seqlen_k,
                mCuSeqlensQ=mCuSeqlensQ,
                mCuSeqlensK=mCuSeqlensK,
                mSeqUsedQ=mSeqUsedQ,
                mSeqUsedK=mSeqUsedK,
            )

            block_info = BlockInfo(
                128,
                128,
                self.is_causal,
                self.is_local,
                window_size_left=window_size_left,
                window_size_right=window_size_right,
                qhead_per_kvhead_packgqa=self.qhead_per_kvhead_packgqa,
            )

            # (seqlen, rel_extent) or ((seqlen, qhead_per_kvhead), rel_extent)
            mPreBias_cur = seqlen_info.offset_batch_Q(mPreBias, batch_idx, dim=3)[
                None, None, head_idx
            ]
            # (rows_per_cta, rel_extent)
            gPreBias = cute.local_tile(mPreBias_cur, self.cta_tiler, (m_block, 0))
            cPreBias = cute.make_identity_tensor(self.cta_tiler)

            g2s_thr_copy = g2s_tiled_copy.get_slice(tidx)

            # (V, M, N)
            tBgPreBias = g2s_thr_copy.partition_S(gPreBias)
            tBsPreBias = g2s_thr_copy.partition_D(sPreBias)
            tBcPreBias = g2s_thr_copy.partition_S(cPreBias)

            if (
                const_expr(self.num_g2s_threads == self.num_threads)
                or warp_idx < self.num_g2s_threads // 32
            ):
                num_rows_per_load = tBgPreBias.shape[1]
                for m in cutlass.range_constexpr(num_rows_per_load):
                    local_m_idx = tBcPreBias[0, m, 0][0]
                    load_m_idx = local_m_idx + m_block * self.rows_per_cta
                    local_m_idx_in_bounds = (
                        const_expr(self.rows_per_cta % 8 == 0)
                        or local_m_idx < self.rows_per_cta
                    )
                    load_m_idx_in_bounds = (
                        load_m_idx // self.qhead_per_kvhead_packgqa
                        < seqlen_info.seqlen_q
                    )
                    if local_m_idx_in_bounds and load_m_idx_in_bounds:
                        cute.copy(
                            g2s_tiled_copy,
                            tBgPreBias[None, m, None],
                            tBsPreBias[None, m, None],
                        )

            cute.arch.cp_async_commit_group()

            # Convention: inclusive min, exclusive max
            m_idx = m_block * self.rows_per_cta + warp_idx
            attn_m_block = m_idx // 128

            _, attn_n_block_max = block_info.get_n_block_min_max(
                seqlen_info,
                attn_m_block,
            )

            n_idx_left, n_idx_right = block_info.get_n_idx_left_right(
                seqlen_info, m_idx
            )
            num_bias_vals = n_idx_right - max(n_idx_left, n_idx_right - self.rel_extent)
            is_even = n_idx_right % 2 == 0

            # get bias block and idx bounds for row
            n_block_for_rel0 = (n_idx_right - 1) // 128  # inclusive
            bias_block_idx_right = 1 + max(
                self.rel_extent_padded // 128 - (attn_n_block_max - n_block_for_rel0), 0
            )
            bias_idx_right = (
                (bias_block_idx_right - 1) * 128 + ((n_idx_right - 1) % 128) + 1
            )
            bias_idx_left = max(0, bias_idx_right - num_bias_vals)
            bias_block_idx_left = bias_idx_left // 128
            # num_bias_blocks = self.num_bias_blocks_padded - bias_block_idx_left
            # num_right_padding_blocks = 0
            num_bias_blocks = (
                bias_block_idx_right - bias_block_idx_left if num_bias_vals > 0 else 0
            )
            num_right_padding_blocks = (
                self.num_bias_blocks_padded - bias_block_idx_right
                if num_bias_vals > 0
                else self.num_bias_blocks_padded
            )
            # might help compiler unroll loops
            num_bias_blocks = min(num_bias_blocks, self.num_bias_blocks_padded)
            num_right_padding_blocks = min(
                num_right_padding_blocks, self.num_bias_blocks_padded
            )

            sPreBias_row = cute.flat_divide(
                sPreBias[(warp_idx, None)], (self.vec_size,)
            )
            sBias_row = cute.flat_divide(sBias[(warp_idx, None)], (self.vec_size,))
            sBias_row_vec4 = cute.flat_divide(sBias[(warp_idx, None)], (4,))

            bias_idx = (
                self.rel_extent_padded + lane_idx * self.vec_size - self.cols_per_iter
            )

            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            if m_idx // self.qhead_per_kvhead_packgqa < seqlen_info.seqlen_q:
                # We can try handling right padding separately
                for i in cutlass.range(num_right_padding_blocks, unroll_full=True):
                    bias_frg = cute.make_rmem_tensor((4,), dtype=self.bias_dtype)
                    bias_frg.fill(self.bias_dtype(right_pad_value))
                    bias_right_pad_idx = (
                        self.num_bias_blocks_padded - 1 - i
                    ) * 32 + lane_idx
                    cute.autovec_copy(
                        bias_frg, sBias_row_vec4[None, bias_right_pad_idx]
                    )
                    bias_idx -= 128

                for _ in cutlass.range(num_bias_blocks, unroll_full=True):
                    # 2 subblocks for half bias dtype
                    for _ in cutlass.range_constexpr(128 // self.cols_per_iter):
                        prebias_idx = bias_idx_right - 1 - bias_idx

                        # (vec_size, lower/upper)
                        prebias_frg = cute.make_rmem_tensor(
                            (self.vec_size, self.vec_size), dtype=self.bias_dtype
                        )

                        in_bounds = (
                            prebias_idx >= 0
                            and prebias_idx - self.vec_size + 1 < num_bias_vals
                        )
                        prebias_idx_lower = (
                            prebias_idx - 1 if is_even else max(prebias_idx - 2, 0)
                        )
                        prebias_idx_upper = (
                            prebias_idx - 1
                            if is_even
                            else min(prebias_idx, self.rel_extent - 2)
                        )

                        if in_bounds:
                            cute.autovec_copy(
                                sPreBias_row[None, prebias_idx_lower // 2],
                                prebias_frg[None, 0],
                            )
                            if const_expr(self.vec_size == 2) and not is_even:
                                cute.autovec_copy(
                                    sPreBias_row[None, prebias_idx_upper // 2],
                                    prebias_frg[None, 1],
                                )

                        bias_frg = cute.make_rmem_tensor(
                            (self.vec_size,), dtype=self.bias_dtype
                        )
                        bias_frg.fill(self.bias_dtype(left_pad_value))

                        if const_expr(self.vec_size == 1):
                            if in_bounds:
                                bias_frg[0] = prebias_frg[0, 0]
                            elif prebias_idx < 0:
                                bias_frg.fill(self.bias_dtype(right_pad_value))
                        else:
                            if in_bounds:
                                if is_even:
                                    # reverse: [prebias_idx, prebias_idx-1] = bias frg
                                    bias_frg[0] = prebias_frg[1, 0]
                                    bias_frg[1] = prebias_frg[0, 0]
                                else:
                                    # lower = [2x-2, 2x-1], upper = [2x, 2x+1], 2x = prebias_idx
                                    # want bias = [2x, 2x-1]
                                    bias_frg[0] = prebias_frg[0, 1]
                                    bias_frg[1] = prebias_frg[1, 0]
                            elif prebias_idx < 0:
                                bias_frg.fill(self.bias_dtype(right_pad_value))

                        cute.autovec_copy(bias_frg, sBias_row[None, bias_idx // 2])
                        bias_idx -= self.cols_per_iter
                        cute.arch.sync_warp()

                # Handle edge cases. For N = rel_extent:
                # [0, -1], -1 at bias_idx_right and [N, N-1], N-1 at bias_idx_left
                if not is_even and num_bias_vals > 0:
                    sBias[(warp_idx, bias_idx_right)] = self.bias_dtype(right_pad_value)
                    if bias_idx_left - 1 >= 0:
                        sBias[(warp_idx, bias_idx_left - 1)] = self.bias_dtype(
                            left_pad_value
                        )

                num_left_padding_blocks = min(
                    self.num_bias_blocks_padded
                    - num_bias_blocks
                    - num_right_padding_blocks,
                    self.num_bias_blocks_padded,
                )
                for i in cutlass.range(num_left_padding_blocks, unroll_full=True):
                    bias_left_pad_idx = i * 32 + lane_idx
                    bias_frg = cute.make_rmem_tensor((4,), dtype=self.bias_dtype)
                    bias_frg.fill(self.bias_dtype(left_pad_value))
                    cute.autovec_copy(bias_frg, sBias_row_vec4[None, bias_left_pad_idx])

            cute.arch.sync_threads()

            s2g_thr_copy = s2g_tiled_copy.get_slice(tidx)

            # (seqlen, rel_extent_padded)
            mBias_cur = seqlen_info.offset_batch_Q(mBias, batch_idx, dim=3)[
                None, None, head_idx
            ]
            # (rows_per_cta, rel_extent_padded)
            gBias = cute.local_tile(mBias_cur, self.cta_out_tiler, (m_block, 0))
            cBias = cute.make_identity_tensor(self.cta_out_tiler)

            # (V, M, N)
            tBsBias = s2g_thr_copy.partition_S(sBias)
            tBgBias = s2g_thr_copy.partition_D(gBias)
            tBcBias = s2g_thr_copy.partition_D(cBias)

            if (
                const_expr(self.num_s2g_threads == self.num_threads)
                or warp_idx < self.num_s2g_threads // 32
            ):
                num_rows_per_store = tBgBias.shape[1]
                for m in cutlass.range_constexpr(num_rows_per_store):
                    local_m_idx = tBcBias[0, m, 0][0]
                    store_m_idx = local_m_idx + m_block * self.rows_per_cta
                    local_m_idx_in_bounds = (
                        const_expr(self.rows_per_cta % 8 == 0)
                        or local_m_idx < self.rows_per_cta
                    )
                    store_m_idx_in_bounds = (
                        store_m_idx // self.qhead_per_kvhead_packgqa
                        < seqlen_info.seqlen_q
                    )
                    if local_m_idx_in_bounds and store_m_idx_in_bounds:
                        cute.copy(
                            s2g_tiled_copy,
                            tBsBias[None, m, None],
                            tBgBias[None, m, None],
                        )
