# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM Step5 CuTe Q/K Norm+RoPE; compute body retained from the source."""

from __future__ import annotations

from functools import partial
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

from sglang.kernels.ops.attention.flash_attn.cute import copy_utils
from sglang.kernels.ops.attention.flash_attn.cute.copy_utils import get_copy_atom

from . import norm_reduction as utils
from .norm_reduction import ReductionBase, row_reduce


class FusedQKNormRope(ReductionBase):
    def __init__(
        self,
        dtype: cutlass.Numeric,
        N: int,
        head_dim: int,
        num_q_head: int,
        num_kv_head: int,
        rotary_dim: int,
        input_size: Optional[int] = None,
    ):
        super().__init__(dtype, N, stage=1)
        self.reload_from = None if N <= 16384 else "smem"
        self.delay_w_load = False
        self.head_dim = head_dim
        self.num_q_head = num_q_head
        self.num_kv_head = num_kv_head
        self.rotary_dim = rotary_dim
        self.input_size = input_size if input_size is not None else N
        # Note(wangbojun/codex): row-group software pipelining. Each CTA handles
        # `row_stages` consecutive row groups with double-buffered smem so the
        # cp.async of the next group overlaps norm+RoPE+store of the current one.
        # Enable for hd192 where the extra smem (~3 KB/stage) stays well within
        # the H800 per-SM budget (228 KB) and register pressure is unchanged.
        self.row_stages = 3 if head_dim == 192 else 1

    def _calculate_threads_per_head(self):
        """Calculate the number of threads per row for the RMSNorm kernel."""
        N = self.head_dim
        if N <= 64:
            threads_per_head = 8
        elif N <= 128:
            threads_per_head = 16
        elif N <= 192:
            # Note(wangbojun/codex): keep hd192 on the 16-thread/head path first.
            # The default 32-thread/head heuristic regressed badly at 4096 tokens,
            # but wider RoPE still needs enough lanes to cover all rotary columns.
            threads_per_head = 16
        elif N <= 3072:
            threads_per_head = 32
        elif N <= 6144:
            threads_per_head = 64
        elif N <= 16384:
            threads_per_head = 128
        else:
            threads_per_head = 256

        vals_per_thread = 64 // self.dtype.width
        rotary_threads = self.rotary_dim // vals_per_thread
        while threads_per_head < rotary_threads:
            threads_per_head *= 2
        return threads_per_head

    def _get_num_threads(self):
        return 128 if self.N <= 16384 else 256

    def _get_num_copy_bits(self):
        # Note(wangbojun/codex): drop cp.async vec from 128b->64b for cases where
        # the smaller vec divides head_dim exactly with the chosen threads_per_head,
        # so tile_n == head_dim (no N-padding) and the post-cp.async row reduction
        # / FFMAs do not run on padded slots.
        # hd192, tpH=16 (rd<=64): vec=4, num_blocks=3, tile_n=192   -> no padding
        # hd192, tpH=32 (rd>=96): vec=4 still leaves padding -> stay on vec=8.
        vals_per_thread_64b = 64 // self.dtype.width
        if (
            self.head_dim % (self._calculate_threads_per_head() * vals_per_thread_64b)
            == 0
            and self.head_dim
            % (self._calculate_threads_per_head() * (128 // self.dtype.width))
            != 0
        ):
            copy_bits = 64
        else:
            copy_bits = 128

        # A fused [QKV|G] row is not necessarily 128-bit aligned. Select the
        # widest cp.async width that aligns every row while preserving the
        # original 64/128-bit choice for the QKV-only path.
        while copy_bits > 32 and self.input_size % (copy_bits // self.dtype.width) != 0:
            copy_bits //= 2
        assert self.input_size % (copy_bits // self.dtype.width) == 0
        return copy_bits

    def _get_tv_layout(self, num_copy_bits=128):
        vecsize = num_copy_bits // self.dtype.width  # 8
        assert self.head_dim % vecsize == 0, (
            f"Input N {self.head_dim} is not divisible by vector size {vecsize}"
        )
        num_threads = self._get_num_threads()
        assert num_threads % cute.arch.WARP_SIZE == 0

        threads_per_head = self._calculate_threads_per_head()  # 16
        num_blocks_N = cute.ceil_div(
            self.head_dim // vecsize, threads_per_head * self.cluster_n
        )  # 1
        cols_per_block = num_threads // threads_per_head  # 8
        tiler_mn = (cols_per_block, vecsize * num_blocks_N * threads_per_head)  # 8, 128
        tv_layout = cute.make_layout(
            ((threads_per_head, cols_per_block), (vecsize, num_blocks_N)),
            stride=(
                (vecsize * cols_per_block, 1),
                (cols_per_block, cols_per_block * vecsize * threads_per_head),
            ),
        )
        return tiler_mn, tv_layout

    def _set_cluster_n(self):
        """
        Set the number of clusters for the RMSNorm kernel.
        Stored in self.cluster_n.
        """
        N = self.head_dim

        # cluster_n = 4 is faster and cluster_n = 2 for N=64k for some reason
        # Similarly cluster_n = 8 is faster for N=128k
        if const_expr(self.dtype.width == 16):
            # 16-bit types (fp16, bf16)
            if N <= 16 * 1024:
                cluster_n = 1
            elif N <= 32 * 1024:
                cluster_n = 2
            elif N <= 64 * 1024:
                cluster_n = 4
            elif N <= 128 * 1024:
                cluster_n = 8
            else:
                cluster_n = 16
        else:
            # 32-bit types (fp32)
            if N <= 32 * 1024:
                cluster_n = 1
            elif N <= 64 * 1024:
                cluster_n = 2
            elif N <= 128 * 1024:
                cluster_n = 4
            elif N <= 256 * 1024:
                cluster_n = 8
            else:
                cluster_n = 16

        self.cluster_n = cluster_n

    @cute.jit
    def _scatter_cache_row(
        self,
        m_cache: cute.Tensor,
        m_src: cute.Tensor,
        m_loc: cute.Tensor,
        token,
        n_tok,
        tidx,
    ):
        """Copy one known-good gmem row ``m_src[token]`` to ``cache[loc]``.

        Uses this thread's scalar columns only. A tiled/autovec store of the
        (8, 192) fragment writes neighboring slots when ``loc`` is consecutive.
        """
        if token < n_tok:
            loc = m_loc[token]
            if loc >= 0:
                for col_off in cutlass.range_constexpr(2):
                    col = tidx + col_off * 128
                    if col < 192:
                        m_cache[loc, col] = m_src[token, col]

    def _smem_size_in_bytes(self, tiler_mn, num_warps):
        return (
            self.row_stages * cute.size_in_bytes(self.dtype, cute.make_layout(tiler_mn))
            + self.stage
            * num_warps
            * self.cluster_n
            * (self.reduction_dtype.width // 8)
            + self.stage * (cutlass.Int64.width // 8)
        )

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mQW: Optional[cute.Tensor],
        mKW: Optional[cute.Tensor],
        mCos: cute.Tensor,
        mSin: cute.Tensor,
        mPosId: cute.Tensor,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mG: Optional[cute.Tensor],
        mKCache: Optional[cute.Tensor],
        mVCache: Optional[cute.Tensor],
        mLoc: Optional[cute.Tensor],
        head_dim: Int32,
        num_q_head: Int32,
        num_kv_head: Int32,
        stream: cuda.CUstream,
        eps: Float32 = 1e-6,
        norm_weight_bias: Float32 = 1.0,
    ):
        semistatic_shape_X = (
            *mX.shape[:-1],
            self.num_q_head + self.num_kv_head * 2,
            self.head_dim,
        )

        input_vec_elems = self._get_num_copy_bits() // mX.element_type.width
        new_stride = lambda t: (
            cute.assume(t.stride[0], divby=input_vec_elems),
            cute.assume(t.stride[0], divby=input_vec_elems),
            t.stride[1],
        )

        mX = cute.make_tensor(
            mX.iterator, cute.make_layout(semistatic_shape_X, stride=new_stride(mX))
        )

        semistatic_shape_Q = (*mQ.shape[:-1], self.num_q_head, self.head_dim)
        semistatic_shape_KV = (*mK.shape[:-1], self.num_kv_head, self.head_dim)

        new_stride_QKV = lambda t: (
            cute.assume(t.stride[0], divby=self.head_dim),
            self.head_dim // (128 // t.element_type.width),
            t.stride[1],
        )
        mQ = cute.make_tensor(
            mQ.iterator, cute.make_layout(semistatic_shape_Q, stride=new_stride_QKV(mQ))
        )

        mK, mV = [
            (
                cute.make_tensor(
                    t.iterator,
                    cute.make_layout(semistatic_shape_KV, stride=new_stride_QKV(t)),
                )
                if const_expr(t is not None)
                else None
            )
            for t in (mK, mV)
        ]
        if const_expr(mG is not None):
            semistatic_shape_G = (*mG.shape[:-1], self.num_q_head)
            mG = cute.make_tensor(
                mG.iterator,
                cute.make_layout(semistatic_shape_G, stride=mG.stride),
            )
        if const_expr(mKCache is not None):
            cache_stride = lambda t: (
                cute.assume(t.stride[0], divby=self.head_dim),
                t.stride[1],
            )
            mKCache = cute.make_tensor(
                mKCache.iterator,
                cute.make_layout(
                    (mKCache.shape[0], self.head_dim), stride=cache_stride(mKCache)
                ),
            )
            mVCache = cute.make_tensor(
                mVCache.iterator,
                cute.make_layout(
                    (mVCache.shape[0], self.head_dim), stride=cache_stride(mVCache)
                ),
            )

        semistatic_shape_Cos_Sin = (*mCos.shape[:-1], self.rotary_dim)
        new_stride_Cos_Sin = lambda t: (
            cute.assume(t.stride[0], divby=64 // t.element_type.width),
            t.stride[1],
        )
        mCos = cute.make_tensor(
            mCos.iterator,
            cute.make_layout(semistatic_shape_Cos_Sin, stride=new_stride_Cos_Sin(mCos)),
        )
        mSin = cute.make_tensor(
            mSin.iterator,
            cute.make_layout(semistatic_shape_Cos_Sin, stride=new_stride_Cos_Sin(mSin)),
        )

        self._set_cluster_n()
        largest_dtype_width = const_expr(
            mX.element_type.width,
        )
        tiler_mn, tv_layout = self._get_tv_layout(
            num_copy_bits=self._get_num_copy_bits()
            // largest_dtype_width
            * mX.element_type.width
        )

        num_threads = cute.size(tv_layout, mode=[0])
        num_warps = num_threads // cute.arch.WARP_SIZE
        if const_expr(mQW is not None):
            mQW_expanded_layout = cute.prepend(
                mQW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))
            )
            mQW = cute.make_tensor(mQW.iterator, mQW_expanded_layout)
        if const_expr(mKW is not None):
            mKW_expanded_layout = cute.prepend(
                mKW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))
            )
            mKW = cute.make_tensor(mKW.iterator, mKW_expanded_layout)

        self.kernel(
            mX,
            mQW,
            mKW,
            mCos,
            mSin,
            mPosId,
            mQ,
            mK,
            mV,
            mG,
            mKCache,
            mVCache,
            mLoc,
            num_q_head,
            num_kv_head,
            eps,
            norm_weight_bias,
            self.reload_from,
        ).launch(
            grid=[
                cute.ceil_div(mX.shape[0], tiler_mn[0] * self.row_stages),
                self.cluster_n,
                self.num_q_head + self.num_kv_head * 2,
            ],
            block=[num_threads, 1, 1],
            cluster=(
                [1, self.cluster_n, 1] if const_expr(self.cluster_n > 1) else None
            ),
            smem=self._smem_size_in_bytes(
                tiler_mn,
                num_warps,
            ),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mQW: Optional[cute.Tensor],
        mKW: Optional[cute.Tensor],
        mCos: cute.Tensor,
        mSin: cute.Tensor,
        mPosId: cute.Tensor,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mG: Optional[cute.Tensor],
        mKCache: Optional[cute.Tensor],
        mVCache: Optional[cute.Tensor],
        mLoc: Optional[cute.Tensor],
        num_q_head: Int32,
        num_kv_head: Int32,
        eps: cute.Float32,
        norm_weight_bias: cute.Float32,
        reload_from: cutlass.Constexpr = None,
        delay_w_load: cutlass.Constexpr = False,
    ):
        largest_dtype_width = cutlass.const_expr(mX.element_type.width)
        tiler_mn, tv_layout = self._get_tv_layout(
            num_copy_bits=self._get_num_copy_bits()
            // largest_dtype_width
            * mX.element_type.width
        )
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, bidz = cute.arch.block_idx()

        VALS_PER_THREAD = 64 // mCos.element_type.width  # 4
        rotary_threads = self.rotary_dim // VALS_PER_THREAD
        LANES_PER_HEAD = tv_layout.shape[0][0]
        num_threads = cute.size(tv_layout, mode=[0])
        NUM_LANES = num_threads // LANES_PER_HEAD
        rowid_in_block = tidx // LANES_PER_HEAD

        if const_expr(self.cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = const_expr(0)

        ROW_STAGES = cutlass.const_expr(self.row_stages)

        # Allocate smem: ROW_STAGES X buffers + reduction buffer/mbar.
        smem = cutlass.utils.SmemAllocator()
        sX_stages = []
        for _stage in cutlass.range_constexpr(ROW_STAGES):
            sX_stages.append(
                smem.allocate_tensor(
                    mX.element_type,
                    cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                    byte_alignment=16,
                )
            )

        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(
            smem, tv_layout
        )

        shape_X = (mX.shape[0] * mX.shape[1], mX.shape[2])
        idX = cute.make_identity_tensor(shape_X)

        # slice for CTAs
        # We use domain_offset_i64 to deal with tensors larger than 2^31 elements
        num_qkv_heads = num_q_head + num_kv_head * 2

        if const_expr(mG is not None):
            mGateInput = cute.make_tensor(
                mX.iterator + self.N,
                cute.make_layout(
                    (mX.shape[0], self.num_q_head),
                    stride=(mX.stride[0], mX.stride[2]),
                ),
            )

        # Adjust base iterators for this head (bidz). All stages reuse these.
        copy_alignment = self._get_num_copy_bits() // 8
        new_iterator = (mX.iterator + bidz * mX.shape[2]).align(copy_alignment)
        mX = cute.make_tensor(
            new_iterator,
            cute.make_layout(
                (mX.shape[0], mX.shape[2]), stride=(mX.stride[0], mX.stride[2])
            ),
        )
        new_iterator = (mQ.iterator + bidz * mQ.shape[2]).align(copy_alignment)
        mQ = cute.make_tensor(
            new_iterator,
            cute.make_layout(
                (mQ.shape[0], mQ.shape[2]), stride=(mQ.stride[0], mQ.stride[2])
            ),
        )
        new_iterator = (mK.iterator + (bidz - self.num_q_head) * mK.shape[2]).align(
            copy_alignment
        )
        mK = cute.make_tensor(
            new_iterator,
            cute.make_layout(
                (mK.shape[0], mK.shape[2]), stride=(mK.stride[0], mK.stride[2])
            ),
        )
        new_iterator = (
            mV.iterator + (bidz - self.num_q_head - self.num_kv_head) * mV.shape[2]
        ).align(copy_alignment)
        mV = cute.make_tensor(
            new_iterator,
            cute.make_layout(
                (mV.shape[0], mV.shape[2]), stride=(mV.stride[0], mV.stride[2])
            ),
        )

        # declare the atoms which will be used later for memory copy
        num_copy_elems_X = tv_layout.shape[1][0]

        copy_atom_load_X_async = get_copy_atom(
            mX.element_type, num_copy_elems_X, is_async=True
        )

        thr_copy_X = cute.make_tiled_copy(
            copy_atom_load_X_async,
            tv_layout,
            tiler_mn,
        ).get_slice(tidx)

        # Shared QW / KW load (same head across stages)
        gQW = cute.local_tile(mQW, tiler_mn, (0, cluster_y))
        tXgQW = thr_copy_X.partition_S(gQW)
        tXrQW = cute.make_fragment_like(tXgQW)
        tXrQW.fill(0.0)

        gKW = cute.local_tile(mKW, tiler_mn, (0, cluster_y))
        tXgKW = thr_copy_X.partition_S(gKW)
        tXrKW = cute.make_fragment_like(tXgKW)
        tXrKW.fill(0.0)

        # Reused scratch register fragment (serial across stages)
        gX_tmpl = cute.local_tile(mX, tiler_mn, (0, cluster_y))
        tXgX_tmpl = thr_copy_X.partition_S(gX_tmpl)
        tXrX = cute.make_fragment_like(tXgX_tmpl)

        # Output register fragments — shape is the same for every stage (only the
        # gmem pointer differs). Allocate once so the compiler can reuse storage
        # across the stage loop rather than duplicating per-stage regs.
        gQ_tmpl = cute.local_tile(mQ, tiler_mn, (0, cluster_y))
        gK_tmpl = cute.local_tile(mK, tiler_mn, (0, cluster_y))
        gV_tmpl = cute.local_tile(mV, tiler_mn, (0, cluster_y))
        tXrQ = cute.make_fragment_like(thr_copy_X.partition_D(gQ_tmpl))
        tXrK = cute.make_fragment_like(thr_copy_X.partition_D(gK_tmpl))
        tXrV = cute.make_fragment_like(thr_copy_X.partition_D(gV_tmpl))

        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        is_even_N = cutlass.const_expr(shape_X[1] == tiler_mn[1] * self.cluster_n)

        # Predicate (shared across stages since depends only on column coord)
        cX = cute.local_tile(idX, tiler_mn, (0, cluster_y))
        tXpX = (
            utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape_X[1])
            if not is_even_N
            else None
        )
        copy = partial(copy_utils.copy, pred=tXpX, num_copy_elems=num_copy_elems_X)

        id_tile = cute.make_identity_tensor(tiler_mn)
        tXc = thr_copy_X.partition_D(cute.local_tile(id_tile, tiler_mn, (0, cluster_y)))

        head_idx = bidz
        threads_per_row = tv_layout.shape[0][0]
        lane = tidx % LANES_PER_HEAD

        # === Phase 1: issue cp.async X for all stages (software pipelining) ===
        for stage_i in cutlass.range_constexpr(ROW_STAGES):
            stage_bidx = bidx * ROW_STAGES + stage_i
            stage_row_base = stage_bidx * tiler_mn[0] + rowid_in_block
            stage_row = stage_row_base * num_qkv_heads + bidz
            gX_s = cute.local_tile(mX, tiler_mn, (stage_bidx, cluster_y))
            tXgX_s = thr_copy_X.partition_S(gX_s)
            tXsX_s = thr_copy_X.partition_D(sX_stages[stage_i])
            if stage_row < shape_X[0]:
                copy(tXgX_s, tXsX_s, is_async=True)
            cute.arch.cp_async_commit_group()
        # Release dependents so FA4 can start prologue while this kernel
        # finishes norm/RoPE/store. No-op when no PDL consumer is waiting.
        cute.arch.griddepcontrol_launch_dependents()

        # === Load QW once (same head across stages) ===
        if not delay_w_load:
            if head_idx < num_q_head:
                copy(tXgQW, tXrQW)
            elif head_idx < num_q_head + num_kv_head:
                copy(tXgKW, tXrKW)

        # === Phase 2: per-stage process (norm + RoPE + store) ===
        for stage_i in cutlass.range_constexpr(ROW_STAGES):
            stage_bidx = bidx * ROW_STAGES + stage_i
            stage_row_base = stage_bidx * tiler_mn[0] + rowid_in_block
            stage_row = stage_row_base * num_qkv_heads + bidz
            stage_token_idx = stage_row_base

            if const_expr(mG is not None):
                is_v_head = head_idx >= num_q_head + num_kv_head
                v_head_idx = head_idx - num_q_head - num_kv_head
                gates_per_round = LANES_PER_HEAD * self.num_kv_head
                gate_rounds = cute.ceil_div(self.num_q_head, gates_per_round)
                for gate_round in cutlass.range_constexpr(gate_rounds):
                    gate_idx = (
                        v_head_idx
                        + (lane + gate_round * LANES_PER_HEAD) * self.num_kv_head
                    )
                    if (
                        is_v_head
                        and gate_idx < num_q_head
                        and cluster_y == 0
                        and stage_token_idx < mG.shape[0]
                    ):
                        mG[stage_token_idx, gate_idx] = mGateInput[
                            stage_token_idx, gate_idx
                        ]

            sX_s = sX_stages[stage_i]
            tXsX_s = thr_copy_X.partition_D(sX_s)

            gQ_s = cute.local_tile(mQ, tiler_mn, (stage_bidx, cluster_y))
            gK_s = cute.local_tile(mK, tiler_mn, (stage_bidx, cluster_y))
            gV_s = cute.local_tile(mV, tiler_mn, (stage_bidx, cluster_y))
            tXgQ_s = thr_copy_X.partition_D(gQ_s)
            tXgK_s = thr_copy_X.partition_D(gK_s)
            tXgV_s = thr_copy_X.partition_D(gV_s)

            # Wait until all older stages are done (keep later stages in flight)
            cute.arch.cp_async_wait_group(ROW_STAGES - 1 - stage_i)

            cute.autovec_copy(tXsX_s, tXrX)
            y = tXrX.load().to(cute.Float32)

            if head_idx < num_q_head + num_kv_head:
                x = tXrX.load().to(cute.Float32)
                sum_sq_x = row_reduce(
                    x * x,
                    cute.ReductionOp.ADD,
                    threads_per_row,
                    reduction_buffer[None, None, 0],
                    mbar_ptr,
                    init_val=0.0,
                    hook_fn=(
                        cute.arch.cluster_wait
                        if const_expr(self.cluster_n > 1)
                        else None
                    ),
                )
                rstd = cute.math.rsqrt(sum_sq_x / shape_X[1] + eps, fastmath=True)

                if delay_w_load:
                    if head_idx < num_q_head:
                        copy(tXgQW, tXrQW)
                    elif head_idx < num_q_head + num_kv_head:
                        copy(tXgKW, tXrKW)

                if reload_from == "smem" or reload_from == "gmem":
                    if reload_from == "smem":
                        cute.autovec_copy(tXsX_s, tXrX)
                    else:
                        gX_s_reload = cute.local_tile(
                            mX, tiler_mn, (stage_bidx, cluster_y)
                        )
                        tXgX_s_reload = thr_copy_X.partition_S(gX_s_reload)
                        copy(tXgX_s_reload, tXrX)
                    x = tXrX.load().to(cute.Float32)

                x_hat = x * rstd

                y = x_hat
                if head_idx < num_q_head:
                    y *= tXrQW.load().to(cute.Float32) + norm_weight_bias
                    tXrQ.store(y.to(tXrQ.element_type))
                    copy(tXrQ, tXsX_s)
                elif head_idx < num_q_head + num_kv_head:
                    y *= tXrKW.load().to(cute.Float32) + norm_weight_bias
                    tXrK.store(y.to(tXrK.element_type))
                    copy(tXrK, tXsX_s)
                cute.arch.sync_warp()

            # Apply RoPE to Q and K heads (not V)
            if const_expr(mCos is not None and mSin is not None and mPosId is not None):
                is_q_or_k = head_idx < (num_q_head + num_kv_head)
                if is_q_or_k:
                    if lane < rotary_threads:
                        pos = (
                            cutlass.Int64(mPosId[stage_token_idx])
                            if stage_row < shape_X[0]
                            else cutlass.Int64(0)
                        )

                        mCos_s = utils.domain_offset_i64((pos, 0), mCos)
                        mSin_s = utils.domain_offset_i64((pos, 0), mSin)
                        gCos = cute.local_tile(mCos_s, (1, self.rotary_dim), (0, 0))
                        gSin = cute.local_tile(mSin_s, (1, self.rotary_dim), (0, 0))

                        copy_atom_b16 = cute.make_copy_atom(
                            cute.nvgpu.CopyUniversalOp(),
                            mCos.element_type,
                            num_bits_per_copy=64,
                        )

                        thr_layout_cs = cute.make_layout(
                            (1, rotary_threads), stride=(0, 1)
                        )
                        val_layout_cs = cute.make_layout(
                            (1, VALS_PER_THREAD), stride=(0, 1)
                        )
                        tiled_cs = cute.make_tiled_copy_tv(
                            copy_atom_b16, thr_layout_cs, val_layout_cs
                        )
                        thr_cs = tiled_cs.get_slice(lane)

                        tCgC = thr_cs.partition_S(gCos)
                        tSgS = thr_cs.partition_S(gSin)

                        cos_frag = cute.make_fragment_like(tCgC)
                        sin_frag = cute.make_fragment_like(tSgS)
                        cute.copy(copy_atom_b16, tCgC, cos_frag)
                        cute.copy(copy_atom_b16, tSgS, sin_frag)

                        # RoPE operates on this stage's own smem buffer.
                        sX0 = cute.make_tensor(
                            sX_s.iterator,
                            cute.make_layout(
                                (tiler_mn[0], self.rotary_dim), stride=(tiler_mn[1], 1)
                            ),
                        )
                        sX1 = cute.make_tensor(
                            sX_s.iterator + self.rotary_dim,
                            cute.make_layout(
                                (tiler_mn[0], self.rotary_dim), stride=(tiler_mn[1], 1)
                            ),
                        )

                        thr_layout = cute.make_layout(
                            (tiler_mn[0], rotary_threads), stride=(rotary_threads, 1)
                        )

                        tiled_copy = cute.make_tiled_copy_tv(
                            copy_atom_b16, thr_layout, val_layout_cs
                        )
                        thr_copy = tiled_copy.get_slice(
                            rowid_in_block * rotary_threads + lane
                        )

                        tXsX0 = thr_copy.partition_S(sX0)
                        tXrX0 = cute.make_fragment_like(tXsX0)
                        cute.copy(copy_atom_b16, tXsX0, tXrX0)

                        tXsX1 = thr_copy.partition_S(sX1)
                        tXrX1 = cute.make_fragment_like(tXsX1)
                        cute.copy(copy_atom_b16, tXsX1, tXrX1)

                        x0 = tXrX0.load().to(cute.Float32)
                        x1 = tXrX1.load().to(cute.Float32)
                        cos = cos_frag.load().to(cute.Float32)
                        sin = sin_frag.load().to(cute.Float32)
                        y0 = x0 * cos - x1 * sin
                        y1 = x0 * sin + x1 * cos
                        tXsX0.store(y0.to(tXsX0.element_type))
                        tXsX1.store(y1.to(tXsX1.element_type))

                    cute.arch.sync_warp()

                    if head_idx < num_q_head:
                        if stage_row < shape_X[0]:
                            copy(tXsX_s, tXgQ_s)
                    elif head_idx < num_q_head + num_kv_head:
                        if stage_row < shape_X[0]:
                            copy(tXsX_s, tXgK_s)
                        if const_expr(mKCache is not None):
                            cute.arch.sync_threads()
                            for row in cutlass.range_constexpr(tiler_mn[0]):
                                self._scatter_cache_row(
                                    mKCache,
                                    mK,
                                    mLoc,
                                    stage_bidx * tiler_mn[0] + row,
                                    mLoc.shape[0],
                                    tidx,
                                )
                else:
                    tXrV.store(y.to(tXrV.element_type))
                    if stage_row < shape_X[0]:
                        copy(tXrV, tXgV_s)
                    if const_expr(mVCache is not None):
                        cute.arch.sync_threads()
                        for row in cutlass.range_constexpr(tiler_mn[0]):
                            self._scatter_cache_row(
                                mVCache,
                                mV,
                                mLoc,
                                stage_bidx * tiler_mn[0] + row,
                                mLoc.shape[0],
                                tidx,
                            )
