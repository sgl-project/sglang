# Copyright (c) 2025, Tri Dao.

from dataclasses import dataclass
from typing import Optional

import cutlass
import cutlass.cute as cute

from sglang.srt.layers.attention.cute_ops import utils


@dataclass(frozen=True)
class AttentionMask:
    m_block_size: cutlass.Constexpr[int]
    n_block_size: cutlass.Constexpr[int]
    seqlen_q: cutlass.Int32
    seqlen_k: cutlass.Int32
    window_size_left: Optional[cutlass.Int32] = None
    window_size_right: Optional[cutlass.Int32] = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = (
        1  # only pass in if we're doing PackGQA
    )

    @cute.jit
    def apply_mask(
        self,
        acc_S: cute.Tensor,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        thr_mma: cute.TiledMma,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_causal: cutlass.Constexpr[bool],
        mask_local: cutlass.Constexpr[bool] = False,
    ) -> None:
        assert not (
            mask_causal and mask_local
        ), "mask_causal and mask_local cannot be both True"
        acc_S_mn = utils.make_acc_tensor_mn_view(acc_S)
        cS = cute.make_identity_tensor((self.m_block_size, self.n_block_size))
        tScS_mn = utils.make_acc_tensor_mn_view(thr_mma.partition_C(cS))
        # We use t0ScS as these indices are known at compile time. We then must subtract the
        # column limit by the thread column offset.
        t0ScS_mn = utils.make_acc_tensor_mn_view(thr_mma.get_slice(0).partition_C(cS))
        thr_col_offset = tScS_mn[0][1]
        seqlenk_col_limit = self.seqlen_k - n_block * self.n_block_size - thr_col_offset
        if cutlass.const_expr(not mask_causal and not mask_local):
            if cutlass.const_expr(mask_seqlen):
                # traverse column index.
                for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                    # if t0ScS_mn[0, c][1] >= seqlenk_col_limit:
                    #     acc_S_mn[None, c].fill(-cutlass.Float32.inf)
                    oob = t0ScS_mn[0, c][1] >= seqlenk_col_limit
                    for r in cutlass.range(
                        cute.size(tScS_mn.shape[0]), unroll_full=True
                    ):
                        acc_S_mn[r, c] = -cutlass.Float32.inf if oob else acc_S_mn[r, c]
        else:  # Causal or local
            # If PackGQA, we split the work of compute divmod among threads in the same row
            threads_per_row = thr_mma.tv_layout_C.shape[0][0]
            if cutlass.const_expr(self.qhead_per_kvhead_packgqa != 1):
                assert (
                    cute.arch.WARP_SIZE % threads_per_row == 0
                ), "threads_per_row must divide WARP_SIZE"
                assert cute.size(acc_S_mn.shape[0]) <= threads_per_row
                tidx = thr_mma.thr_idx
                mma_m_idx = (
                    m_block * self.m_block_size + tScS_mn[tidx % threads_per_row, 0][0]
                ) // self.qhead_per_kvhead_packgqa
            causal_row_offset = (
                1
                + self.seqlen_k
                - n_block * self.n_block_size
                - self.seqlen_q
                - thr_col_offset
            )
            if cutlass.const_expr(mask_causal):
                for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                    # get the column index limit based on current row. Only consider the row index, so the column index sets to 0.
                    if cutlass.const_expr(self.qhead_per_kvhead_packgqa == 1):
                        row_idx = tScS_mn[r, 0][0] + m_block * self.m_block_size
                    else:
                        row_idx = utils.shuffle_sync(
                            mma_m_idx, r % threads_per_row, width=threads_per_row
                        )
                    col_limit_right = row_idx + causal_row_offset
                    if cutlass.const_expr(mask_seqlen):
                        col_limit_right = cutlass.min(
                            col_limit_right, seqlenk_col_limit
                        )
                    # traverse column index.
                    for c in cutlass.range(
                        cute.size(tScS_mn.shape[1]), unroll_full=True
                    ):
                        # only consider the column index, so the row index sets to 0.
                        # if t0ScS_mn[0, c][1] >= col_limit_right:
                        # acc_S_mn[r, c] = -cutlass.Float32.inf
                        acc_S_mn[r, c] = (
                            -cutlass.Float32.inf
                            if t0ScS_mn[0, c][1] >= col_limit_right
                            else acc_S_mn[r, c]
                        )
            else:  # Local
                local_row_offset_right = (
                    causal_row_offset + self.window_size_right
                    if cutlass.const_expr(self.window_size_right is not None)
                    else None
                )
                local_row_offset_left = (
                    causal_row_offset - 1 - self.window_size_left
                    if cutlass.const_expr(self.window_size_left is not None)
                    else None
                )
                for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                    if cutlass.const_expr(self.qhead_per_kvhead_packgqa == 1):
                        row_idx = tScS_mn[r, 0][0] + m_block * self.m_block_size
                    else:
                        row_idx = utils.shuffle_sync(
                            mma_m_idx, r % threads_per_row, width=threads_per_row
                        )
                    if cutlass.const_expr(self.window_size_right is not None):
                        col_limit_right = row_idx + local_row_offset_right
                        if cutlass.const_expr(mask_seqlen):
                            col_limit_right = cutlass.min(
                                col_limit_right, seqlenk_col_limit
                            )
                    else:
                        col_limit_right = self.n_block_size
                    col_limit_left = (
                        row_idx + local_row_offset_left
                        if cutlass.const_expr(self.window_size_left is not None)
                        else 0
                    )
                    # if cute.arch.thread_idx()[0] == 128: cute.printf("n_block = {}, r = {}, row_idx = {}, causal_row_offset = {}, col_limit_right = {}, col_limit_left = {}", n_block, r, row_idx, causal_row_offset, col_limit_right, col_limit_left)
                    # traverse column index.
                    for c in cutlass.range(
                        cute.size(tScS_mn.shape[1]), unroll_full=True
                    ):
                        col_idx = t0ScS_mn[0, c][1]
                        # only consider the column index, so the row index sets to 0.
                        if col_idx >= col_limit_right or col_idx < col_limit_left:
                            acc_S_mn[r, c] = -cutlass.Float32.inf

    @cute.jit
    def apply_mask_sm100(
        self,
        acc_S: cute.Tensor,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        thr_mma: cute.TiledMma,
        thr_tmem_load: cute.TiledCopy,
        mask_seqlen: cutlass.Constexpr,
        mask_causal: cutlass.Constexpr,
        mask_local: cutlass.Constexpr,
    ) -> None:
        assert not (
            mask_causal and mask_local
        ), "mask_causal and mask_local cannot be both True"
        cS = cute.make_identity_tensor((self.m_block_size, self.n_block_size))
        tScS = thr_mma.partition_C(cS)
        tScS_t2r = thr_tmem_load.partition_D(tScS)
        seqlenk_col_limit = self.seqlen_k - n_block * self.n_block_size
        if cutlass.const_expr(not mask_causal and not mask_local):
            if cutlass.const_expr(mask_seqlen):
                ncol = cutlass.const_expr(cute.size(tScS_t2r.shape))
                if cutlass.const_expr(not ncol % 16 == 0):
                    for i in cutlass.range(ncol, unroll_full=True):
                        # if tScS_t2r[i][1] >= seqlenk_col_limit:
                        #     acc_S[i] = -cutlass.Float32.inf
                        # For some reason the 2 lines above generate really bad SASS
                        acc_S[i] = (
                            -cutlass.Float32.inf
                            if tScS_t2r[i][1] >= seqlenk_col_limit
                            else acc_S[i]
                        )
                else:
                    # Bit manipulation, compiles down to the R2P instruction
                    # We know that tScS_t2r[i][1] == i, for the particular tmem copy atom we're using
                    # Ideally we'd move by 32 instead of 16, but mask >> i isn't correct for i == 31
                    # (see below).
                    for s in cutlass.range(ncol // 16, unroll_full=True):
                        col_limit_right_s = seqlenk_col_limit - s * 16
                        # Don't need to clamp to 32 since the shr.u32 instruction does that already
                        col_limit_right_cur = cutlass.Uint32(max(col_limit_right_s, 0))
                        # 0 -> 0b00...00, 1 -> 0b00...01, ..., 31 -> 0b01...11, 32 -> 0b11...11
                        mask = cutlass.Uint32((1 << col_limit_right_cur) - 1)
                        # if tidx == 0: cute.printf("mask = 0x%x, col_limit_right_s = %d, col_limit_right_cur = %d", mask, col_limit_right_s, col_limit_right_cur)
                        # This needs to be range_constexpr, otherwise the compiler can't generate
                        # the R2P instruction
                        for i in cutlass.range_constexpr(16):
                            # mask >> i does not produce correct result for 0b11..11 >> 31
                            # However, if we use utils.shr_u32, the compiler doesn't generate
                            # the R2P instruction, so it's slower.
                            # Instead we just move by 16 instead of 32.
                            mask_i_bit = cutlass.Boolean(mask & (1 << i))
                            # mask_i_bit = cutlass.Boolean(utils.shr_u32(mask, i) & 1)
                            # if tidx == 0: cute.printf("mask_i_bit = %d, after shift = 0x%x, i = %d, s = %d", mask_i_bit, utils.shr_u32(mask, i), i, s)
                            acc_S[s * 16 + i] = (
                                acc_S[s * 16 + i]
                                if mask_i_bit
                                else -cutlass.Float32.inf
                            )
                            # This is the equivalent of:
                            # acc_S[s * 16 + i] = acc_S[s * 16 + i] if col_limit_right_s <= i else -cutlass.Float32.inf
                    # if tidx == 0: cute.print_tensor(acc_S)
        else:  # Causal or local
            causal_row_offset = (
                1 + self.seqlen_k - n_block * self.n_block_size - self.seqlen_q
            )
            row_idx = tScS_t2r[0][0] + m_block * self.m_block_size
            if cutlass.const_expr(self.qhead_per_kvhead_packgqa != 1):
                row_idx = row_idx // self.qhead_per_kvhead_packgqa
            if cutlass.const_expr(mask_causal):
                col_limit_right = row_idx + causal_row_offset
                if cutlass.const_expr(mask_seqlen):
                    col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
                # if cute.arch.thread_idx()[0] % 32 == 0:
                #     cute.printf("tidx = %d, tidx tmem = %d, row_idx = %d, col_limit_right = %d, causal_row_offset = %d\n", cute.arch.thread_idx()[0], thr_tmem_load.thr_idx, row_idx, col_limit_right, causal_row_offset)
                ncol = cutlass.const_expr(cute.size(tScS_t2r.shape))
                if cutlass.const_expr(not ncol % 16 == 0):
                    for i in cutlass.range(ncol, unroll_full=True):
                        acc_S[i] = (
                            -cutlass.Float32.inf
                            if tScS_t2r[i][1] >= col_limit_right
                            else acc_S[i]
                        )
                else:
                    # Bit manipulation, compiles down to the R2P instruction
                    # We know that tScS_t2r[i][1] == i, for the particular tmem copy atom we're using
                    for s in cutlass.range(ncol // 16, unroll_full=True):
                        col_limit_right_s = col_limit_right - s * 16
                        col_limit_right_cur = cutlass.Uint32(max(col_limit_right_s, 0))
                        # 0 -> 0b00...00, 1 -> 0b00...01, ..., 31 -> 0b01...11, 32 -> 0b11...11
                        mask = cutlass.Uint32((1 << col_limit_right_cur) - 1)
                        # This needs to be range_constexpr, otherwise the compiler can't generate
                        # the R2P instruction
                        for i in cutlass.range_constexpr(16):
                            # mask_i_bit = cutlass.Boolean(utils.shr_u32(mask, i) & 1)
                            mask_i_bit = cutlass.Boolean(mask & (1 << i))
                            acc_S[s * 16 + i] = (
                                acc_S[s * 16 + i]
                                if mask_i_bit
                                else -cutlass.Float32.inf
                            )
                            # This is the equivalent of:
                            # acc_S[s * 16 + i] = acc_S[s * 16 + i] if col_limit_right_s <= i else -cutlass.Float32.inf
            else:
                local_row_offset_right = (
                    causal_row_offset + self.window_size_right
                    if cutlass.const_expr(self.window_size_right is not None)
                    else None
                )
                local_row_offset_left = (
                    causal_row_offset - 1 - self.window_size_left
                    if cutlass.const_expr(self.window_size_left is not None)
                    else None
                )
                if cutlass.const_expr(self.window_size_right is not None):
                    col_limit_right = row_idx + local_row_offset_right
                    if cutlass.const_expr(mask_seqlen):
                        col_limit_right = cutlass.min(
                            col_limit_right, seqlenk_col_limit
                        )
                else:
                    col_limit_right = self.n_block_size
                col_limit_left = (
                    row_idx + local_row_offset_left
                    if cutlass.const_expr(self.window_size_left is not None)
                    else 0
                )
                # if cute.arch.thread_idx()[0] == 0 or cute.arch.thread_idx()[0] == 128: cute.printf("m_block = {}, n_block = {}, row_idx = {}, causal_row_offset = {}, col_limit_right = {}, col_limit_left = {}", m_block, n_block, row_idx, causal_row_offset, col_limit_right, col_limit_left)
                for i in cutlass.range(cute.size(tScS_t2r.shape), unroll_full=True):
                    col_idx = tScS_t2r[i][1]
                    acc_S[i] = (
                        -cutlass.Float32.inf
                        if col_idx >= col_limit_right or col_idx < col_limit_left
                        else acc_S[i]
                    )
