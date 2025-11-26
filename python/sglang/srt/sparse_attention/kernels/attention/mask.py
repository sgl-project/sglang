# Copyright (c) 2025, Tri Dao.

from dataclasses import dataclass
from typing import Optional

import cutlass
import cutlass.cute as cute

import sglang.srt.sparse_attention.kernels.attention.utils as utils


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

    # Streaming mask
    sink_size: Optional[cutlass.Int32] = None
    enable_streaming: cutlass.Constexpr[bool] = False

    # Chunked Attention
    position_ids: Optional[cute.Tensor] = None

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
                if cutlass.const_expr(False):
                    # traverse column index.
                    for c in cutlass.range(
                        cute.size(tScS_mn.shape[1]), unroll_full=True
                    ):
                        oob = t0ScS_mn[0, c][1] >= seqlenk_col_limit
                        for r in cutlass.range(
                            cute.size(tScS_mn.shape[0]), unroll_full=True
                        ):
                            acc_S_mn[r, c] = (
                                -cutlass.Float32.inf if oob else acc_S_mn[r, c]
                            )
                else:  # R2P trick, see apply_mask_sm100
                    # Instead of comparing limit to 0, 1, 8, 9, 16, 17, ...,
                    # we compare a transformed version of limit to 0, 1, 2, 3, 4, 5, ...
                    # This is so that we can use the R2P instruction.
                    col_limit_transformed = seqlenk_col_limit // 8 * 2 + min(
                        seqlenk_col_limit % 8, 2
                    )
                    ncol = cutlass.const_expr(cute.size(tScS_mn.shape[1]))
                    for s in cutlass.range_constexpr(cute.ceil_div(ncol, 24)):
                        col_limit_right_s = max(col_limit_transformed - s * 24, 0)
                        mask = (1 << col_limit_right_s) - 1
                        for i in cutlass.range_constexpr(min(24, ncol - s * 24)):
                            in_bound = cutlass.Boolean(mask & (1 << i))
                            c = s * 24 + i
                            for r in cutlass.range(
                                cute.size(tScS_mn.shape[0]), unroll_full=True
                            ):
                                acc_S_mn[r, c] = (
                                    acc_S_mn[r, c] if in_bound else -cutlass.Float32.inf
                                )
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
                    if cutlass.const_expr(False):
                        # traverse column index.
                        for c in cutlass.range(
                            cute.size(tScS_mn.shape[1]), unroll_full=True
                        ):
                            acc_S_mn[r, c] = (
                                -cutlass.Float32.inf
                                if t0ScS_mn[0, c][1] >= col_limit_right
                                else acc_S_mn[r, c]
                            )
                    else:  # R2P trick, see apply_mask_sm100
                        col_limit_transformed = col_limit_right // 8 * 2 + min(
                            col_limit_right % 8, 2
                        )
                        ncol = cutlass.const_expr(cute.size(tScS_mn.shape[1]))
                        for s in cutlass.range_constexpr(cute.ceil_div(ncol, 24)):
                            col_limit_right_s = max(col_limit_transformed - s * 24, 0)
                            mask = (1 << col_limit_right_s) - 1
                            for i in cutlass.range_constexpr(min(24, ncol - s * 24)):
                                in_bound = cutlass.Boolean(mask & (1 << i))
                                c = s * 24 + i
                                acc_S_mn[r, c] = (
                                    acc_S_mn[r, c] if in_bound else -cutlass.Float32.inf
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

                        # TODO(KuangjuX): Compilation error.
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
                if cutlass.const_expr(False):
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
                    # Ideally we'd move by 32 instead of 24, but mask >> i isn't correct for i == 31
                    # (see below).
                    for s in cutlass.range_constexpr(cute.ceil_div(ncol, 24)):
                        # Don't need to clamp to 32 since the shr.u32 instruction does that already
                        col_limit_right_s = max(seqlenk_col_limit - s * 24, 0)
                        # 0 -> 0b00...00, 1 -> 0b00...01, ..., 31 -> 0b01...11, 32 -> 0b11...11
                        mask = (1 << col_limit_right_s) - 1
                        # if tidx == 0: cute.printf("mask = 0x%x, col_limit_right_s = %d, col_limit_right_s = %d", mask, col_limit_right_s, col_limit_right_s)
                        # This needs to be range_constexpr, otherwise the compiler can't generate
                        # the R2P instruction
                        for i in cutlass.range_constexpr(min(24, ncol - s * 24)):
                            # mask >> i does not produce correct result for 0b11..11 >> 31
                            # However, if we use utils.shr_u32, the compiler doesn't generate
                            # the R2P instruction, so it's slower.
                            # Instead we just move by 24 instead of 32.
                            # if tidx == 0: cute.printf("mask_i_bit = %d, after shift = 0x%x, i = %d, s = %d", mask_i_bit, utils.shr_u32(mask, i), i, s)
                            acc_S[s * 24 + i] = (
                                acc_S[s * 24 + i]
                                if cutlass.Boolean(mask & (1 << i))
                                else -cutlass.Float32.inf
                            )
                            # This is the equivalent of:
                            # acc_S[s * 24 + i] = acc_S[s * 24 + i] if col_limit_right_s <= i else -cutlass.Float32.inf
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
                if cutlass.const_expr(False):
                    for i in cutlass.range(ncol, unroll_full=True):
                        acc_S[i] = (
                            -cutlass.Float32.inf
                            if tScS_t2r[i][1] >= col_limit_right
                            else acc_S[i]
                        )
                else:
                    # Bit manipulation, compiles down to the R2P instruction
                    # We know that tScS_t2r[i][1] == i, for the particular tmem copy atom we're using
                    for s in cutlass.range_constexpr(cute.ceil_div(ncol, 24)):
                        col_limit_right_s = max(col_limit_right - s * 24, 0)
                        # 0 -> 0b00...00, 1 -> 0b00...01, ..., 31 -> 0b01...11, 32 -> 0b11...11
                        mask = (1 << col_limit_right_s) - 1
                        # This needs to be range_constexpr, otherwise the compiler can't generate
                        # the R2P instruction
                        for i in cutlass.range_constexpr(min(24, ncol - s * 24)):
                            acc_S[s * 24 + i] = (
                                acc_S[s * 24 + i]
                                if cutlass.Boolean(mask & (1 << i))
                                else -cutlass.Float32.inf
                            )
                            # This is the equivalent of:
                            # acc_S[s * 24 + i] = acc_S[s * 24 + i] if col_limit_right_s <= i else -cutlass.Float32.inf
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

    # @cute.jit
    # def apply_streaming_mask(
    #     self,
    #     acc_S: cute.Tensor,
    #     m_block: cutlass.Int32,
    #     n_block: cutlass.Int32,
    #     thr_mma: cute.TiledMma,
    #     mask_seqlen: cutlass.Constexpr
    # ):
    #     acc_S_mn = utils.make_acc_tensor_mn_view(acc_S)
    #     cS = cute.make_identity_tensor((self.m_block_size, self.n_block_size))
    #     tScS_mn = utils.make_acc_tensor_mn_view(thr_mma.partition_C(cS))
    #     t0ScS_mn = utils.make_acc_tensor_mn_view(thr_mma.get_slice(0).partition_C(cS))

    #     # This offset calculation was correct in the original code. It correctly
    #     # brings global coordinates into a local context for comparison.
    #     thr_col_offset = tScS_mn[0][1]
    #     global_col_offset = n_block * self.n_block_size + thr_col_offset

    #     causal_row_offset = (
    #         self.seqlen_k - self.seqlen_q - global_col_offset
    #     )

    #     # The rest of the offset calculations remain the same as the original.
    #     sink_col_limit = self.sink_size - global_col_offset
    #     local_row_offset_right = causal_row_offset

    #     if cutlass.const_expr(self.window_size_left is not None):
    #         local_row_offset_left = -self.window_size_left - global_col_offset
    #     else:
    #         local_row_offset_left = -self.seqlen_k

    #     seqlenk_col_limit = self.seqlen_k - global_col_offset

    #     for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
    #         # if cutlass.const_expr(self.qhead_per_kvhead_packgqa == 1):
    #         #     row_idx = tScS_mn[r, 0][0] + m_block * self.m_block_size
    #         # else:
    #         #     # TODO(KuangjuX): Packed GQA Support
    #         #     # Throw runtime error
    #         #     raise RuntimeError("Packed GQA is not supported currently.")

    #         local_row_idx_in_block = tScS_mn[r, 0][0]
    #         if cutlass.const_expr(self.position_ids is not None):
    #             row_idx = self.position_ids[local_row_idx_in_block]
    #         else:
    #             if cutlass.const_expr(self.qhead_per_kvhead_packgqa == 1):
    #                 row_idx = tScS_mn[r, 0][0] + m_block * self.m_block_size
    #             else:
    #                 # TODO(KuangjuX): Packed GQA Support
    #                 # Packed GQA is not supported currently - set row_idx to 0 as fallback
    #                 row_idx = 0

    #         col_limit_right = row_idx + local_row_offset_right
    #         col_limit_left = row_idx + local_row_offset_left

    #         if cutlass.const_expr(mask_seqlen):
    #             # Note: The padding check should still use strict inequality, as sequence length
    #             # is exclusive (e.g., for seqlen=128, valid indices are 0-127).
    #             # We apply min to the causal limit, which is fine.
    #             col_limit_right_with_padding = seqlenk_col_limit - 1
    #             col_limit_right = cutlass.min(col_limit_right, col_limit_right_with_padding)

    #         for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
    #             col_idx = t0ScS_mn[r, c][1]

    #             should_mask = True

    #             is_in_sink = col_idx < sink_col_limit
    #             is_causal = col_idx <= col_limit_right
    #             is_in_window = col_idx >= col_limit_left

    #             if is_causal:
    #                 if is_in_sink or is_in_window:
    #                     should_mask = False

    #             if should_mask:
    #                 acc_S_mn[r, c] = -cutlass.Float32.inf

    @cute.jit
    def apply_streaming_mask(
        self,
        acc_S: cute.Tensor,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        thr_mma: cute.TiledMma,
        mask_seqlen: cutlass.Constexpr,
    ):
        """
        A simplified, direct-logic version of the masking function for debugging.
        It calculates absolute q_pos and k_pos for each element and compares them directly,
        bypassing the complex offset calculations.
        """
        acc_S_mn = utils.make_acc_tensor_mn_view(acc_S)
        cS = cute.make_identity_tensor((self.m_block_size, self.n_block_size))
        # tScS_mn gives the coordinates (row, col) within the entire MxN block
        tScS_mn = thr_mma.partition_C(cS)

        # Iterate through each element this thread is responsible for in the accumulator
        for i in cutlass.range(cute.size(acc_S)):
            # Get the (row, col) coordinates of this element within the current block
            # tScS_mn is a "Tensor of Coordinates"
            coord_in_block = tScS_mn[i]
            q_coord_in_block = coord_in_block[0]
            k_coord_in_block = coord_in_block[1]

            # Step 1: Determine the absolute query position (q_pos)
            if cutlass.const_expr(self.position_ids is not None):
                # For chunked prefill, look up the true absolute position
                q_pos = self.position_ids[q_coord_in_block]
            else:
                # For regular attention, calculate it from the block index
                q_pos = m_block * self.m_block_size + q_coord_in_block

            # Step 2: Determine the absolute key position (k_pos)
            k_pos = n_block * self.n_block_size + k_coord_in_block

            # Step 3: Apply masking rules using direct, absolute positions
            should_mask = True

            # Rule A: Is the key within the sink?
            is_in_sink = k_pos < self.sink_size

            # Rule B: Is the key within the sliding window?
            is_in_window = True  # Default to true if no window size is specified
            if cutlass.const_expr(self.window_size_left is not None):
                is_in_window = k_pos >= q_pos - self.window_size_left

            # Rule C: Is the connection causal?
            is_causal = k_pos <= q_pos

            # An element is NOT masked if it's causal AND (it's in the sink OR it's in the window)
            if is_causal:
                if is_in_sink or is_in_window:
                    should_mask = False

            # Rule D: Padding mask (optional, based on sequence lengths)
            if cutlass.const_expr(mask_seqlen):
                # Note: self.seqlen_q is the length of the current Q chunk, not total length
                # The q_pos check is implicitly handled by the launch grid.
                # We only need to check k_pos against the total key length.
                if k_pos >= self.seqlen_k:
                    should_mask = True

            # Step 4: Apply the mask if needed
            if should_mask:
                acc_S[i] = -cutlass.Float32.inf
