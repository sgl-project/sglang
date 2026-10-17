# Copyright (c) 2025, Tri Dao.

import math
import operator
from dataclasses import dataclass
from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32

import sglang.kernels.ops.attention.step5.cute.sm100_union.flash_attn.utils as utils
from sglang.kernels.ops.attention.step5.cute.sm100_union.flash_attn.cute_dsl_utils import (
    ParamsBase,
)


@dataclass
class Softmax(ParamsBase):
    scale_log2: Float32
    num_rows: cutlass.Constexpr[int]
    row_max: cute.Tensor
    row_sum: cute.Tensor
    arch: cutlass.Constexpr[int] = 80
    softmax_scale: Float32 | None = None

    @staticmethod
    def create(
        scale_log2: Float32,
        num_rows: cutlass.Constexpr[int],
        arch: cutlass.Constexpr[int] = 80,
        softmax_scale: Float32 | None = None,
    ):
        row_max = cute.make_rmem_tensor(num_rows, Float32)
        row_sum = cute.make_rmem_tensor(num_rows, Float32)
        return Softmax(scale_log2, num_rows, row_max, row_sum, arch, softmax_scale)

    def reset(self) -> None:
        self.row_max.fill(-Float32.inf)
        self.row_sum.fill(0.0)

    def _compute_row_max(
        self, acc_S_row: cute.TensorSSA, init_val: float | Float32 | None = None
    ) -> Float32:
        return utils.fmax_reduce(acc_S_row, init_val, arch=self.arch)

    def _compute_row_sum(
        self, acc_S_row_exp: cute.TensorSSA, init_val: float | Float32 | None = None
    ) -> Float32:
        return utils.fadd_reduce(acc_S_row_exp, init_val, arch=self.arch)

    @cute.jit
    def online_softmax(
        self,
        acc_S: cute.Tensor,
        is_first: cutlass.Constexpr[bool] = False,
        check_inf: cutlass.Constexpr[bool] = True,
    ) -> cute.Tensor:
        """Apply online softmax and return the row_scale to rescale O.

        :param acc_S: acc_S tensor
        :type acc_S: cute.Tensor
        :param is_first: is first n_block
        :type is_first: cutlass.Constexpr
        """
        # Change acc_S to M,N layout view.
        acc_S_mn = utils.make_acc_tensor_mn_view(acc_S)
        row_scale = cute.make_fragment_like(self.row_max, Float32)

        row_max = self.row_max
        row_sum = self.row_sum
        scale_log2 = self.scale_log2
        arch = self.arch

        # Each iteration processes one row of acc_S
        for r in cutlass.range(cute.size(row_max), unroll_full=True):
            acc_S_row = acc_S_mn[r, None].load()  # (n_block_size)

            row_max_cur = utils.fmax_reduce(
                acc_S_row,
                init_val=row_max[r] if cutlass.const_expr(not is_first) else None,
                arch=arch,
            )

            row_max_cur = utils.warp_reduce(row_max_cur, cute.arch.fmax, width=4)
            if cutlass.const_expr(check_inf):
                row_max_cur = 0.0 if row_max_cur == -Float32.inf else row_max_cur

            if cutlass.const_expr(is_first):
                row_max_cur_scaled = row_max_cur * scale_log2
                acc_S_row_exp = utils.exp2f(acc_S_row * scale_log2 - row_max_cur_scaled)

                acc_S_row_sum = utils.fadd_reduce(
                    acc_S_row_exp, init_val=None, arch=arch
                )
                row_scale[r] = 1.0
            else:
                row_max_prev = row_max[r]
                row_max_cur_scaled = row_max_cur * scale_log2
                acc_S_row_exp = utils.exp2f(acc_S_row * scale_log2 - row_max_cur_scaled)
                # row_scale[r] = utils.exp2f(row_max_prev * self.scale_log2 - row_max_cur_scaled)
                row_scale[r] = utils.exp2f((row_max_prev - row_max_cur) * scale_log2)

                acc_S_row_sum = utils.fadd_reduce(
                    acc_S_row_exp, init_val=row_sum[r] * row_scale[r], arch=arch
                )

            row_max[r] = row_max_cur
            row_sum[r] = acc_S_row_sum
            acc_S_mn[r, None].store(acc_S_row_exp)

        return row_scale

    @cute.jit
    def finalize(
        self, final_scale: Float32 = 1.0, sink_val: Float32 | cute.Tensor | None = None
    ) -> cute.Tensor:
        """Finalize the online softmax by computing the scale and logsumexp."""
        if cutlass.const_expr(
            sink_val is not None and isinstance(sink_val, cute.Tensor)
        ):
            assert cute.size(sink_val) == cute.size(self.row_sum)
        row_sum = self.row_sum
        row_max = self.row_max
        scale_log2 = self.scale_log2

        # quad reduction for row_sum as we didn't do it during each iteration of online softmax
        row_sum.store(utils.warp_reduce(row_sum.load(), operator.add, width=4))
        row_scale = cute.make_fragment_like(row_max, Float32)

        for r in cutlass.range(cute.size(row_sum), unroll_full=True):
            if cutlass.const_expr(sink_val is not None):
                sink_val_cur = (
                    sink_val if not isinstance(sink_val, cute.Tensor) else sink_val[r]
                )
                LOG2_E = math.log2(math.e)
                row_sum[r] += utils.exp2f(
                    sink_val_cur * LOG2_E - row_max[r] * scale_log2
                )

            # if row_sum is zero or nan, set acc_O_mn_row to 1.0
            acc_O_mn_row_is_zero_or_nan = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]
            row_scale[r] = (
                cute.arch.rcp_approx(
                    row_sum[r] if not acc_O_mn_row_is_zero_or_nan else 1.0
                )
            ) * final_scale
            row_sum_cur = row_sum[r]
            LN2 = math.log(2.0)
            row_sum[r] = (
                (row_max[r] * scale_log2 + utils.log2f(row_sum_cur)) * LN2
                if not acc_O_mn_row_is_zero_or_nan
                else -Float32.inf
            )
        return row_scale

    @cute.jit
    def rescale_O(self, acc_O: cute.Tensor, row_scale: cute.Tensor) -> None:
        """Scale each row of acc_O by the given scale tensor.
        :param acc_O: input tensor
        :type acc_O: cute.Tensor
        :param row_scale: row_scale tensor
        :type row_scale: cute.Tensor
        """
        acc_O_mn = utils.make_acc_tensor_mn_view(acc_O)
        assert cute.size(row_scale) == cute.size(acc_O_mn, mode=[0])
        for r in cutlass.range(cute.size(row_scale), unroll_full=True):
            acc_O_mn[r, None].store(acc_O_mn[r, None].load() * row_scale[r])


@dataclass
class SoftmaxSm100(Softmax):
    rescale_threshold: cutlass.Constexpr[float] = 0.0

    @staticmethod
    def create(
        scale_log2: Float32,
        rescale_threshold: cutlass.Constexpr[float] = 0.0,
        softmax_scale: Float32 | None = None,
    ):
        num_rows = 1
        arch = 100
        row_max = cute.make_rmem_tensor(num_rows, Float32)
        row_sum = cute.make_rmem_tensor(num_rows, Float32)
        return SoftmaxSm100(
            scale_log2,
            num_rows,
            row_max,
            row_sum,
            arch,
            softmax_scale,
            rescale_threshold=rescale_threshold,
        )

    @cute.jit
    def compute_row_max_local(
        self, acc_S_row: cute.TensorSSA, is_first: Boolean
    ) -> Float32:
        if cutlass.const_expr(is_first):
            row_max_new = self._compute_row_max(acc_S_row)
        else:
            row_max_old = self.row_max[0]
            row_max_new = self._compute_row_max(acc_S_row, init_val=row_max_old)
        return row_max_new

    @cute.jit
    def update_row_max_from_local(
        self,
        row_max_new: Float32,
        is_first: Boolean,
    ) -> Tuple[Float32, Float32]:
        if cutlass.const_expr(is_first):
            row_max_safe = row_max_new if row_max_new != -cutlass.Float32.inf else 0.0
            acc_scale = 0.0
        else:
            row_max_old = self.row_max[0]
            row_max_safe = row_max_new if row_max_new != -cutlass.Float32.inf else 0.0
            acc_scale_ = (row_max_old - row_max_safe) * self.scale_log2
            acc_scale = utils.exp2f(acc_scale_)
            if cutlass.const_expr(self.rescale_threshold > 0.0):
                if acc_scale_ >= -self.rescale_threshold:
                    row_max_new = row_max_old
                    row_max_safe = row_max_old
                    acc_scale = 1.0
        self.row_max[0] = row_max_new
        return row_max_safe, acc_scale

    @cute.jit
    def update_row_max(
        self, acc_S_row: cute.TensorSSA, is_first: int
    ) -> Tuple[Float32, Float32]:
        if cutlass.const_expr(is_first):
            row_max_new = self._compute_row_max(acc_S_row)
            row_max_safe = row_max_new if row_max_new != -cutlass.Float32.inf else 0.0
            acc_scale = 0.0
        else:
            row_max_old = self.row_max[0]
            row_max_new = self._compute_row_max(acc_S_row, init_val=row_max_old)
            row_max_safe = row_max_new if row_max_new != -cutlass.Float32.inf else 0.0
            acc_scale_ = (row_max_old - row_max_safe) * self.scale_log2
            acc_scale = utils.exp2f(acc_scale_)
            if cutlass.const_expr(self.rescale_threshold > 0.0):
                if acc_scale_ >= -self.rescale_threshold:
                    row_max_new = row_max_old
                    row_max_safe = row_max_old
                    acc_scale = 1.0
        self.row_max[0] = row_max_new
        return row_max_safe, acc_scale

    def update_row_sum(
        self, acc_S_row_exp: cute.TensorSSA, row_scale: Float32, is_first: int = False
    ) -> None:
        init_val = (
            self.row_sum[0] * row_scale if cutlass.const_expr(not is_first) else None
        )
        self.row_sum[0] = self._compute_row_sum(acc_S_row_exp, init_val=init_val)

    def update_paired_row_sum(
        self,
        acc_S_row_exp: cute.TensorSSA,
        row_scale: Float32,
        is_first: int = False,
    ) -> None:
        init_val = (
            self.row_sum[0] * row_scale if cutlass.const_expr(not is_first) else None
        )
        self.row_sum[0] = self._compute_paired_row_sum(acc_S_row_exp, init_val)

    @cute.jit
    def _compute_paired_row_sum(
        self,
        acc_S_row_exp: cute.TensorSSA,
        init_val: Float32 | None,
    ) -> Float32:
        fragment_elements = cutlass.const_expr(cute.size(acc_S_row_exp.shape))
        assert fragment_elements in (32, 64)
        values = cute.make_fragment(acc_S_row_exp.shape, Float32)
        values.store(acc_S_row_exp)

        lane_is_lower = cute.arch.lane_idx() < 16
        accum = [(Float32.zero, Float32.zero)] * 4
        for segment in cutlass.range_constexpr(fragment_elements // 8):
            owner_is_lower = cutlass.const_expr(segment % 2 == 0)
            if cutlass.const_expr(segment > 0):
                for group in cutlass.range_constexpr(4):
                    peer = (
                        cute.arch.shuffle_sync_bfly(accum[group][0], offset=16),
                        cute.arch.shuffle_sync_bfly(accum[group][1], offset=16),
                    )
                    if lane_is_lower == owner_is_lower:
                        accum[group] = peer

            if lane_is_lower == owner_is_lower:
                base = cutlass.const_expr((segment // 2) * 16)
                if cutlass.const_expr(segment == 0):
                    accum[0] = (
                        utils.add_packed_f32x2(
                            (init_val, 0.0),
                            (values[base], values[base + 1]),
                        )
                        if cutlass.const_expr(init_val is not None)
                        else (values[base], values[base + 1])
                    )
                    for group in cutlass.range_constexpr(1, 4):
                        offset = cutlass.const_expr(base + group * 2)
                        accum[group] = (values[offset], values[offset + 1])
                else:
                    for group in cutlass.range_constexpr(4):
                        offset = cutlass.const_expr(base + group * 2)
                        accum[group] = utils.add_packed_f32x2(
                            accum[group],
                            (values[offset], values[offset + 1]),
                        )

                for group in cutlass.range_constexpr(4):
                    offset = cutlass.const_expr(base + 8 + group * 2)
                    accum[group] = utils.add_packed_f32x2(
                        accum[group],
                        (values[offset], values[offset + 1]),
                    )

        accum[0] = utils.add_packed_f32x2(accum[0], accum[1])
        accum[2] = utils.add_packed_f32x2(accum[2], accum[3])
        accum[0] = utils.add_packed_f32x2(accum[0], accum[2])
        row_sum = accum[0][0] + accum[0][1]
        peer_row_sum = cute.arch.shuffle_sync_bfly(
            row_sum,
            offset=16,
        )
        return peer_row_sum if lane_is_lower else row_sum

    @cute.jit
    def scale_subtract_rowmax(
        self,
        acc_S_row: cute.Tensor,
        row_max: Float32,
    ):
        assert cute.size(acc_S_row.shape) % 2 == 0, (
            "acc_S_row must have an even number of elements"
        )
        row_max_scaled = row_max * self.scale_log2
        for i in cutlass.range(0, cute.size(acc_S_row.shape), 2, unroll_full=True):
            acc_S_row[i], acc_S_row[i + 1] = utils.fma_packed_f32x2(
                (acc_S_row[i], acc_S_row[i + 1]),
                (self.scale_log2, self.scale_log2),
                (-row_max_scaled, -row_max_scaled),
            )

    @cute.jit
    def apply_exp2_convert(
        self,
        acc_S_row: cute.Tensor,
        acc_S_row_converted: cute.Tensor,
        e2e: cutlass.Constexpr[bool] = False,
        e2e_freq: cutlass.Constexpr[int] = 16,
        e2e_res: cutlass.Constexpr[int] = 4,
        e2e_frg_limit: cutlass.Constexpr[int] = 1,
    ):
        assert cute.size(acc_S_row.shape) % 2 == 0, (
            "acc_S_row must have an even number of elements"
        )
        frg_tile = 32
        assert frg_tile % 2 == 0
        frg_cnt = cute.size(acc_S_row) // frg_tile
        assert cute.size(acc_S_row) % frg_tile == 0
        acc_S_row_frg = cute.logical_divide(acc_S_row, cute.make_layout(frg_tile))
        acc_S_row_converted_frg = cute.logical_divide(
            acc_S_row_converted, cute.make_layout(frg_tile)
        )
        for j in cutlass.range_constexpr(frg_cnt):
            for k in cutlass.range_constexpr(0, cute.size(acc_S_row_frg, mode=[0]), 2):
                # acc_S_row_frg[k, j] = utils.exp2f(acc_S_row_frg[k, j])
                # acc_S_row_frg[k + 1, j] = utils.exp2f(acc_S_row_frg[k + 1, j])
                if cutlass.const_expr(not e2e):
                    acc_S_row_frg[k, j] = cute.arch.exp2(acc_S_row_frg[k, j])
                    acc_S_row_frg[k + 1, j] = cute.arch.exp2(acc_S_row_frg[k + 1, j])
                else:
                    if cutlass.const_expr(
                        k % e2e_freq < e2e_freq - e2e_res
                        or j >= frg_cnt - e2e_frg_limit
                    ):
                        acc_S_row_frg[k, j] = cute.arch.exp2(acc_S_row_frg[k, j])
                        acc_S_row_frg[k + 1, j] = cute.arch.exp2(
                            acc_S_row_frg[k + 1, j]
                        )
                    else:
                        # acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j] = utils.e2e_asm2(acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j])
                        acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j] = (
                            utils.ex2_emulation_2(
                                acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j]
                            )
                        )
            acc_S_row_converted_frg[None, j].store(
                acc_S_row_frg[None, j].load().to(acc_S_row_converted.element_type)
            )

    @cute.jit
    def scale_apply_exp2_convert(
        self,
        acc_S_row: cute.Tensor,
        row_max: Float32,
        acc_S_row_converted: cute.Tensor,
    ):
        assert cute.size(acc_S_row.shape) % 2 == 0, (
            "acc_S_row must have an even number of elements"
        )
        minus_row_max_scaled = -row_max * self.scale_log2
        for i in cutlass.range_constexpr(0, cute.size(acc_S_row.shape), 2):
            acc_S_row[i], acc_S_row[i + 1] = utils.fma_packed_f32x2(
                (acc_S_row[i], acc_S_row[i + 1]),
                (self.scale_log2, self.scale_log2),
                (minus_row_max_scaled, minus_row_max_scaled),
            )

        # for i in cutlass.range_constexpr(0, cute.size(acc_S_row.shape), 2):
        #     acc_S_row[i], acc_S_row[i + 1] = utils.fma_packed_f32x2(
        #         (acc_S_row[i], acc_S_row[i + 1]),
        #         (self.scale_log2, self.scale_log2),
        #         (minus_row_max_scaled, minus_row_max_scaled),
        #     )
        #     acc_S_row[i] = cute.arch.exp2(acc_S_row[i])
        #     acc_S_row[i + 1] = cute.arch.exp2(acc_S_row[i + 1])

        frg_tile = 32
        assert frg_tile % 2 == 0
        frg_cnt = cute.size(acc_S_row) // frg_tile
        assert cute.size(acc_S_row) % frg_tile == 0
        acc_S_row_frg = cute.logical_divide(acc_S_row, cute.make_layout(frg_tile))
        acc_S_row_converted_frg = cute.logical_divide(
            acc_S_row_converted, cute.make_layout(frg_tile)
        )
        for j in cutlass.range_constexpr(frg_cnt):
            for k in cutlass.range_constexpr(0, cute.size(acc_S_row_frg, mode=[0]), 2):
                # acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j] = (
                #     utils.fma_packed_f32x2(
                #         (acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j]),
                #         (self.scale_log2, self.scale_log2),
                #         (minus_row_max_scaled, minus_row_max_scaled),
                #     )
                # )
                # acc_S_row_frg[k, j] = utils.exp2f(acc_S_row_frg[k, j])
                # acc_S_row_frg[k + 1, j] = utils.exp2f(acc_S_row_frg[k + 1, j])
                acc_S_row_frg[k, j] = cute.arch.exp2(acc_S_row_frg[k, j])
                acc_S_row_frg[k + 1, j] = cute.arch.exp2(acc_S_row_frg[k + 1, j])
            acc_S_row_converted_frg[None, j].store(
                acc_S_row_frg[None, j].load().to(acc_S_row_converted.element_type)
            )


@cute.jit
def floor_if_packed(
    q_idx,
    qhead_per_kvhead: cutlass.Constexpr[int],
) -> cute.Tensor:
    """Convert q_idx to packed format for Pack-GQA."""
    if cutlass.const_expr(qhead_per_kvhead == 1):
        return q_idx
    return q_idx // qhead_per_kvhead


@cute.jit
def apply_score_mod_inner(
    score_tensor,
    index_tensor,
    score_mod: cutlass.Constexpr,
    batch_idx,
    head_idx,
    softmax_scale,
    vec_size: cutlass.Constexpr,
    qk_acc_dtype: cutlass.Constexpr,
    aux_tensors,
    fastdiv_mods,
    constant_q_idx: cutlass.Constexpr,
    qhead_per_kvhead: cutlass.Constexpr[int] = 1,
):
    """Shared implementation for applying score modification.

    Args:
        score_tensor: The scores to modify (acc_S for flash_fwd, tSrS_t2r for sm100)
        index_tensor: Index positions (tScS for flash_fwd, tScS_t2r for sm100)
        score_mod: The score modification function to apply
        batch_idx: Batch index
        head_idx: Head index
        softmax_scale: Scale to apply
        vec_size: Vector size for processing elements
        qk_acc_dtype: Data type for accumulator
        aux_tensors: Optional aux_tensors for FlexAttention
        fastdiv_mods: Tuple of (seqlen_q_divmod, seqlen_k_divmod) for wrapping
        constant_q_idx: If provided, use this constant for all q_idx values
                       If None, compute q_idx per-element
        qhead_per_kvhead_packgqa: Pack-GQA replication factor. Divide q_idx by this
                                  when greater than 1 so score mods see logical heads.
    """
    n_vals = cutlass.const_expr(cute.size(score_tensor.shape))
    score_vec = cute.make_rmem_tensor(vec_size, qk_acc_dtype)
    kv_idx_vec = cute.make_rmem_tensor(vec_size, cutlass.Int32)

    # SSA values for batch (constant across all elements)
    batch_idx_ssa = utils.scalar_to_ssa(batch_idx, cutlass.Int32).broadcast_to(
        (vec_size,)
    )

    # Handle q_idx based on whether it's constant
    q_idx_vec = cute.make_rmem_tensor(vec_size, cutlass.Int32)

    # For Pack-GQA with non-constant q_idx, we need per-element head indices
    # since a thread my process multiple query head indices
    if cutlass.const_expr(qhead_per_kvhead > 1 and constant_q_idx is None):
        head_idx_vec = cute.make_rmem_tensor(vec_size, cutlass.Int32)

    for i in cutlass.range(0, n_vals, vec_size, unroll_full=True):
        for j in cutlass.range(vec_size, unroll_full=True):
            score_vec[j] = score_tensor[i + j] * softmax_scale

            # Extract head offset from packed q_idx for Pack-GQA
            if cutlass.const_expr(qhead_per_kvhead > 1 and constant_q_idx is None):
                q_idx_packed = index_tensor[i + j][0]
                # Building up the logical q_head idx: final_q_head = kv_head * qhead_per_kvhead + (q_physical % qhead_per_kvhead)
                q_idx_logical = q_idx_packed // qhead_per_kvhead
                head_offset = q_idx_packed - q_idx_logical * qhead_per_kvhead
                head_idx_vec[j] = head_idx * qhead_per_kvhead + head_offset

            # If we will do loads we mod, in order to not read OOB
            if cutlass.const_expr(aux_tensors is not None and fastdiv_mods is not None):
                if cutlass.const_expr(constant_q_idx is None):
                    seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
                    q_idx_floored = floor_if_packed(
                        index_tensor[i + j][0], qhead_per_kvhead
                    )
                    _, q_idx_wrapped = seqlen_q_divmod.divmod(q_idx_floored)
                    q_idx_vec[j] = q_idx_wrapped
                else:
                    _, seqlen_k_divmod = fastdiv_mods

                _, kv_idx_wrapped = seqlen_k_divmod.divmod(index_tensor[i + j][1])
                kv_idx_vec[j] = kv_idx_wrapped
            else:
                # No bounds checking - direct indexing
                if constant_q_idx is None:
                    q_idx_vec[j] = floor_if_packed(
                        index_tensor[i + j][0], qhead_per_kvhead
                    )
                kv_idx_vec[j] = index_tensor[i + j][1]

        # Convert to SSA for score_mod call
        score_ssa = score_vec.load()
        kv_idx_ssa = kv_idx_vec.load()
        if cutlass.const_expr(constant_q_idx is None):
            q_idx_ssa = q_idx_vec.load()
        else:
            # NB we do not apply Pack-GQA division here, as constant_q_idx is assumed to already be logical
            q_idx_const = constant_q_idx
            q_idx_ssa = utils.scalar_to_ssa(q_idx_const, cutlass.Int32).broadcast_to(
                (vec_size,)
            )

        # Compute head_idx_ssa: per-element for Pack-GQA with non-constant q_idx, constant otherwise
        if cutlass.const_expr(qhead_per_kvhead > 1 and constant_q_idx is None):
            head_idx_ssa = head_idx_vec.load()
        else:
            head_idx_ssa = utils.scalar_to_ssa(head_idx, cutlass.Int32).broadcast_to(
                (vec_size,)
            )

        aux_args = []
        if cutlass.const_expr(aux_tensors is not None):
            aux_args = aux_tensors

        post_mod_scores = score_mod(
            score_ssa,
            batch_idx_ssa,
            head_idx_ssa,
            q_idx=q_idx_ssa,
            kv_idx=kv_idx_ssa,
            aux_tensors=aux_args,
        )

        # Write back modified scores
        score_vec.store(post_mod_scores)
        for j in cutlass.range(vec_size, unroll_full=True):
            score_tensor[i + j] = score_vec[j]
