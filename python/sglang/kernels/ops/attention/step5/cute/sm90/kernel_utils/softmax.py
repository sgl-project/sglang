# Copyright (c) 2025, Tri Dao.
# Copyright (c) 2026 StepFun Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# Portions derived from Dao-AILab/flash-attention
# @0256114fe2381ab293503219bdd9078de3cd26b3, flash_attn/cute/softmax.py

import math
import operator
from dataclasses import dataclass, fields

import cutlass
import cutlass.cute as cute
from cutlass import Float32
from cutlass.cutlass_dsl import NumericMeta, dsl_user_op

import sglang.kernels.ops.attention.step5.cute.sm90.kernel_utils.cute_utils as utils


@cute.jit
def _exp2_tensor_or_scalar(x: cute.TensorSSA | Float32) -> cute.TensorSSA | Float32:
    return utils.exp2f(x)


@dsl_user_op
def _log2_approx_f32(a: float | Float32, *, loc=None, ip=None) -> Float32:
    return utils.log2f(a, loc=loc, ip=ip)


@cute.jit
def _fmax_reduce_f32(
    x: cute.TensorSSA, init_val: float | Float32 | None = None
) -> Float32:
    return utils.fmax_reduce(x, init_val=init_val)


@cute.jit
def _fadd_reduce_f32(
    x: cute.TensorSSA, init_val: float | Float32 | None = None
) -> Float32:
    return utils.fadd_reduce(x, init_val=init_val)


_STATIC_PARAM_TYPES = (
    cutlass.Constexpr,
    NumericMeta,
    int,
    bool,
    str,
    float,
    type(None),
)


class _DataclassMlirParamsBase:
    def __extract_mlir_values__(self):
        all_fields = [getattr(self, field.name) for field in fields(self)]
        dynamic_fields = [
            f for f in all_fields if not isinstance(f, _STATIC_PARAM_TYPES)
        ]
        values, self._values_pos = [], []
        for obj in dynamic_fields:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        all_fields = {field.name: getattr(self, field.name) for field in fields(self)}
        static_fields = {
            n: f for n, f in all_fields.items() if isinstance(f, _STATIC_PARAM_TYPES)
        }
        dynamic_fields = {
            n: f
            for n, f in all_fields.items()
            if not isinstance(f, _STATIC_PARAM_TYPES)
        }
        for (name, field), n_items in zip(
            dynamic_fields.items(), self._values_pos, strict=True
        ):
            dynamic_fields[name] = cutlass.new_from_mlir_values(field, values[:n_items])
            values = values[n_items:]
        return self.__class__(**dynamic_fields, **static_fields)


@dataclass
class Softmax(_DataclassMlirParamsBase):
    scale_log2: Float32
    num_rows: cutlass.Constexpr[int]
    row_max: cute.Tensor
    row_sum: cute.Tensor

    @staticmethod
    def create(
        scale_log2: Float32,
        num_rows: cutlass.Constexpr[int],
    ):
        row_max = cute.make_rmem_tensor(num_rows, Float32)
        row_sum = cute.make_rmem_tensor(num_rows, Float32)
        return Softmax(scale_log2, num_rows, row_max, row_sum)

    def reset(self) -> None:
        self.row_max.fill(-Float32.inf)
        self.row_sum.fill(0.0)

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
        acc_S_mn = utils.make_acc_tensor_mn_view_from_mma(acc_S)
        row_scale = cute.make_fragment_like(self.row_max, Float32)

        row_max = self.row_max
        row_sum = self.row_sum
        scale_log2 = self.scale_log2

        # Each iteration processes one row of acc_S
        for r in cutlass.range(cute.size(row_max), unroll_full=True):
            acc_S_row = acc_S_mn[r, None].load()  # (n_block_size)

            row_max_cur = _fmax_reduce_f32(
                acc_S_row,
                init_val=row_max[r] if cutlass.const_expr(not is_first) else None,
            )

            row_max_cur = utils.warp_reduce(row_max_cur, cute.arch.fmax, width=4)
            # A fully masked tile must not replace a finite negative running
            # maximum; only the exponent offset uses the zero substitute.
            row_max_safe = row_max_cur
            if cutlass.const_expr(check_inf):
                row_max_safe = 0.0 if row_max_cur == -Float32.inf else row_max_cur

            if cutlass.const_expr(is_first):
                row_max_cur_scaled = row_max_safe * scale_log2
                acc_S_row_exp = _exp2_tensor_or_scalar(
                    acc_S_row * scale_log2 - row_max_cur_scaled
                )

                acc_S_row_sum = _fadd_reduce_f32(acc_S_row_exp, init_val=None)
                row_scale[r] = 1.0
            else:
                row_max_prev = row_max[r]
                row_max_cur_scaled = row_max_safe * scale_log2
                acc_S_row_exp = _exp2_tensor_or_scalar(
                    acc_S_row * scale_log2 - row_max_cur_scaled
                )
                row_scale[r] = _exp2_tensor_or_scalar(
                    (row_max_prev - row_max_safe) * scale_log2
                )

                acc_S_row_sum = _fadd_reduce_f32(
                    acc_S_row_exp, init_val=row_sum[r] * row_scale[r]
                )

            row_max[r] = row_max_cur
            row_sum[r] = acc_S_row_sum
            acc_S_mn[r, None].store(acc_S_row_exp)

        return row_scale

    @cute.jit
    def finalize(self, final_scale: Float32 = 1.0) -> cute.Tensor:
        """Finalize the online softmax by computing the scale and logsumexp."""
        row_sum = self.row_sum
        row_max = self.row_max
        scale_log2 = self.scale_log2

        # quad reduction for row_sum as we didn't do it during each iteration of online softmax
        row_sum.store(utils.warp_reduce(row_sum.load(), operator.add, width=4))
        row_scale = cute.make_fragment_like(row_max, Float32)

        for r in cutlass.range(cute.size(row_sum), unroll_full=True):
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
                (row_max[r] * scale_log2 + _log2_approx_f32(row_sum_cur)) * LN2
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
        acc_O_mn = utils.make_acc_tensor_mn_view_from_mma(acc_O)
        if cutlass.const_expr(cute.size(row_scale) != cute.size(acc_O_mn, mode=[0])):
            raise ValueError("row_scale rows must match acc_O rows")
        for r in cutlass.range(cute.size(row_scale), unroll_full=True):
            acc_O_mn[r, None].store(acc_O_mn[r, None].load() * row_scale[r])


__all__ = ["Softmax"]
