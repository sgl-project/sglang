# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Use of this software is governed by the terms and conditions of the
# NVIDIA End User License Agreement (EULA), available at:
# https://docs.nvidia.com/cutlass/media/docs/pythonDSL/license.html
#
# Any use, reproduction, disclosure, or distribution of this software
# and related documentation outside the scope permitted by the EULA
# is strictly prohibited.

from typing import Type, Tuple
from enum import Enum
from typing_extensions import deprecated
import warnings

from cutlass.utils.layout import LayoutEnum
from cutlass.cutlass_dsl import (
    Float16,
    BFloat16,
    Float8E5M2,
    Float8E4M3FN,
    Numeric,
    NumericMeta,
    dsl_user_op,
)

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu.common import CopyUniversalOp
from cutlass.cute.nvgpu.warp import StMatrix8x8x16bOp
from cutlass.cute.nvgpu.warpgroup import (
    MmaF16BF16Op,
    MmaF8Op,
    OperandMajorMode,
    OperandSource,
)


@deprecated("Use get_smem_capacity_in_bytes from cutlass.utils.smem_capacity instead")
class SmemCapacity(Enum):
    SM90_SMEM_CAPACITY_BYTES = (228 - 1) * 1024


warnings.warn(
    "SMEM_CAPACITY is deprecated: Use get_smem_capacity_in_bytes from cutlass.utils.smem_capacity instead",
    DeprecationWarning,
    stacklevel=2,
)
# Dictionary to map compute capability to SMEM capacity
SMEM_CAPACITY = {
    "sm90": SmemCapacity.SM90_SMEM_CAPACITY_BYTES.value,
}


@dsl_user_op
def sm90_get_smem_store_op(
    layout_d: LayoutEnum,
    elem_ty_d: Type[Numeric],
    elem_ty_acc: Type[Numeric],
    *,
    loc=None,
    ip=None,
) -> cute.CopyAtom:
    """
    Selects the largest vectorized smem store atom available subject to constraint of gmem layout.

    Parameters:
    -----------
    layout_d : LayoutEnum
        The layout enum of the output tensor D.

    elem_ty_d : Type[Numeric]
        The element type for output tensor D.

    elem_ty_acc : Type[Numeric]
        The element type for accumulator.

    Returns:
    --------
    Either SmemStoreMatrix or SimtSyncCopy, based on the input parameters.
    """

    def validate_type(ty, ty_name):
        if not isinstance(ty, NumericMeta):
            raise TypeError(f"{ty_name} must be a Numeric, but got {ty}")

    validate_type(elem_ty_d, "elem_ty_d")
    validate_type(elem_ty_acc, "elem_ty_acc")

    is_m_major = layout_d.is_m_major_c()

    if elem_ty_d.width == 16:
        return cute.make_copy_atom(
            StMatrix8x8x16bOp(is_m_major, 4), elem_ty_d, loc=loc, ip=ip
        )
    else:
        return cute.make_copy_atom(CopyUniversalOp(), elem_ty_d, loc=loc, ip=ip)


def make_trivial_tiled_mma(
    a_dtype: Type[Numeric],
    b_dtype: Type[Numeric],
    a_leading_mode: OperandMajorMode,
    b_leading_mode: OperandMajorMode,
    acc_dtype: Type[Numeric],
    atom_layout_mnk: Tuple[int, int, int],
    tiler_mn: Tuple[int, int],
    a_source: OperandSource = OperandSource.SMEM,
    *,
    loc=None,
    ip=None,
) -> cute.TiledMma:
    """Make a tiled MMA atom with given data type, leading dimension, cta group and mma tile shape.
    By default, the MMA atom is created with SMEM operand source for A.

    :param a_dtype: Data type of operand A.
    :type a_dtype: type[Numeric]
    :param b_dtype: Data type of operand B.
    :type b_dtype: type[Numeric]
    :param a_leading_mode: Leading dimension of operand A (1 for K, 0 for M/N).
    :type a_leading_mode: warpgroup.OperandMajorMode
    :param b_leading_mode: Leading dimension of operand B (1 for K, 0 for M/N).
    :type b_leading_mode: warpgroup.OperandMajorMode
    :param acc_dtype: Data type of the accumulator.
    :type acc_dtype: type[Numeric]
    :param atom_layout_mnk: A integer tuple describing the tiling of Atom across threads.
    :type atom_layout_mnk: Tuple[int, int, int]
    :param tiler_mn: The shape (M, N) of the cta tiler.
    :type tiler_mn: Tuple[int, int]

    :return: A tiled MMA atom.
    :rtype: cute.TiledMma

    :raises TypeError: If the data type is not supported.
    """

    if a_dtype in {Float16, BFloat16}:
        if cutlass.const_expr(a_dtype != b_dtype):
            raise TypeError(f"Type mismatch: {a_dtype} != {b_dtype}")
        if cutlass.const_expr(a_dtype.width != b_dtype.width):
            raise TypeError(f"Type width mismatch: {a_dtype.width} != {b_dtype.width}")

        mma_op = MmaF16BF16Op(
            a_dtype,
            acc_dtype,
            (*tiler_mn, 16),
            a_source,
            a_leading_mode,
            b_leading_mode,
        )
    elif a_dtype in {Float8E4M3FN, Float8E5M2} and b_dtype in {
        Float8E4M3FN,
        Float8E5M2,
    }:
        mma_op = MmaF8Op(
            a_dtype,
            b_dtype,
            acc_dtype,
            (*tiler_mn, 32),
            a_source,
            a_leading_mode,
            b_leading_mode,
        )
    else:
        raise TypeError(f"unsupported a_dtype and b_dtype, got {a_dtype} and {b_dtype}")

    return cute.make_tiled_mma(cute.make_mma_atom(mma_op), atom_layout_mnk)

def get_smem_layout_atom(
    layout: LayoutEnum,
    element_type: Type[Numeric],
    major_mode_size: int,
    *,
    loc=None,
    ip=None,
):
    """Select the optimal shared memory layout atom based on parameters.

    :param layout: Layout enum of the tensor
    :type layout: LayoutEnum
    :param element_type: Data type of the elements
    :type element_type: type[cutlass.Numeric]
    :param major_mode_size: Size of the major mode dimension
    :type major_mode_size: int

    :return: Selected shared memory layout atom kind
    :rtype: cute.nvgpu.warpgroup.SmemLayoutAtomKind
    """
    assert major_mode_size % 8 == 0
    sw128_num_contiguous_bits = 1024
    sw64_num_contiguous_bits = 512
    sw32_num_contiguous_bits = 256
    major_mode_size_bits = major_mode_size * element_type.width
    if layout.sm90_mma_major_mode() == OperandMajorMode.MN:
        if major_mode_size_bits % sw128_num_contiguous_bits == 0:
            return cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_SW128
        if major_mode_size_bits % sw64_num_contiguous_bits == 0:
            return cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_SW64
        if major_mode_size_bits % sw32_num_contiguous_bits == 0:
            return cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_SW32
        return cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_INTER
    if major_mode_size_bits % sw128_num_contiguous_bits == 0:
        return cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128
    if major_mode_size_bits % sw64_num_contiguous_bits == 0:
        return cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW64
    if major_mode_size_bits % sw32_num_contiguous_bits == 0:
        return cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW32
    return cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_INTER
