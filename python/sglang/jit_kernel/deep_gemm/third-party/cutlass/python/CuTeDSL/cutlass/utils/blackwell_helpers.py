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

from enum import Enum
from math import log2, ceil
from typing import List, Type, Union, Tuple
from typing_extensions import deprecated
import warnings

from cutlass.cutlass_dsl import (
    Float16,
    BFloat16,
    TFloat32,
    Float32,
    Uint8,
    Int8,
    Float8E4M3FN,
    Float8E5M2,
    Float4E2M1FN,
    Numeric,
    NumericMeta,
    dsl_user_op,
)
import cutlass.cute as cute
from cutlass.cute.nvgpu.common import CopyUniversalOp
from cutlass.cute.nvgpu.warp import StMatrix8x8x16bOp, StMatrix16x8x8bOp
from cutlass.cute.nvgpu.tcgen05 import (
    MmaF16BF16Op,
    MmaTF32Op,
    MmaI8Op,
    MmaFP8Op,
    MmaMXF8Op,
    MmaMXF4Op,
    MmaMXF4NVF4Op,
    OperandSource,
    OperandMajorMode,
    CtaGroup,
    Ld16x64bOp,
    Ld16x128bOp,
    Ld16x256bOp,
    Ld16x32bx2Op,
    Ld32x32bOp,
    Repetition,
    Pack,
    find_tmem_tensor_col_offset,
    SmemLayoutAtomKind,
    make_smem_layout_atom,
    tile_to_mma_shape,
    is_tmem_load,
    get_tmem_copy_properties,
)
from cutlass.cute.nvgpu.cpasync import (
    CopyBulkTensorTileG2SMulticastOp,
    CopyBulkTensorTileG2SOp,
)
from cutlass.utils.layout import LayoutEnum


@deprecated("Use get_smem_capacity_in_bytes from cutlass.utils.smem_capacity instead")
class SmemCapacity(Enum):
    SM100_SMEM_CAPACITY_BYTES = (228 - 1) * 1024
    SM120_SMEM_CAPACITY_BYTES = (100 - 1) * 1024


warnings.warn(
    "SMEM_CAPACITY is deprecated: Use get_smem_capacity_in_bytes from cutlass.utils.smem_capacity instead",
    DeprecationWarning,
    stacklevel=2,
)
# Dictionary to map compute capability to SMEM capacity
SMEM_CAPACITY = {
    "sm100": SmemCapacity.SM100_SMEM_CAPACITY_BYTES.value,
    "sm120": SmemCapacity.SM120_SMEM_CAPACITY_BYTES.value,
}


@dsl_user_op
def compute_epilogue_tile_shape(
    cta_tile_shape: cute.Shape,
    use_2cta_instrs: bool,
    layout_d: LayoutEnum,
    elem_ty_d: Type[Numeric],
    *,
    layout_c: LayoutEnum = None,
    elem_ty_c: Union[Type[Numeric], None] = None,
    loc=None,
    ip=None,
) -> cute.Tile:
    """Attempts to compute a reasonable epilogue tile based on block tile shape or allows the user to provide one.

    :param cta_tile_shape: A tuple or list representing the dimensions of the CTA tile, where
        cta_tile_shape[0] corresponds to the height (M) and cta_tile_shape[1]
        corresponds to the width (N) of the tile.
    :type cta_tile_shape: cute.Shape
    :param use_2cta_instrs: A flag indicating whether the configuration is for a 2SM setup.
    :type use_2cta_instrs: bool
    :param layout_d: The layout enum of the output tensor D.
    :type layout_d: LayoutEnum
    :param elem_ty_d: The element type of output tensor D.
    :type elem_ty_d: Type[Numeric]
    :param layout_c: The layout enum of the input tensor C. Defaults to None.
    :type layout_c: LayoutEnum, optional
    :param elem_ty_c: The element type for input tensor C. Defaults to None.
    :type elem_ty_c: Union[Type[Numeric], None], optional

    :return: Returns epilog tiler, which is used in subsequent epilog partitions.
    :rtype: cute.Tile

    :raises ValueError: If the computed tile cute.size does not meet minimum requirements based on CTA dimensions.
    """

    def validate_type(ty, ty_name):
        if not isinstance(ty, NumericMeta):
            raise TypeError(f"{ty_name} must be Numeric, but got {ty}")

    validate_type(elem_ty_d, "elem_ty_d")
    if elem_ty_c is not None:
        validate_type(elem_ty_c, "elem_ty_c")

    cta_m, cta_n = cta_tile_shape[:2]
    (warp_m, warp_n) = (2, 2) if (cta_m == 64 and use_2cta_instrs) else (4, 1)
    disable_source = elem_ty_c == None
    max_bits = (
        elem_ty_d.width if disable_source else max(elem_ty_c.width, elem_ty_d.width)
    )

    dp_full = 32
    tile_m = min(cta_m, dp_full * warp_m)
    n_perf = 0
    if disable_source:
        if max_bits == 4:
            compute_elts = 8192
        else:
            compute_elts = 4096
        n_perf = compute_elts // tile_m
    else:
        if max_bits == 32:
            n_perf = 16 if (cta_m > 64 and cta_n <= 128) else 32
        elif max_bits == 16:
            n_perf = 32 if cta_n <= 128 else 64
        else:
            n_perf = 64

    d_is_m_major = layout_d.is_m_major_c()
    c_is_m_major = True if layout_c is None else layout_c.is_m_major_c()

    n_min_d = (
        8 * warp_n
        if d_is_m_major
        else (128 * warp_n if elem_ty_d.width == 6 else 128 // elem_ty_d.width * warp_n)
    )
    n_min_c = (
        8 * warp_n
        if (c_is_m_major or disable_source)
        else (128 * warp_n if elem_ty_c.width == 6 else 128 // elem_ty_c.width * warp_n)
    )
    tile_n = min(cta_n, max(n_perf, n_min_c, n_min_d))

    if cta_n < n_min_c or cta_n < n_min_d:
        raise ValueError(f"CTA tile too small: {cta_tile_shape=}")

    # stride by tmem warp layout and return a by-mode tiler
    tile_m_layout = cute.make_layout(tile_m, loc=loc, ip=ip)
    tile_n_layout = cute.make_layout(
        (tile_n // warp_n, warp_n), stride=(1, cta_n // warp_n), loc=loc, ip=ip
    )
    return (tile_m_layout, cute.coalesce(tile_n_layout, loc=loc, ip=ip))


@dsl_user_op
def get_smem_store_op(
    layout_d: LayoutEnum,
    elem_ty_d: Type[Numeric],
    elem_ty_acc: Type[Numeric],
    tiled_tmem_load: cute.TiledCopy,
    *,
    loc=None,
    ip=None,
) -> cute.CopyAtom:
    """Selects the largest vectorized smem store atom available subject to
    constraint of gmem layout and chosen TMEM_LOAD's thread-value ownership.

    :param layout_d: The layout enum of the output tensor D.
    :type layout_d: LayoutEnum
    :param elem_ty_d: The element type for output tensor D.
    :type elem_ty_d: Type[Numeric]
    :param elem_ty_acc: The element type for accumulator.
    :type elem_ty_acc: Type[Numeric]
    :param tiled_tmem_load: An instance of TiledCopy that represents the tmem load operation.
    :type tiled_tmem_load: cute.TiledCopy

    :return: Either SmemStoreMatrix or SimtSyncCopy, based on the input parameters.
    :rtype: cute.CopyAtom
    """

    def validate_type(ty, ty_name):
        if not isinstance(ty, NumericMeta):
            raise TypeError(f"{ty_name} must be a Numeric, but got {ty}")

    validate_type(elem_ty_d, "elem_ty_d")
    validate_type(elem_ty_acc, "elem_ty_acc")

    is_m_major = layout_d.is_m_major_c()
    is_n_major = layout_d.is_n_major_c()

    if not is_tmem_load(tiled_tmem_load):
        return cute.make_copy_atom(CopyUniversalOp(), elem_ty_d, loc=loc, ip=ip)

    num_dp, num_bits, num_rep, pack = get_tmem_copy_properties(tiled_tmem_load)

    use_stmatrix_m8n8_4x = (
        all(
            [
                elem_ty_acc.width == 32,
                elem_ty_d.width == 32,
                is_n_major,
                num_dp == 16,
                num_bits == 128,
                num_rep in (2, 4, 8, 16, 32, 64),
                pack == Pack.NONE,
            ]
        )
        or all(
            [
                elem_ty_acc.width == 32,
                elem_ty_d.width == 16,
                num_dp == 16,
                num_bits == 256,
                num_rep in (2, 4, 8, 16, 32),
                pack == Pack.NONE,
            ]
        )
        or all(
            [
                elem_ty_acc.width == 16,
                elem_ty_d.width == 16,
                num_dp == 16,
                num_bits == 128,
                num_rep in (2, 4, 8, 16, 32, 64),
                pack == Pack.PACK_16b_IN_32b,
            ]
        )
    )
    use_stmatrix_m16n8_4x = all(
        [
            elem_ty_acc.width == 32,
            elem_ty_d.width == 8,
            is_m_major,
            num_dp == 16,
            num_bits == 256,
            num_rep in (4, 8, 16, 32),
            pack == Pack.NONE,
        ]
    )
    use_stmatrix_m8n8_2x = (
        all(
            [
                elem_ty_acc.width == 32,
                elem_ty_d.width == 32,
                is_n_major,
                num_dp == 16,
                num_bits == 128,
                num_rep == 1,
                pack == Pack.NONE,
            ]
        )
        or all(
            [
                elem_ty_acc.width == 32,
                elem_ty_d.width == 16,
                num_dp == 16,
                num_bits == 256,
                num_rep == 1,
                pack == Pack.NONE,
            ]
        )
        or all(
            [
                elem_ty_acc.width == 16,
                elem_ty_d.width == 16,
                num_dp == 16,
                num_bits == 128,
                num_rep == 1,
                pack == Pack.PACK_16b_IN_32b,
            ]
        )
    )
    use_stmatrix_m16n8_2x = all(
        [
            elem_ty_acc.width == 32,
            elem_ty_d.width == 8,
            is_m_major,
            num_dp == 16,
            num_bits == 256,
            num_rep == 2,
            pack == Pack.NONE,
        ]
    )
    use_stmatrix_m16n8_1x = all(
        [
            elem_ty_acc.width == 32,
            elem_ty_d.width == 8,
            is_m_major,
            num_dp == 16,
            num_bits == 256,
            num_rep == 1,
            pack == Pack.NONE,
        ]
    )

    if use_stmatrix_m8n8_4x:
        op = StMatrix8x8x16bOp(is_m_major, 4)
        return cute.make_copy_atom(op, elem_ty_d, loc=loc, ip=ip)
    elif use_stmatrix_m8n8_2x:
        op = StMatrix8x8x16bOp(is_m_major, 2)
        return cute.make_copy_atom(op, elem_ty_d, loc=loc, ip=ip)
    elif use_stmatrix_m16n8_4x:
        op = StMatrix16x8x8bOp(4)
        return cute.make_copy_atom(op, elem_ty_d, loc=loc, ip=ip)
    elif use_stmatrix_m16n8_2x:
        op = StMatrix16x8x8bOp(2)
        return cute.make_copy_atom(op, elem_ty_d, loc=loc, ip=ip)
    elif use_stmatrix_m16n8_1x:
        op = StMatrix16x8x8bOp(1)
        return cute.make_copy_atom(op, elem_ty_d, loc=loc, ip=ip)
    else:
        op = CopyUniversalOp()
        return cute.make_copy_atom(op, elem_ty_d, loc=loc, ip=ip)


@dsl_user_op
def get_tmem_load_op(
    cta_tile_shape: cute.Shape,
    layout_d: LayoutEnum,
    elem_ty_d: Type[Numeric],
    elem_ty_acc: Type[Numeric],
    epi_tile: cute.Tile,
    use_2cta_instrs: bool,
    *,
    loc=None,
    ip=None,
) -> cute.CopyAtom:
    """Finds a performant TMEM_LOAD copy op for the selected epilogue
    tile (epi_tile), element types, and tcgen05.mma instruction used.

    :param cta_tile_shape: A tuple or list representing the dimensions of the CTA tile.
    :type cta_tile_shape: cute.Shape
    :param layout_d: The layout enum of the output tensor D.
    :type layout_d: LayoutEnum
    :param elem_ty_d: The element type for output tensor D.
    :type elem_ty_d: Type[Numeric]
    :param elem_ty_acc: The element type for accumulation.
    :type elem_ty_acc: Type[Numeric]
    :param epi_tile: The epilogue tile configuration.
    :type epi_tile: cute.Tile
    :param use_2cta_instrs: A flag indicating whether the configuration is for 2 SMs.
    :type use_2cta_instrs: bool

    :return: An instance of Sm100TmemLoad with the computed configuration.
    :rtype: cute.CopyAtom

    :raises ValueError: If the function cannot handle the given combination of accumulation
    and dimension types, or if it cannot determine the appropriate configuration based on
    the input parameters.
    """
    is_m_major = layout_d.is_m_major_c()

    acc_bits = elem_ty_acc.width
    d_bits = elem_ty_d.width

    tmem_warp_shape_mn = (
        (2, 2) if (cta_tile_shape[0] == 64 and use_2cta_instrs) else (4, 1)
    )
    epilog_tile_shape_mn = cute.product_each(
        cute.shape(epi_tile, loc=loc, ip=ip), loc=loc, ip=ip
    )
    epilog_warp_tile_shape_mn = cute.shape_div(
        epilog_tile_shape_mn, tmem_warp_shape_mn, loc=loc, ip=ip
    )

    num_dp = cute.size(epilog_warp_tile_shape_mn[0], loc=loc, ip=ip)
    if num_dp not in {16, 32}:
        raise ValueError("Cta tile and 2sm config does not generate correct num dp.")

    num_col_bits = cute.size(epilog_warp_tile_shape_mn[1], loc=loc, ip=ip) * acc_bits

    tmem_dp = 0
    tmem_bit = 0
    tmem_rep = 0
    tmem_pack16b = False
    if acc_bits == 32 and d_bits == 32:
        if num_dp == 16:
            if is_m_major:
                tmem_dp = 16
                tmem_bit = 256
            else:
                tmem_dp = 16
                tmem_bit = 128
        else:
            tmem_dp = 32
            tmem_bit = 32
    elif acc_bits == 32 and d_bits == 16:
        if num_dp == 16:
            if is_m_major:
                tmem_dp = 16
                tmem_bit = 256
            else:
                tmem_dp = 16
                tmem_bit = 256
        else:
            if is_m_major:
                tmem_dp = 16
                tmem_bit = 256
            else:
                tmem_dp = 32
                tmem_bit = 32
    elif acc_bits == 32 and d_bits == 8:
        if num_dp == 16:
            if is_m_major:
                tmem_dp = 16
                tmem_bit = 256
            else:
                tmem_dp = 16
                tmem_bit = 32
        else:
            if is_m_major:
                tmem_dp = 16
                tmem_bit = 256
            else:
                tmem_dp = 32
                tmem_bit = 32
    elif acc_bits == 16 and d_bits == 16:
        tmem_pack16b = True
        if num_dp == 16:
            if is_m_major:
                tmem_dp = 16
                tmem_bit = 128
            else:
                tmem_dp = 16
                tmem_bit = 128
        else:
            if is_m_major:
                tmem_dp = 16
                tmem_bit = 128
            else:
                tmem_dp = 32
                tmem_bit = 32
    elif acc_bits == 32 and d_bits == 6:
        if not num_dp == 32:
            raise ValueError("Num dp must be 32.")
        tmem_dp = 32
        tmem_bit = 32
    elif acc_bits == 32 and d_bits == 4:
        if not num_dp == 32:
            raise ValueError("Num dp must be 32.")
        tmem_dp = 32
        tmem_bit = 32
    else:
        raise ValueError(
            f"Can not handle acc/d type combination: {elem_ty_acc=}, {elem_ty_d=}"
        )

    num_bit_div = tmem_bit
    if tmem_dp == 16 and tmem_bit == 32:
        num_bit_div = 64

    if (num_col_bits % (num_bit_div * 128) == 0) and (
        (tmem_dp == 16 and tmem_bit == 64)
        or (tmem_dp == 16 and tmem_bit == 32)
        or (tmem_dp == 32 and tmem_bit == 32)
    ):
        tmem_rep = 128
    elif (num_col_bits % (num_bit_div * 64) == 0) and (
        (tmem_dp == 16 and tmem_bit == 128)
        or (tmem_dp == 16 and tmem_bit == 64)
        or (tmem_dp == 16 and tmem_bit == 32)
        or (tmem_dp == 32 and tmem_bit == 32)
    ):
        tmem_rep = 64
    elif num_col_bits % (num_bit_div * 32) == 0:
        tmem_rep = 32
    elif num_col_bits % (num_bit_div * 16) == 0:
        tmem_rep = 16
    elif num_col_bits % (num_bit_div * 8) == 0:
        tmem_rep = 8
    elif num_col_bits % (num_bit_div * 4) == 0:
        tmem_rep = 4
    elif num_col_bits % (num_bit_div * 2) == 0:
        tmem_rep = 2
    elif num_col_bits % (num_bit_div * 1) == 0:
        tmem_rep = 1
    else:
        raise ValueError("Can not pick tmem_rep based on cta tile shape and tmem atom.")

    if tmem_dp == 16 and tmem_bit == 64:
        op = Ld16x64bOp(
            Repetition(tmem_rep), Pack.PACK_16b_IN_32b if tmem_pack16b else Pack.NONE
        )
        return cute.make_copy_atom(op, elem_ty_acc, loc=loc, ip=ip)
    elif tmem_dp == 16 and tmem_bit == 128:
        op = Ld16x128bOp(
            Repetition(tmem_rep), Pack.PACK_16b_IN_32b if tmem_pack16b else Pack.NONE
        )
        return cute.make_copy_atom(op, elem_ty_acc, loc=loc, ip=ip)
    elif tmem_dp == 16 and tmem_bit == 256:
        op = Ld16x256bOp(
            Repetition(tmem_rep), Pack.PACK_16b_IN_32b if tmem_pack16b else Pack.NONE
        )
        return cute.make_copy_atom(op, elem_ty_acc, loc=loc, ip=ip)
    elif tmem_dp == 16 and tmem_bit == 32:
        op = Ld16x32bx2Op(
            Repetition(tmem_rep), Pack.PACK_16b_IN_32b if tmem_pack16b else Pack.NONE
        )
        return cute.make_copy_atom(op, elem_ty_acc, loc=loc, ip=ip)

    elif tmem_dp == 32 and tmem_bit == 32:
        op = Ld32x32bOp(
            Repetition(tmem_rep), Pack.PACK_16b_IN_32b if tmem_pack16b else Pack.NONE
        )
        return cute.make_copy_atom(op, elem_ty_acc, loc=loc, ip=ip)
    else:
        raise ValueError()


def get_num_tmem_alloc_cols(
    tmem_tensors: Union[cute.Tensor, List[cute.Tensor]], rounding=True
) -> int:
    """Get the total number of TMEM allocation columns for the given TMEM tensors.

    :param tmem_tensors: The TMEM tensors to get the number of allocation columns for.
    :type tmem_tensors: Union[cute.Tensor, List[cute.Tensor]]
    :param rounding: Whether to round up the number of allocation columns to the nearest power of 2.
    :type rounding: bool

    :return: The total number of TMEM allocation columns.
    :rtype: int

    :raises ValueError: If the number of TMEM allocation columns exceeds the maximum capacity of 512 or is less than 32.
    """
    # Turn tmem_tensors into a list
    if isinstance(tmem_tensors, cute.Tensor):
        tmem_tensors = [tmem_tensors]

    # For each tensor in tmem_tensors, find the tmem_tensor_col_offset
    num_tmem_alloc_cols_per_tensor = [
        find_tmem_tensor_col_offset(t) for t in tmem_tensors
    ]

    # Sum up the num_tmem_alloc_cols_per_tensor
    num_tmem_alloc_cols = sum(num_tmem_alloc_cols_per_tensor)

    # Round up num_tmem_cols_total to the nearest power of 2
    if rounding:
        num_tmem_alloc_cols = 1 << ceil(log2(num_tmem_alloc_cols))

    # Validate the number of TMEM allocation columns
    SM100_TMEM_CAPACITY_COLUMNS = 512
    SM100_TMEM_MIN_ALLOC_COLUMNS = 32
    if (
        num_tmem_alloc_cols > SM100_TMEM_CAPACITY_COLUMNS
        or num_tmem_alloc_cols < SM100_TMEM_MIN_ALLOC_COLUMNS
    ):
        raise ValueError(
            f"TMEM allocation columns {num_tmem_alloc_cols} exceeds the maximum capacity of {SM100_TMEM_CAPACITY_COLUMNS} or less than {SM100_TMEM_MIN_ALLOC_COLUMNS}"
        )
    return num_tmem_alloc_cols


def get_smem_layout_atom_ab(
    major_mode: OperandMajorMode,
    element_type: Type[Numeric],
    smem_shape_mn_k: Tuple[int, int],
    *,
    loc=None,
    ip=None,
) -> SmemLayoutAtomKind:
    """Simple heuristics to select the optimal SMEM layout atom based on the
    majorness, the data type, and the major mode size.

    :param major_mode: The major mode for the SMEM tensor is K major.
    :type major_mode: OperandMajorMode
    :param element_type: The element type for the SMEM tensor.
    :type element_type: Type[Numeric]
    :param smem_shape_mn_k: The shape of the SMEM tensor.
    :type smem_shape_mn_k: Tuple[int, int]

    :return: The SMEM layout atom kind
    :rtype: SmemLayoutAtomKind
    """
    is_k_major = major_mode == OperandMajorMode.K
    major_mode_size = smem_shape_mn_k[1] if is_k_major else smem_shape_mn_k[0]

    assert major_mode_size % 8 == 0
    sw128_num_contiguous_bits = 1024
    sw64_num_contiguous_bits = 512
    sw32_num_contiguous_bits = 256
    inter_num_contiguous_bits = 128
    major_mode_size_bits = major_mode_size * element_type.width
    assert major_mode_size_bits % inter_num_contiguous_bits == 0

    if not is_k_major:
        if (element_type.width == 32) and (
            major_mode_size_bits % sw128_num_contiguous_bits == 0
        ):
            return SmemLayoutAtomKind.MN_SW128_32B
        if major_mode_size_bits % sw128_num_contiguous_bits == 0:
            return SmemLayoutAtomKind.MN_SW128
        if major_mode_size_bits % sw64_num_contiguous_bits == 0:
            return SmemLayoutAtomKind.MN_SW64
        if major_mode_size_bits % sw32_num_contiguous_bits == 0:
            return SmemLayoutAtomKind.MN_SW32
        return SmemLayoutAtomKind.MN_INTER
    if major_mode_size_bits % sw128_num_contiguous_bits == 0:
        return SmemLayoutAtomKind.K_SW128
    if major_mode_size_bits % sw64_num_contiguous_bits == 0:
        return SmemLayoutAtomKind.K_SW64
    if major_mode_size_bits % sw32_num_contiguous_bits == 0:
        return SmemLayoutAtomKind.K_SW32
    return SmemLayoutAtomKind.K_INTER


@dsl_user_op
def make_smem_layout_a(
    tiled_mma: cute.TiledMma,
    mma_tiler_mnk: cute.Tile,
    a_dtype: Type[Numeric],
    num_stages: int,
    *,
    loc=None,
    ip=None,
) -> Union[cute.Layout, cute.ComposedLayout]:
    """This function helps with:
    1. Get the partitioned shape of the A tensor based on the tiled_mma & MMA tiler.
    2. Select the heuristic SMEM layout atom based on the A tensor's majorness, the data type, and the major mode size.
    3. cute.Tile the SMEM layout atom to the MMA tile shape.
    4. Stage the SMEM layout based on the number of stages.

    :param tiled_mma: The tiled MMA used to partition tensor A
    :type tiled_mma: cute.TiledMma
    :param mma_tiler_mnk: The MMA tile shape
    :type mma_tiler_mnk: cute.cute.Tile
    :param a_dtype: The element type for tensor A
    :type a_dtype: Type[Numeric]
    :param num_stages: The number of pipeline stages for tensor A
    :type num_stages: int

    :return: SMEM layout for tensor A
    :rtype: Union[cute.Layout, cute.ComposedLayout]
    """

    is_k_major = tiled_mma.op.a_major_mode == OperandMajorMode.K
    a_smem_shape = tiled_mma.partition_shape_A(
        cute.dice(mma_tiler_mnk, (1, None, 1), loc=loc, ip=ip)
    )
    a_smem_shape_mn_k = (
        cute.size(a_smem_shape[0][0], loc=loc, ip=ip) * a_smem_shape[1],
        cute.size(a_smem_shape[0][1], loc=loc, ip=ip) * a_smem_shape[2],
    )
    a_smem_layout_atom = make_smem_layout_atom(
        get_smem_layout_atom_ab(
            tiled_mma.op.a_major_mode,
            a_dtype,
            a_smem_shape_mn_k,
            loc=loc,
            ip=ip,
        ),
        a_dtype,
        loc=loc,
        ip=ip,
    )
    a_smem_layout_staged = tile_to_mma_shape(
        a_smem_layout_atom,
        cute.append(a_smem_shape, num_stages, loc=loc, ip=ip),
        order=((1, 0, 2) if not is_k_major else (0, 1, 2)),
        loc=loc,
        ip=ip,
    )
    return a_smem_layout_staged


@dsl_user_op
def make_smem_layout_b(
    tiled_mma: cute.TiledMma,
    mma_tiler_mnk: cute.Tile,
    b_dtype: Type[Numeric],
    num_stages: int,
    *,
    loc=None,
    ip=None,
) -> Union[cute.Layout, cute.ComposedLayout]:
    """This function helps:
    1. Get the partitioned shape of the B tensor based on the tiled_mma & MMA tiler.
    2. Select the heuristic SMEM layout atom based on the B tensor's majorness, the data type, and the major mode size.
    3. cute.Tile the SMEM layout atom to the MMA tile shape.
    4. Stage the SMEM layout based on the number of stages.

    :param tiled_mma: The tiled MMA which is used to partition the B tensor.
    :type tiled_mma: cute.TiledMma
    :param mma_tiler_mnk: The MMA tile shape.
    :type mma_tiler_mnk: cute.cute.Tile
    :param b_dtype: The element type for the B tensor.
    :type b_dtype: Type[Numeric]
    :param num_stages: The stage of the B tensor.
    :type num_stages: int

    :return: SMEM layout for the B tensor.
    :rtype: Union[cute.Layout, cute.ComposedLayout]
    """

    is_k_major = tiled_mma.op.b_major_mode == OperandMajorMode.K
    b_smem_shape = tiled_mma.partition_shape_B(
        cute.dice(mma_tiler_mnk, (None, 1, 1), loc=loc, ip=ip)
    )
    b_smem_shape_nk = (
        cute.size(b_smem_shape[0][0], loc=loc, ip=ip) * b_smem_shape[1],
        cute.size(b_smem_shape[0][1], loc=loc, ip=ip) * b_smem_shape[2],
    )
    b_smem_layout_atom = make_smem_layout_atom(
        get_smem_layout_atom_ab(
            tiled_mma.op.b_major_mode,
            b_dtype,
            b_smem_shape_nk,
            loc=loc,
            ip=ip,
        ),
        b_dtype,
        loc=loc,
        ip=ip,
    )
    b_smem_layout_staged = tile_to_mma_shape(
        b_smem_layout_atom,
        cute.append(b_smem_shape, num_stages, loc=loc, ip=ip),
        order=((1, 0, 2) if not is_k_major else (0, 1, 2)),
        loc=loc,
        ip=ip,
    )

    return b_smem_layout_staged


@dsl_user_op
def get_smem_layout_atom_epi(
    layout: LayoutEnum,
    element_type: Type[Numeric],
    epi_tile: cute.Tile,
    *,
    loc=None,
    ip=None,
) -> SmemLayoutAtomKind:
    """Simple heuristics to select the optimal SMEM layout atom for epilog tensors.

    :param layout: The layout enum for the SMEM tensor.
    :type layout: LayoutEnum
    :param element_type: The element type for the SMEM tensor.
    :type element_type: Type[Numeric]
    :param epi_tile: The epilogue tile shape.
    :type epi_tile: cute.Tile

    :return: The SMEM layout atom kind
    :rtype: SmemLayoutAtomKind
    """
    # Get the max contiguous tile usable by TMA
    tma_shape = tuple(
        (
            # assumes get<0>(epi_tile) is coalesced and unit stride
            cute.coalesce(cute.right_inverse(x, loc=loc, ip=ip), loc=loc, ip=ip).shape
            if isinstance(x, cute.Layout)
            else x
        )
        for x in epi_tile
    )

    if layout.is_m_major_c():
        # ColMajor C/D (M-major)
        return get_smem_layout_atom_ab(
            OperandMajorMode.MN, element_type, tma_shape, loc=loc, ip=ip
        )
    else:
        # RowMajor C/D (N-major)
        return get_smem_layout_atom_ab(
            OperandMajorMode.K, element_type, tma_shape, loc=loc, ip=ip
        )


@dsl_user_op
def make_smem_layout_epi(
    epi_dtype: Type[Numeric],
    epi_layout: LayoutEnum,
    epi_tile: cute.Tile,
    epi_stage: int,
    *,
    loc=None,
    ip=None,
) -> Union[cute.Layout, cute.ComposedLayout]:
    """This function helps:
    1. Select the heuristic SMEM layout atom based on the epilog tile shape,
       the epilog tensor's majorness, and the element type.
    2. cute.Tile the SMEM layout atom to the epilog tile shape.
    3. Stage the SMEM layout based on the number of stages.

    :param epi_dtype: The element type for the epilog tensor.
    :type epi_dtype: Type[Numeric]
    :param epi_layout: The layout enum for the epilog tensor.
    :type epi_layout: LayoutEnum
    :param epi_tile: The epilogue tile shape.
    :type epi_tile: cute.cute.Tile
    :param epi_stage: The stage of the epilog tensor.
    :type epi_stage: int

    :return: SMEM layout for epilog tensors (usually C & D which are processed in the epilog)
    :rtype: Union[cute.Layout, cute.ComposedLayout]
    """

    epilog_shape = cute.product_each(
        cute.shape(epi_tile, loc=loc, ip=ip), loc=loc, ip=ip
    )

    c_smem_layout_atom = make_smem_layout_atom(
        get_smem_layout_atom_epi(
            epi_layout,
            epi_dtype,
            epi_tile,
            loc=loc,
            ip=ip,
        ),
        epi_dtype,
        loc=loc,
        ip=ip,
    )
    epi_smem_layout_staged = cute.tile_to_shape(
        c_smem_layout_atom,
        cute.append(epilog_shape, epi_stage, loc=loc, ip=ip),
        order=((1, 0, 2) if not epi_layout.is_n_major_c() else (0, 1, 2)),
        loc=loc,
        ip=ip,
    )

    return epi_smem_layout_staged


@dsl_user_op
def make_trivial_tiled_mma(
    ab_dtype: Type[Numeric],
    a_leading_mode: OperandMajorMode,
    b_leading_mode: OperandMajorMode,
    acc_dtype: Type[Numeric],
    cta_group: CtaGroup,
    mma_tiler_mn: Tuple[int, int],
    a_source: OperandSource = OperandSource.SMEM,
    *,
    loc=None,
    ip=None,
) -> cute.TiledMma:
    """Make a tiled MMA atom with given data type, leading dimension, cta group and mma tile shape.
    By default, the MMA atom is created with SMEM operand source for A.

    :param ab_dtype: Data type of operands A and B.
    :type ab_dtype: type[Numeric]
    :param a_leading_mode: Leading dimension of operand A (1 for K, 0 for M/N).
    :type a_leading_mode: tcgen05.OperandMajorMode
    :param b_leading_mode: Leading dimension of operand B (1 for K, 0 for M/N).
    :type b_leading_mode: tcgen05.OperandMajorMode
    :param acc_dtype: Data type of the accumulator.
    :type acc_dtype: type[Numeric]
    :param cta_group: The CTA group to use.
    :type cta_group: tcgen05.CtaGroup
    :param mma_tiler_mn: The shape (M, N, K) of the MMA tiler.
    :type mma_tiler_mn: Tuple[int, int]
    :param a_source: The source of operand A (SMEM by default or TMEM).
    :type a_source: OperandSource

    :return: A tiled MMA atom.
    :rtype: cute.TiledMma

    :raises TypeError: If the data type is not supported.
    """

    if ab_dtype in {Float16, BFloat16}:
        mma_op = MmaF16BF16Op(
            ab_dtype,
            acc_dtype,
            (*mma_tiler_mn, 16),
            cta_group,
            a_source,
            a_leading_mode,
            b_leading_mode,
        )
    elif ab_dtype in {TFloat32, Float32}:
        mma_op = MmaTF32Op(
            (*mma_tiler_mn, 8),
            cta_group,
            a_source,
            a_leading_mode,
            b_leading_mode,
        )
    elif ab_dtype in {
        Uint8,
        Int8,
    }:
        mma_op = MmaI8Op(
            ab_dtype,
            (*mma_tiler_mn, 32),
            cta_group,
            a_source,
            a_leading_mode,
            b_leading_mode,
        )
    elif ab_dtype in {Float8E4M3FN, Float8E5M2}:
        mma_op = MmaFP8Op(
            ab_dtype,
            acc_dtype,
            (*mma_tiler_mn, 32),
            cta_group,
            a_source,
            a_leading_mode,
            b_leading_mode,
        )
    else:
        raise TypeError(f"unsupported ab_dtype, got {ab_dtype}")

    return cute.make_tiled_mma(cute.make_mma_atom(mma_op))


@dsl_user_op
def make_blockscaled_trivial_tiled_mma(
    ab_dtype: Type[Numeric],
    a_leading_mode: OperandMajorMode,
    b_leading_mode: OperandMajorMode,
    sf_dtype: Type[Numeric],
    sf_vec_size: int,
    cta_group: CtaGroup,
    mma_tiler_mn: Tuple[int, int],
    a_source: OperandSource = OperandSource.SMEM,
    *,
    loc=None,
    ip=None,
) -> cute.TiledMma:
    """Make a BlockScaled tiled MMA atom with given data type, leading dimension, cta group and mma tile shape.
    By default, the MMA atom is created with SMEM operand source for A.

    :param ab_dtype: Data type of operands A and B.
    :type ab_dtype: type[Numeric]
    :param a_leading_mode: Leading dimension of operand A (1 for K, 0 for M/N).
    :type a_leading_mode: tcgen05.OperandMajorMode
    :param b_leading_mode: Leading dimension of operand B (1 for K, 0 for M/N).
    :type b_leading_mode: tcgen05.OperandMajorMode
    :param sf_dtype: Data type of the Scale Factor.
    :type sf_dtype: type[Numeric]
    :param sf_vec_size: The vector size of the Scale Factor.
    :type sf_vec_size: int
    :param cta_group: The CTA group to use.
    :type cta_group: tcgen05.CtaGroup
    :param mma_tiler_mn: The shape (M, N, K) of the MMA tiler.
    :type mma_tiler_mn: Tuple[int, int]
    :param a_source: The source of operand A (SMEM by default or TMEM).
    :type a_source: OperandSource

    :return: A tiled MMA atom.
    :rtype: cute.TiledMma

    :raises TypeError: If the data type is not supported.
    """
    if ab_dtype in {Float8E4M3FN, Float8E5M2}:
        mma_op = MmaMXF8Op(
            ab_dtype,
            (*mma_tiler_mn, 32),
            cta_group,
            a_source,
            a_leading_mode,
            b_leading_mode,
        )
    elif ab_dtype == Float4E2M1FN:
        if sf_vec_size == 32:
            mma_op = MmaMXF4Op(
                (*mma_tiler_mn, 64),
                cta_group,
                a_source,
            )
        elif sf_vec_size == 16:
            mma_op = MmaMXF4NVF4Op(
                sf_dtype,
                (*mma_tiler_mn, 64),
                cta_group,
                a_source,
            )
        else:
            raise ValueError(f"unsupported sf_vec_size, got {sf_vec_size}")
    else:
        raise TypeError(f"unsupported ab_dtype, got {ab_dtype}")

    return cute.make_tiled_mma(cute.make_mma_atom(mma_op))


@dsl_user_op
def cluster_shape_to_tma_atom_A(
    cluster_shape_mnk: cute.Shape, atom_thr_id: cute.Layout, *, loc=None, ip=None
) -> Union[CopyBulkTensorTileG2SMulticastOp, CopyBulkTensorTileG2SOp]:
    """
    Select the appropriate TMA copy atom for A based on the number of SMs and the multicast flag.

    :param cluster_shape_mnk: The shape of the cluster
    :type cluster_shape_mnk: cute.Shape
    :param atom_thr_id: The thread ID of the atom
    :type atom_thr_id: cute.Layout

    :return: The appropriate TMA copy atom kind
    :rtype: cpasync.CopyBulkTensorTileG2SMulticastOp or cpasync.CopyBulkTensorTileG2SOp

    :raise ValueError: If the atom_sm_cnt is invalid
    :raise ValueError: If the cluster shape is not divisible by the atom SM count
    """
    atom_sm_cnt = cute.size(atom_thr_id, loc=loc, ip=ip)
    mcast = not (cute.size(cluster_shape_mnk, mode=[1], loc=loc, ip=ip) == 1)
    cluster_size = cute.size(cluster_shape_mnk, loc=loc, ip=ip)

    if not isinstance(cluster_size, int) or not isinstance(atom_sm_cnt, int):
        raise ValueError(
            f"Dynamic cluster shape or atom SM count is not supported: {cluster_shape_mnk} and {atom_thr_id}"
        )

    if cute.size(cluster_shape_mnk, mode=[0], loc=loc, ip=ip) % atom_sm_cnt != 0:
        raise ValueError(
            f"Cluster shape not divisible by MMA size: {cluster_shape_mnk} and {atom_thr_id}"
        )

    if atom_sm_cnt == 2 and mcast:
        return CopyBulkTensorTileG2SMulticastOp(CtaGroup.TWO)
    elif atom_sm_cnt == 2 and not mcast:
        return CopyBulkTensorTileG2SOp(CtaGroup.TWO)
    elif atom_sm_cnt == 1 and mcast:
        return CopyBulkTensorTileG2SMulticastOp(CtaGroup.ONE)
    elif atom_sm_cnt == 1 and not mcast:
        return CopyBulkTensorTileG2SOp(CtaGroup.ONE)

    raise ValueError(
        f"Unsupported Configuration for SM100 TMA: {cluster_shape_mnk} and {atom_thr_id}"
    )


@dsl_user_op
def cluster_shape_to_tma_atom_B(
    cluster_shape_mnk: cute.Shape, atom_thr_id: cute.Layout, *, loc=None, ip=None
) -> Union[CopyBulkTensorTileG2SMulticastOp, CopyBulkTensorTileG2SOp]:
    """
    Select the appropriate TMA copy atom for Bbased on the number of SMs and the multicast flag.

    :param cluster_shape_mnk: The shape of the cluster
    :type cluster_shape_mnk: cute.Shape
    :param atom_thr_id: The thread ID of the atom
    :type atom_thr_id: cute.Layout

    :return: The appropriate TMA copy atom kind
    :rtype: cpasync.CopyBulkTensorTileG2SMulticastOp or cpasync.CopyBulkTensorTileG2SOp

    :raise ValueError: If the atom_sm_cnt is invalid
    :raise ValueError: If the cluster shape is not divisible by the atom SM count
    """
    atom_sm_cnt = cute.size(atom_thr_id, loc=loc, ip=ip)
    mcast = not (cute.size(cluster_shape_mnk, mode=[0], loc=loc, ip=ip) == atom_sm_cnt)
    cluster_size = cute.size(cluster_shape_mnk, loc=loc, ip=ip)

    if not isinstance(cluster_size, int) or not isinstance(atom_sm_cnt, int):
        raise ValueError(
            f"Dynamic cluster shape or atom SM count is not supported: {cluster_shape_mnk} and {atom_thr_id}"
        )

    if cute.size(cluster_shape_mnk, mode=[0], loc=loc, ip=ip) % atom_sm_cnt != 0:
        raise ValueError(
            f"Cluster shape not divisible by MMA size: {cluster_shape_mnk} and {atom_thr_id}"
        )

    if atom_sm_cnt == 2 and mcast:
        return CopyBulkTensorTileG2SMulticastOp(CtaGroup.TWO)
    elif atom_sm_cnt == 2 and not mcast:
        return CopyBulkTensorTileG2SOp(CtaGroup.TWO)
    elif atom_sm_cnt == 1 and mcast:
        return CopyBulkTensorTileG2SMulticastOp(CtaGroup.ONE)
    elif atom_sm_cnt == 1 and not mcast:
        return CopyBulkTensorTileG2SOp(CtaGroup.ONE)

    raise ValueError(
        f"Unsupported Configuration for SM100 TMA: {cluster_shape_mnk} and {atom_thr_id}"
    )


@dsl_user_op
def cluster_shape_to_tma_atom_SFB(
    cluster_shape_mnk: cute.Shape, atom_thr_id: cute.Layout, *, loc=None, ip=None
) -> Union[CopyBulkTensorTileG2SMulticastOp, CopyBulkTensorTileG2SOp]:
    """
    Select the appropriate TMA copy atom for SFB based on the number of SMs and the multicast flag.

    :param cluster_shape_mnk: The shape of the cluster
    :type cluster_shape_mnk: cute.Shape
    :param atom_thr_id: The thread ID of the atom
    :type atom_thr_id: cute.Layout

    :return: The appropriate TMA copy atom kind
    :rtype: cpasync.CopyBulkTensorTileG2SMulticastOp or cpasync.CopyBulkTensorTileG2SOp

    :raise ValueError: If the atom_sm_cnt is invalid
    :raise ValueError: If the cluster shape is not divisible by the atom SM count
    """
    atom_sm_cnt = cute.size(atom_thr_id, loc=loc, ip=ip)
    mcast = not (cute.size(cluster_shape_mnk, mode=[0], loc=loc, ip=ip) == 1)
    cluster_size = cute.size(cluster_shape_mnk, loc=loc, ip=ip)

    if not isinstance(cluster_size, int) or not isinstance(atom_sm_cnt, int):
        raise ValueError(
            f"Dynamic cluster shape or atom SM count is not supported: {cluster_shape_mnk} and {atom_thr_id}"
        )

    if cute.size(cluster_shape_mnk, mode=[0], loc=loc, ip=ip) % atom_sm_cnt != 0:
        raise ValueError(
            f"Cluster shape not divisible by MMA size: {cluster_shape_mnk} and {atom_thr_id}"
        )

    if atom_sm_cnt == 2:
        return CopyBulkTensorTileG2SMulticastOp(CtaGroup.TWO)
    elif atom_sm_cnt == 1 and mcast:
        return CopyBulkTensorTileG2SMulticastOp(CtaGroup.ONE)
    elif atom_sm_cnt == 1 and not mcast:
        return CopyBulkTensorTileG2SOp(CtaGroup.ONE)

    raise ValueError(
        f"Unsupported Configuration for SM100 TMA: {cluster_shape_mnk} and {atom_thr_id}"
    )
