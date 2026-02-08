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

from typing import overload, Type, Tuple, Union

from cutlass.cutlass_dsl import dsl_user_op

import cutlass._mlir.dialects.cute as _cute_ir
import cutlass._mlir.dialects.cute_nvgpu as _cute_nvgpu_ir
from cutlass._mlir.dialects import nvvm

from ...typing import (
    Shape,
    IntTuple,
    Layout,
    Tensor,
    Int,
    Numeric,
    NumericMeta,
    Int16,
    Int32,
)
from ... import core
from .mma import SmemLayoutAtomKind, CtaGroup
from .copy import (
    Pack,
    Unpack,
    Ld16x64bOp,
    Ld16x128bOp,
    Ld16x256bOp,
    Ld16x32bx2Op,
    Ld32x32bOp,
    St16x64bOp,
    St16x128bOp,
    St16x256bOp,
    St16x32bx2Op,
    St32x32bOp,
)


####################################################################################################
#
# Helper functions for MMA
#
####################################################################################################


@dsl_user_op
def make_smem_layout_atom(
    kind: SmemLayoutAtomKind, element_type: Type[Numeric], *, loc=None, ip=None
) -> core.ComposedLayout:
    """
    Makes a SMEM layout Atom.

    This function creates a composed layout in unit of elements consistent with the requested layout
    Atom kind and element data type.

    :param kind:         The kind of layout Atom
    :type kind:          SmemLayoutAtomKind
    :param element_type: The element data type to construct the layout for
    :type element_type:  Type[Numeric]
    :return:             The SMEM layout atom
    :rtype:              core.ComposedLayout
    """
    if not isinstance(element_type, NumericMeta):
        raise TypeError(f"element_type must be a Numeric, but got {element_type}")

    if kind in (SmemLayoutAtomKind.MN_INTER, SmemLayoutAtomKind.K_INTER):
        num_contiguous_bits = 128
        sw = core.make_swizzle(0, 4, 3)
    elif kind in (SmemLayoutAtomKind.MN_SW32, SmemLayoutAtomKind.K_SW32):
        num_contiguous_bits = 256
        sw = core.make_swizzle(1, 4, 3)
    elif kind in (SmemLayoutAtomKind.MN_SW64, SmemLayoutAtomKind.K_SW64):
        num_contiguous_bits = 512
        sw = core.make_swizzle(2, 4, 3)
    elif kind in (SmemLayoutAtomKind.MN_SW128, SmemLayoutAtomKind.K_SW128):
        num_contiguous_bits = 1024
        sw = core.make_swizzle(3, 4, 3)
    elif kind == SmemLayoutAtomKind.MN_SW128_32B:
        num_contiguous_bits = 1024
        sw = core.make_swizzle(2, 5, 2)
    else:
        raise ValueError("unrecognized SMEM layout atom kind")
    num_contiguous_elems = num_contiguous_bits // element_type.width

    if kind in (
        SmemLayoutAtomKind.MN_INTER,
        SmemLayoutAtomKind.MN_SW32,
        SmemLayoutAtomKind.MN_SW64,
        SmemLayoutAtomKind.MN_SW128,
        SmemLayoutAtomKind.MN_SW128_32B,
    ):
        # M/N-major layout
        return core.make_composed_layout(
            sw,
            0,
            core.make_layout(
                (num_contiguous_elems, 8), stride=(1, num_contiguous_elems)
            ),
            loc=loc,
            ip=ip,
        )
    else:
        # K-major layout
        return core.make_composed_layout(
            sw,
            0,
            core.make_layout(
                (8, num_contiguous_elems), stride=(num_contiguous_elems, 1)
            ),
            loc=loc,
            ip=ip,
        )


@overload
def tile_to_mma_shape(
    atom: Layout, mma_tile_shape: Shape, order: IntTuple = None, *, loc=None, ip=None
) -> Layout: ...


@overload
def tile_to_mma_shape(
    atom: core.ComposedLayout,
    mma_tile_shape: Shape,
    order: IntTuple = None,
    *,
    loc=None,
    ip=None,
) -> core.ComposedLayout: ...


@dsl_user_op
def tile_to_mma_shape(
    atom, mma_tile_shape: Shape, order: IntTuple = None, *, loc=None, ip=None
):
    """
    Tiles a layout to an MMA shape.
    """
    # Default order is colexicographical
    if order is None:
        order = tuple(range(core.rank(mma_tile_shape) - 1))
    if core.rank(order) != core.rank(mma_tile_shape) - 1:
        raise ValueError(
            f"rank(order)={core.rank(order)} must be equal to "
            f"rank(mma_tile_shape)-1={core.rank(mma_tile_shape)-1}"
        )
    order_val = core._pack_int_tuple(order, loc=loc, ip=ip)
    mma_tile_shape_val = core._pack_shape(mma_tile_shape, loc=loc, ip=ip)

    if not (
        core.is_static(atom)
        and core.is_static(mma_tile_shape_val)
        and core.is_static(order_val)
    ):
        raise ValueError("tile_to_mma_shape only supports static inputs")

    res_ty = _cute_nvgpu_ir.tile_to_mma_shape(atom, mma_tile_shape_val, order_val)
    return _cute_ir.static(res_ty, loc=loc, ip=ip)


@dsl_user_op
def commit(
    mbar_ptr: core.Pointer,
    mask=None,
    cta_group: CtaGroup = CtaGroup.ONE,
    *,
    loc=None,
    ip=None,
) -> None:
    """
    Perform an arrive operation on a mbarrier upon completion of previous MMA operations.

    :param mbar_ptr: A pointer to the mbarrier in SMEM
    :type mbar_ptr:  Pointer
    :param mask:     An optional multicast mask for the CTAs in the cluster to signal arrival to
    :type mask:      Int
    """
    if cta_group == CtaGroup.ONE:
        group = nvvm.Tcgen05GroupKind.CTA_1
    else:
        assert cta_group == CtaGroup.TWO
        group = nvvm.Tcgen05GroupKind.CTA_2

    mbar_ptr = mbar_ptr.llvm_ptr
    if mask is not None:
        mask = Int16(mask).ir_value(loc=loc, ip=ip)
        nvvm.tcgen05_commit_arrive(
            mbar_ptr, multicast_mask=mask, group=group, loc=loc, ip=ip
        )
    else:
        nvvm.tcgen05_commit_arrive(mbar_ptr, group=group, loc=loc, ip=ip)
    return


####################################################################################################
#
# Helper functions for Copies
#
####################################################################################################


def is_tmem_load(atom: core.CopyAtom) -> bool:
    """
    Returns whether a CopyAtom instance is a TMEM load.
    """
    return isinstance(
        atom.op,
        (
            Ld16x64bOp,
            Ld16x128bOp,
            Ld16x256bOp,
            Ld16x32bx2Op,
            Ld32x32bOp,
        ),
    )


def is_tmem_store(atom: core.CopyAtom) -> bool:
    """
    Returns whether a CopyAtom instance is a TMEM store.
    """
    return isinstance(
        atom.op,
        (
            St16x64bOp,
            St16x128bOp,
            St16x256bOp,
            St16x32bx2Op,
            St32x32bOp,
        ),
    )


def get_tmem_copy_properties(
    atom: core.CopyAtom,
) -> Tuple[int, int, int, Union[Pack, Unpack]]:
    """
    Returns the properties of a TMEM copy atom (number of data paths, bits, repetitions,
    and whether packing/unpacking is used).
    """
    if isinstance(atom.op, (Ld16x64bOp, St16x64bOp)):
        num_dp, num_bits = 16, 64
    elif isinstance(atom.op, (Ld16x128bOp, St16x128bOp)):
        num_dp, num_bits = 16, 128
    elif isinstance(atom.op, (Ld16x256bOp, St16x256bOp)):
        num_dp, num_bits = 16, 256
    elif isinstance(atom.op, (Ld16x32bx2Op, St16x32bx2Op)):
        num_dp, num_bits = 16, 32
    elif isinstance(atom.op, (Ld32x32bOp, St32x32bOp)):
        num_dp, num_bits = 32, 32
    else:
        raise ValueError(f"expects 'atom' to be a TMEM copy, but got {atom}")
    if is_tmem_load(atom):
        return num_dp, num_bits, atom.op.repeat.value, atom.op.pack
    else:
        assert is_tmem_store(atom), "atom must be a TMEM store"
        return num_dp, num_bits, atom.op.repeat.value, atom.op.unpack


@dsl_user_op
def find_tmem_tensor_col_offset(tmem_tensor: Tensor, *, loc=None, ip=None) -> Int:
    """
    Computes the TMEM column offset given a TMEM tensor.

    :param tmem_tensor: The TMEM tensor to use to compute the columns offset
    :type tmem_tensor:  Tensor
    :return:            The columns offset
    :rtype:             Int
    """
    tmem_col_mask = 0x0000FFFF
    offset = (
        core.cosize(core.recast_tensor(tmem_tensor, Int32).layout, loc=loc, ip=ip)
        & tmem_col_mask
    )
    if isinstance(offset, int):
        return offset
    return Int32(offset, loc=loc, ip=ip)


@dsl_user_op
def make_tmem_copy(
    atom: core.CopyAtom, tmem_tensor: Tensor, *, loc=None, ip=None
) -> core.TiledCopy:
    """
    Makes a Tiled Copy instance from a TMEM Copy Atom and a TMEM tensor.
    """
    tiled_copy_val = _cute_nvgpu_ir.atom_make_tmem_copy(
        atom._trait.value, tmem_tensor.value, loc=loc, ip=ip
    )
    new_trait = type(atom._trait)(tiled_copy_val)
    return core.TiledCopy(atom.op, new_trait)


@dsl_user_op
def make_s2t_copy(
    atom: core.CopyAtom, tmem_tensor: Tensor, *, loc=None, ip=None
) -> core.TiledCopy:
    """
    Makes a Tiled Copy instance from a TMEM Copy Atom and a TMEM tensor.
    """
    tiled_copy_val = _cute_nvgpu_ir.atom_make_s2t_copy(
        atom._trait.value, tmem_tensor.value, loc=loc, ip=ip
    )
    new_trait = type(atom._trait)(tiled_copy_val)
    return core.TiledCopy(atom.op, new_trait)


@dsl_user_op
def get_s2t_smem_desc_tensor(
    atom: core.CopyAtom, smem_tensor: Tensor, *, loc=None, ip=None
) -> Tensor:
    """
    Returns the SMEM descriptor tensor from a S2T copy atom and a SMEM tensor.
    """
    smem_desc_tensor = _cute_nvgpu_ir.atom_get_copy_s2t_smem_desc_view(
        atom._trait.value, smem_tensor.value, loc=loc, ip=ip
    )
    return smem_desc_tensor
