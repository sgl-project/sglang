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

from typing import Optional, Tuple, Type, Union

from cutlass.cutlass_dsl import dsl_user_op

import cutlass._mlir.dialects.cute_nvgpu as _cute_nvgpu_ir
from cutlass._mlir.dialects import llvm

from ...typing import Coord, Layout, Tensor, Tiler, Pointer, Int16, Numeric, NumericMeta
from ... import core
from .copy import (
    CopyBulkTensorTileG2SOp,
    CopyBulkTensorTileG2SMulticastOp,
    CopyBulkTensorTileS2GOp,
    CopyReduceBulkTensorTileS2GOp,
    CopyBulkTensorTileG2SNonExecTrait,
    CopyBulkTensorTileG2SMulticastNonExecTrait,
    CopyBulkTensorTileS2GTrait,
    CopyReduceBulkTensorTileS2GTrait,
)


@dsl_user_op
def make_tiled_tma_atom(
    op: Union[
        CopyBulkTensorTileG2SOp,
        CopyBulkTensorTileG2SMulticastOp,
        CopyBulkTensorTileS2GOp,
        CopyReduceBulkTensorTileS2GOp,
    ],
    gmem_tensor: Tensor,
    smem_layout: Union[Layout, core.ComposedLayout],
    cta_tiler: Tiler,
    num_multicast: int = 1,
    *,
    internal_type: Optional[Type[Numeric]] = None,
    loc=None,
    ip=None,
) -> Tuple[core.CopyAtom, Tensor]:
    """
    Makes a TMA Copy Atom in the ``.tile`` mode to copy tiles of a GMEM tensor to/from SMEM
    buffer with the given Layout.

    Given

    - a GMEM tensor
    - a SMEM layout
    - a CTA-level Tiler

    this function figures out the bulk tensor asynchronous copy instruction to use with the maximum
    "TMA vector length" to copy tiles of the GMEM tensor to/from an SMEM buffer with the provided
    layout and consistent with the provided Tiler.

    This function returns two results:

    1. the Copy Atom
    2. the so-called TMA tensor used to map logical coordinates of the GMEM tensor to coordinates \
       that the TMA unit can consume. TMA tensors have so-called basis stride elements so that the \
       associated layout can output coordinates. Otherwise, TMA tensors can be partitioned \
       similarly to any other CuTe tensors using the algebra.

    :param op:            The Copy Operation to construct an Atom for
    :type op:             Union[CopyBulkTensorTileG2SOp, CopyBulkTensorTileG2SMulticastOp, CopyBulkTensorTileS2GOp, CopyReduceBulkTensorTileS2GOp]
    :param gmem_tensor:   The GMEM tensor involved in the Copy
    :type gmem_tensor:    Tensor
    :param smem_layout:   The SMEM layout to construct the Copy Atom for
    :type smem_layout:    Union[Layout, core.ComposedLayout]
    :param cta_tiler:     The CTA Tiler to use
    :type cta_tiler:      Tiler
    :param num_multicast: The multicast factor
    :type num_multicast:  int
    :param internal_type: An optional parameter for the internal data type to use when the actual data type is not supported by the TMA unit
    :type internal_type:  Type[Numeric]
    :return:              A Copy Atom for this Operation and the associated TMA tensor
    :rtype:               Tuple[core.CopyAtom, Tensor]
    """

    if internal_type is not None:
        if not isinstance(internal_type, NumericMeta):
            raise TypeError(f"internal_type must be a Numeric, but got {internal_type}")
        internal_type = internal_type.mlir_type

    cta_v_map = core.composition(
        core.make_identity_layout(gmem_tensor.shape, loc=loc, ip=ip),
        cta_tiler,
        loc=loc,
        ip=ip,
    )

    if isinstance(op, CopyBulkTensorTileG2SOp):
        if num_multicast != 1:
            raise ValueError(
                f"expects num_multicast to be 1 for non multicast G2S copies, "
                f"but got {num_multicast}"
            )
        res = _cute_nvgpu_ir.atom_make_non_exec_tiled_tma_load(
            gmem_tensor.value,
            smem_layout,
            cta_v_map,
            op._to_ir(),
            num_multicast=num_multicast,
            internal_type=internal_type,
            loc=loc,
            ip=ip,
        )
        return core.CopyAtom(op, CopyBulkTensorTileG2SNonExecTrait(res[0])), res[1]
    elif isinstance(op, CopyBulkTensorTileG2SMulticastOp):
        if num_multicast < 1:
            raise ValueError(
                f"expects num_multicast to be >= 1 for multicast G2S copies, "
                f"but got {num_multicast}"
            )
        res = _cute_nvgpu_ir.atom_make_non_exec_tiled_tma_load(
            gmem_tensor.value,
            smem_layout,
            cta_v_map,
            op._to_ir(),
            num_multicast=num_multicast,
            internal_type=internal_type,
            loc=loc,
            ip=ip,
        )
        return (
            core.CopyAtom(op, CopyBulkTensorTileG2SMulticastNonExecTrait(res[0])),
            res[1],
        )
    elif isinstance(op, CopyBulkTensorTileS2GOp):
        res = _cute_nvgpu_ir.atom_make_non_exec_tiled_tma_store(
            gmem_tensor.value,
            smem_layout,
            cta_v_map,
            internal_type=internal_type,
            loc=loc,
            ip=ip,
        )
        return core.CopyAtom(op, CopyBulkTensorTileS2GTrait(res[0])), res[1]
    elif isinstance(op, CopyReduceBulkTensorTileS2GOp):
        res = _cute_nvgpu_ir.atom_make_non_exec_tiled_tma_reduce(
            gmem_tensor.value,
            smem_layout,
            cta_v_map,
            op._to_ir(),
            internal_type=internal_type,
            loc=loc,
            ip=ip,
        )
        return core.CopyAtom(op, CopyReduceBulkTensorTileS2GTrait(res[0])), res[1]
    else:
        raise ValueError(f"expects a bulk tensor (TMA) Copy Op, but got {op}")


@dsl_user_op
def tma_partition(
    atom: core.CopyAtom,
    cta_coord: Coord,
    cta_layout: Layout,
    smem_tensor: Tensor,
    gmem_tensor: Tensor,
    *,
    loc=None,
    ip=None,
) -> Tuple[Tensor, Tensor]:
    """
    Tiles the GMEM and SMEM tensors for the provided TMA Copy Atom.
    """
    cta_coord_val = core._pack_coord(cta_coord, loc=loc, ip=ip)
    s, d = _cute_nvgpu_ir.atom_tma_partition(
        atom._trait.value,
        cta_coord=cta_coord_val,
        cta_layout=cta_layout,
        smem_tensor=smem_tensor.value,
        gmem_tensor=gmem_tensor.value,
        loc=loc,
        ip=ip,
    )
    return s, d


@dsl_user_op
def create_tma_multicast_mask(
    cta_layout_vmnk: Layout,
    cta_coord_vmnk: Coord,
    mcast_mode: int,
    *,
    loc=None,
    ip=None,
) -> Int16:
    """
    Computes a multicast mask for a TMA load Copy.

    :param cta_layout_vmnk: The VMNK layout of the cluster
    :type cta_layout_vmnk:  Layout
    :param cta_coord_vmnk:  The VMNK coordinate of the current CTA
    :type cta_coord_vmnk:   Coord
    :param mcast_mode:      The tensor mode in which to multicast
    :type mcast_mode:       int
    :return:                The resulting mask
    :rtype:                 Int16
    """
    if core.rank(cta_layout_vmnk) != 4:
        raise ValueError(
            f"cta_layout_vmnk must be rank 4, but got {core.pretty_str(cta_layout_vmnk)}"
        )
    if core.rank(cta_coord_vmnk) != 4:
        raise ValueError(
            f"cta_coord_vmnk must be rank 4, but got {core.pretty_str(cta_coord_vmnk)}"
        )
    return core.make_layout_image_mask(
        cta_layout_vmnk, cta_coord_vmnk, mcast_mode, loc=loc, ip=ip
    )


@dsl_user_op
def prefetch_descriptor(tma_atom: core.CopyAtom, *, loc=None, ip=None) -> None:
    """
    Prefetches the TMA descriptor associated with the TMA Atom.
    """
    _cute_nvgpu_ir.prefetch_tma_desc(tma_atom._trait.value, loc=loc, ip=ip)


@dsl_user_op
def copy_tensormap(
    tma_atom: core.CopyAtom, tensormap_ptr: Pointer, *, loc=None, ip=None
) -> None:
    """
    Copies the tensormap held by a TMA Copy Atom to the memory location pointed to by the provided
    pointer.

    :param tma_atom:      The TMA Copy Atom
    :type tma_atom:       CopyAtom
    :param tensormap_ptr: The pointer to the memory location to copy the tensormap to
    :type tensormap_ptr:  Pointer
    """
    _cute_nvgpu_ir.copy_tma_desc(
        tma_atom._trait.value, tensormap_ptr.value, loc=loc, ip=ip
    )


@dsl_user_op
def update_tma_descriptor(
    tma_atom: core.CopyAtom,
    gmem_tensor: Tensor,
    tma_desc_ptr: Pointer,
    *,
    loc=None,
    ip=None,
) -> None:
    """
    Updates the TMA descriptor in the memory location pointed to by the provided pointer using
    information from a TMA Copy Atom and the provided GMEM tensor.

    Specifically, the following fields of the TMA descriptor will be updated:

    1. the GMEM tensor base address
    2. the GMEM tensor shape
    3. the GMEM tensor stride

    Other fields of the TMA descriptor are left unchanged.

    :param tma_atom:      The TMA Copy Atom
    :type tma_atom:       CopyAtom
    :param gmem_tensor:   The GMEM tensor
    :type gmem_tensor:    Tensor
    :param tensormap_ptr: The pointer to the memory location of the descriptor to udpate
    :type tensormap_ptr:  Pointer
    """
    _cute_nvgpu_ir.update_tma_desc(
        tma_atom._trait.value, gmem_tensor.value, tma_desc_ptr.value, loc=loc, ip=ip
    )


@dsl_user_op
def fence_tma_desc_acquire(
    tma_desc_ptr: Pointer,
    *,
    loc=None,
    ip=None,
) -> None:
    """
    See the `PTX documentation <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar>`__.
    """
    tma_desc_ptr_i64 = tma_desc_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [tma_desc_ptr_i64],
        "fence.proxy.tensormap::generic.acquire.gpu [$0], 128;",
        "l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def cp_fence_tma_desc_release(
    tma_desc_global_ptr: Pointer,
    tma_desc_shared_ptr: Pointer,
    *,
    loc=None,
    ip=None,
) -> None:
    """
    See the `PTX documentation <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-tensormap-cp-fenceproxy>`__.
    """
    tma_desc_global_ptr_i64 = tma_desc_global_ptr.toint(loc=loc, ip=ip).ir_value()
    tma_desc_shared_ptr_i32 = tma_desc_shared_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [tma_desc_global_ptr_i64, tma_desc_shared_ptr_i32],
        "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [$0], [$1], 128;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def fence_tma_desc_release(*, loc=None, ip=None) -> None:
    """
    See the `PTX documentation <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar>`__.
    """
    llvm.inline_asm(
        None,
        [],
        "fence.proxy.tensormap::generic.release.gpu;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
