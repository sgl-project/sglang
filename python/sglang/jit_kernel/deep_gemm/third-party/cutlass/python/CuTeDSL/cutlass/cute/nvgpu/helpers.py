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

from .. import core
from ..typing import Shape, Layout, Tensor, Numeric, NumericMeta
from ...impl_utils import check_type_in
from .cpasync.copy import (
    CopyBulkTensorTileG2SOp,
    CopyBulkTensorTileG2SNonExecTrait,
    CopyBulkTensorTileG2SMulticastOp,
    CopyBulkTensorTileG2SMulticastNonExecTrait,
)


####################################################################################################
#
# TMA creation helpers for tcgen05 MMAs
#
####################################################################################################


@dsl_user_op
def make_tiled_tma_atom_A(
    op: Union[CopyBulkTensorTileG2SOp, CopyBulkTensorTileG2SMulticastOp],
    gmem_tensor: Tensor,
    smem_layout: Union[Layout, core.ComposedLayout],
    mma_tiler_mnk: Shape,
    tiled_mma: core.TiledMma,
    cluster_shape_vmnk: Shape,
    *,
    internal_type: Optional[Type[Numeric]] = None,
    loc=None,
    ip=None,
) -> Tuple[core.CopyAtom, Tensor]:
    """
    Makes a TMA Copy atom mapping to ``.tile`` mode for ``cp.async.bulk.tensor`` PTX operation
    accounting for the MK projections of the TiledMMA for A tensor loads.

    Given

    - a GMEM tensor
    - a SMEM layout
    - a MMA Tiler
    - a TiledMma
    - a Cluster-level shape

    this function figures out the bulk tensor asynchronous copy instruction to use with the maximum
    "TMA vector length" to copy tiles of the GMEM tensor to an SMEM buffer with the provided
    layout and consistent with the provided Tiler & tiled_mma (considering the M-mode & K-mode).
    The Cluster-level shape is used to determine the multicast factor across the N-mode for A tensor loads.

    This function returns two results:

    1. the Copy Atom
    2. the so-called TMA tensor used to map logical coordinates of the GMEM tensor to coordinates
       that the TMA unit can consume. TMA tensors have so-called basis stride elements so that the
       associated layout can output coordinates. Otherwise, TMA tensors can be partitioned
       similarly to any other CuTe tensors using the algebra.

    :param op:                 The Copy Operation to construct an Atom for
    :type op:                  Union[CopyBulkTensorTileG2SOp, CopyBulkTensorTileG2SMulticastOp]
    :param gmem_tensor:        The GMEM tensor to be loaded by this copy atom
    :type gmem_tensor:         Tensor
    :param smem_layout:        Shared memory layout to load the tensor into (PDSL)
    :type smem_layout:         Union[Layout, core.ComposedLayout]
    :param mma_tiler_mnk:      The MMA Tiler shape (TILE_M, TILE_N, TILE_K) in MNK dimensions
    :type mma_tiler_mnk:       Shape
    :param tiled_mma:          The TiledMMA that will consume the load as operands
    :type tiled_mma:           core.TiledMma
    :param cluster_shape_vmnk: The Cluster-level shape in VMNK dimensions
    :type cluster_shape_vmnk:  Shape
    :param internal_type:      An optional parameter for the internal data type to when element
                               type does not match the copy type
    :type internal_type:       Type[Numeric]
    :return:                   A copy atom for this operation and the associated TMA coord tensor
    :rtype:                    Tuple[core.CopyAtom, Tensor]

    """

    if internal_type is not None:
        if not isinstance(internal_type, NumericMeta):
            raise TypeError(f"internal_type must be a Numeric, but got {internal_type}")
        internal_type = internal_type.mlir_type
    check_type_in(
        op,
        [CopyBulkTensorTileG2SOp, CopyBulkTensorTileG2SMulticastOp],
        "op",
        "make_tiled_tma_atom_A",
    )

    ident = core.make_identity_layout(gmem_tensor.shape, loc=loc, ip=ip)
    mma_tiler_mk = (mma_tiler_mnk[0], *mma_tiler_mnk[2:])
    g_tile = core.composition(ident, mma_tiler_mk, loc=loc, ip=ip)
    cta_v_map = tiled_mma._thrfrg_A(g_tile)
    cta_v_map = core.get(cta_v_map, mode=[1])
    cta_v_map = core.dice(cta_v_map, (1, (1,) * core.rank(g_tile)))

    if isinstance(op, CopyBulkTensorTileG2SOp):
        num_multicast = 1
    else:
        assert isinstance(op, CopyBulkTensorTileG2SMulticastOp)
        # multicast across the N-mode since those would share the same tile of A
        num_multicast = core.size(cluster_shape_vmnk, mode=[2])

    # res[0] = the IR Value for the non-executable atom instance
    # res[1] = the IR Value for the associated TMA tensor
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
    if isinstance(op, CopyBulkTensorTileG2SOp):
        return core.CopyAtom(op, CopyBulkTensorTileG2SNonExecTrait(res[0])), res[1]
    else:
        assert isinstance(op, CopyBulkTensorTileG2SMulticastOp)
        return (
            core.CopyAtom(op, CopyBulkTensorTileG2SMulticastNonExecTrait(res[0])),
            res[1],
        )


@dsl_user_op
def make_tiled_tma_atom_B(
    op: Union[CopyBulkTensorTileG2SOp, CopyBulkTensorTileG2SMulticastOp],
    gmem_tensor: Tensor,
    smem_layout: Union[Layout, core.ComposedLayout],
    mma_tiler_mnk: Shape,
    tiled_mma: core.TiledMma,
    cluster_shape_vmnk: Shape,
    *,
    internal_type: Optional[Type[Numeric]] = None,
    loc=None,
    ip=None,
) -> Tuple[core.CopyAtom, Tensor]:
    """
    Makes a TMA Copy atom mapping to ``.tile`` mode for ``cp.async.bulk.tensor`` PTX operation
    accounting for the NK projections of the TiledMMA for B tensor loads.

    Given

    - a GMEM tensor
    - a SMEM layout
    - a MMA Tiler
    - a TiledMma
    - a Cluster-level shape

    this function figures out the bulk tensor asynchronous copy instruction to use with the maximum
    "TMA vector length" to copy tiles of the GMEM tensor to an SMEM buffer with the provided
    layout and consistent with the provided Tiler & tiled_mma (considering the N-mode & K-mode).
    The Cluster-level shape is used to determine the multicast factor across the M-mode for B tensor loads.

    This function returns two results:

    1. the Copy Atom
    2. the so-called TMA tensor used to map logical coordinates of the GMEM tensor to coordinates
       that the TMA unit can consume. TMA tensors have so-called basis stride elements so that the
       associated layout can output coordinates. Otherwise, TMA tensors can be partitioned
       similarly to any other CuTe tensors using the algebra.

    :param op:                 The Copy Operation to construct an Atom for
    :type op:                  Union[CopyBulkTensorTileG2SOp, CopyBulkTensorTileG2SMulticastOp]
    :param gmem_tensor:        The GMEM tensor to be loaded by this copy atom
    :type gmem_tensor:         Tensor
    :param smem_layout:        Shared memory layout to load the tensor into (PDSL)
    :type smem_layout:         Union[Layout, core.ComposedLayout]
    :param mma_tiler_mnk:      The MMA Tiler shape (TILE_M, TILE_N, TILE_K) in MNK dimensions
    :type mma_tiler_mnk:       Shape
    :param tiled_mma:          The TiledMMA that will consume the load as operands
    :type tiled_mma:           core.TiledMma
    :param cluster_shape_vmnk: The Cluster-level shape in VMNK dimensions
    :type cluster_shape_vmnk:  Shape
    :param internal_type:      An optional parameter for the internal data type to when element
                               type does not match the copy type
    :type internal_type:       Type[Numeric]
    :return:                   A Copy Atom for this Operation and the associated TMA tensor
    :rtype:                    Tuple[core.CopyAtom, Tensor]

    """

    if internal_type is not None:
        if not isinstance(internal_type, NumericMeta):
            raise TypeError(f"internal_type must be a Numeric, but got {internal_type}")
        internal_type = internal_type.mlir_type
    check_type_in(
        op,
        [CopyBulkTensorTileG2SOp, CopyBulkTensorTileG2SMulticastOp],
        "op",
        "make_tiled_tma_atom_B",
    )

    ident = core.make_identity_layout(gmem_tensor.shape, loc=loc, ip=ip)
    mma_tiler_nk = (mma_tiler_mnk[1], *mma_tiler_mnk[2:])
    g_tile = core.composition(ident, mma_tiler_nk, loc=loc, ip=ip)
    cta_v_map = tiled_mma._thrfrg_B(g_tile)
    cta_v_map = core.get(cta_v_map, mode=[1])
    cta_v_map = core.dice(cta_v_map, (1, (1,) * core.rank(g_tile)))

    if isinstance(op, CopyBulkTensorTileG2SOp):
        num_multicast = 1
    else:
        assert isinstance(op, CopyBulkTensorTileG2SMulticastOp)
        # multicast across the M-mode since those would share the same tile of B
        num_multicast = core.size(cluster_shape_vmnk, mode=[1])

    # res[0] = the IR Value for the non-executable atom instance
    # res[1] = the IR Value for the associated TMA tensor
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
    if isinstance(op, CopyBulkTensorTileG2SOp):
        return core.CopyAtom(op, CopyBulkTensorTileG2SNonExecTrait(res[0])), res[1]
    else:
        assert isinstance(op, CopyBulkTensorTileG2SMulticastOp)
        return (
            core.CopyAtom(op, CopyBulkTensorTileG2SMulticastNonExecTrait(res[0])),
            res[1],
        )


__all__ = [
    "make_tiled_tma_atom_A",
    "make_tiled_tma_atom_B",
]
