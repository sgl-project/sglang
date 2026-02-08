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

from dataclasses import dataclass, field
from typing import Union

from cutlass.cutlass_dsl import dsl_user_op

import cutlass.cute as cute
from cutlass.cute.nvgpu.tcgen05 import OperandMajorMode

import cutlass._mlir.dialects.cute as _cute_ir
import cutlass._mlir.dialects.cute_nvgpu as _cute_nvgpu_ir


@dataclass(frozen=True)
class BlockScaledBasicChunk:
    """
    The basic scale factor atom layout decided by tcgen05 BlockScaled MMA Ops.

    This class represents the fixed layout pattern for scale factors used in
    tcgen05 BlockScaled MMA Ops. The layout is determined by the
    instruction specification and cannot be modified.
    See `PTX documentation <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x>`.
    """

    sf_vec_size: int
    major_mode: OperandMajorMode = OperandMajorMode.K
    _layout: cute.Layout = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.major_mode == OperandMajorMode.K:
            # K-major layout: (AtomMN, AtomK)
            atom_shape = ((32, 4), (self.sf_vec_size, 4))
            atom_stride = ((16, 4), (0, 1))
        else:
            # MN-major layout: (AtomK, AtomMN)
            atom_shape = ((self.sf_vec_size, 4), (32, 4))
            atom_stride = ((0, 1), (16, 4))

        object.__setattr__(
            self, "_layout", cute.make_layout(atom_shape, stride=atom_stride)
        )

    @property
    def layout(self) -> cute.Layout:
        """
        Get the layout for this block scaled chunk.

        :return: The layout representing the scale factor atom
        :rtype: cute.Layout
        """
        return self._layout


@dsl_user_op
def tile_atom_to_shape_SF(
    Shape: cute.Shape,
    sf_vec_size: int,
    *,
    loc=None,
    ip=None,
) -> cute.Layout:
    """
    A helper function to get dynamic SFA/SFB layout by filling dynamic A/B shape to the scale factor atom layout.

    :param Shape: The shape of the A/B tensor
    :param sf_vec_size: Scale factor vector size

    :return: The layout of the SFA/SFB tensor
    :rtype: cute.Layout
    """
    # ((Atom_MN, Rest_MN),(Atom_K, Rest_K),RestL)
    sf_layout = cute.tile_to_shape(
        BlockScaledBasicChunk(sf_vec_size).layout, Shape, (2, 1, 3)
    )
    return sf_layout


@dsl_user_op
def make_smem_layout_sfa(
    tiled_mma: cute.TiledMma,
    mma_tiler_mnk: cute.Tile,
    sf_vec_size: int,
    num_stages: int,
    *,
    loc=None,
    ip=None,
) -> cute.Layout:
    """
    Make smem layout for SFA based on:
    1. BlockScaledBasicChunk
    2. MMA tiler shape
    3. Scale factor vector size
    4. Number of stages

    :param tiled_mma: The tiled MMA
    :type tiled_mma: cute.TiledMma
    :param mma_tiler_mnk: The mma tiler shape
    :type mma_tiler_mnk: cute.Tile
    :param sf_vec_size: The scale factor vector size
    :type sf_vec_size: int
    :param num_stages: The number of stages
    :type num_stages: int

    :return: Smem layout for SFA
    :rtype: cute.Layout
    """
    # (CTA_Tile_Shape_M, MMA_Tile_Shape_K)
    sfa_tile_shape = (
        mma_tiler_mnk[0] // cute.size(tiled_mma.thr_id.shape),
        mma_tiler_mnk[2],
    )

    # ((Atom_M, Rest_M),(Atom_K, Rest_K))
    smem_layout = cute.tile_to_shape(
        BlockScaledBasicChunk(sf_vec_size).layout,
        sfa_tile_shape,
        (2, 1),
    )

    mma_tile_inst_k = 4
    # (CTA_Tile_Shape_M, MMA_Inst_Shape_K)
    sfa_tile_shape = cute.shape_div(sfa_tile_shape, (1, mma_tile_inst_k))
    # ((Atom_Inst_M, Atom_Inst_K), MMA_M, MMA_K))
    smem_layout = cute.tiled_divide(smem_layout, sfa_tile_shape)

    atom_m = 128
    tiler_inst = ((atom_m, sf_vec_size),)
    # (((Atom_Inst_M, Rest_M),(Atom_Inst_K, Rest_K)), MMA_M, MMA_K)
    smem_layout = cute.logical_divide(smem_layout, tiler_inst)

    # (((Atom_Inst_M, Rest_M),(Atom_Inst_K, Rest_K)), MMA_M, MMA_K, STAGE)
    sfa_smem_layout_staged = cute.append(
        smem_layout,
        cute.make_layout(
            num_stages, stride=cute.cosize(cute.filter_zeros(smem_layout))
        ),
    )

    return sfa_smem_layout_staged


@dsl_user_op
def make_smem_layout_sfb(
    tiled_mma: cute.TiledMma,
    mma_tiler_mnk: cute.Tile,
    sf_vec_size: int,
    num_stages: int,
    *,
    loc=None,
    ip=None,
) -> cute.Layout:
    """
    Make smem layout for SFB based on:
    1. BlockScaledBasicChunk
    2. MMA tiler shape
    3. Scale factor vector size
    4. Number of stages

    :param tiled_mma: The tiled MMA
    :type tiled_mma: cute.TiledMma
    :param mma_tiler_mnk: The mma tiler shape
    :type mma_tiler_mnk: cute.Tile
    :param sf_vec_size: The scale factor vector size
    :type sf_vec_size: int
    :param num_stages: The number of stages
    :type num_stages: int

    :return: Smem layout for SFA
    :rtype: cute.Layout
    """
    # (Round_Up(CTA_Tile_Shape_N, 128), MMA_Tile_Shape_K)
    sfb_tile_shape = (
        cute.round_up(mma_tiler_mnk[1], 128),
        mma_tiler_mnk[2],
    )

    # ((Atom_N, Rest_N),(Atom_K, Rest_K))
    smem_layout = cute.tile_to_shape(
        BlockScaledBasicChunk(sf_vec_size).layout,
        sfb_tile_shape,
        (2, 1),
    )

    mma_tile_inst_k = 4
    # (CTA_Tile_Shape_N, MMA_Inst_Shape_K)
    sfb_tile_shape = cute.shape_div(sfb_tile_shape, (1, mma_tile_inst_k))
    # ((Atom_Inst_N, Atom_Inst_K), MMA_N, MMA_K)
    smem_layout = cute.tiled_divide(smem_layout, sfb_tile_shape)

    atom_n = 128
    tiler_inst = ((atom_n, sf_vec_size),)
    # (((Atom_Inst_M, Rest_M),(Atom_Inst_K, Rest_K)), MMA_M, MMA_K)
    smem_layout = cute.logical_divide(smem_layout, tiler_inst)

    # (((Atom_Inst_M, Rest_M),(Atom_Inst_K, Rest_K)), MMA_M, MMA_K, STAGE)
    sfb_smem_layout_staged = cute.append(
        smem_layout,
        cute.make_layout(
            num_stages, stride=cute.cosize(cute.filter_zeros(smem_layout))
        ),
    )

    return sfb_smem_layout_staged


@dsl_user_op
def make_tmem_layout_sfa(
    tiled_mma: cute.TiledMma,
    mma_tiler_mnk: cute.Tile,
    sf_vec_size: int,
    smem_layout: cute.Layout,
    *,
    loc=None,
    ip=None,
) -> cute.Layout:
    """Make tmem layout for SFA based on:
    1. SFA smem layout per stage
    2. Cta tile shape m
    3. tiled MMA atom thr size
    4. Scale factor vector size

    :param tiled_mma: The tiled MMA
    :type tiled_mma: cute.TiledMma
    :param mma_tiler_mnk: The mma tiler shape
    :type mma_tiler_mnk: cute.Tile
    :param sf_vec_size: The scale factor vector size
    :type sf_vec_size: int
    :param smem_layout: The smem layout of SFA per stage
    :type smem_layout: cute.Layout

    :return: TMEM layout for SFA
    :rtype: cute.Layout
    """
    atom_thr_size = cute.size(tiled_mma.thr_id.shape)
    cta_tile_shape_m = mma_tiler_mnk[0] // atom_thr_size

    sfa_layout_ty = _cute_nvgpu_ir.make_tmem_layout_sfa(
        smem_layout, cta_tile_shape_m, atom_thr_size, sf_vec_size
    )
    return _cute_ir.static(sfa_layout_ty, loc=loc, ip=ip)


@dsl_user_op
def make_tmem_layout_sfb(
    tiled_mma: cute.TiledMma,
    mma_tiler_mnk: cute.Tile,
    sf_vec_size: int,
    smem_layout: cute.Layout,
    *,
    loc=None,
    ip=None,
) -> cute.Layout:
    """Make tmem layout for SFB based on:
    1. SFB smem layout per stage
    2. Cta tile shape m
    3. tiled MMA atom thr size
    4. Scale factor vector size

    :param tiled_mma: The tiled MMA
    :type tiled_mma: cute.TiledMma
    :param mma_tiler_mnk: The mma tiler shape
    :type mma_tiler_mnk: cute.Tile
    :param sf_vec_size: The scale factor vector size
    :type sf_vec_size: int
    :param smem_layout: The smem layout of SFB per stage
    :type smem_layout: cute.Layout

    :return: TMEM layout for SFB
    :rtype: cute.Layout
    """
    atom_thr_size = cute.size(tiled_mma.thr_id.shape)
    cta_tile_shape_m = mma_tiler_mnk[0] // atom_thr_size

    sfb_layout_ty = _cute_nvgpu_ir.make_tmem_layout_sfb(
        smem_layout, cta_tile_shape_m, atom_thr_size, sf_vec_size
    )
    return _cute_ir.static(sfb_layout_ty, loc=loc, ip=ip)
