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

from typing import Type

from cutlass.cutlass_dsl import dsl_user_op

from cutlass._mlir.dialects import nvvm

from ...typing import Numeric, NumericMeta
from ... import core
from .mma import SmemLayoutAtomKind


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
    else:
        raise ValueError("unrecognized SMEM layout atom kind")
    num_contiguous_elems = num_contiguous_bits // element_type.width

    if kind in (
        SmemLayoutAtomKind.MN_INTER,
        SmemLayoutAtomKind.MN_SW32,
        SmemLayoutAtomKind.MN_SW64,
        SmemLayoutAtomKind.MN_SW128,
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


@dsl_user_op
def fence(*, loc=None, ip=None) -> None:
    """
    See the `PTX documentation <https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-multiply-and-accumulate-instruction-wgmma-fence>`__.
    """
    nvvm.wgmma_fence_aligned(loc=None, ip=None)


@dsl_user_op
def commit_group(*, loc=None, ip=None) -> None:
    """
    See the `PTX documentation <https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-commit-group>`__.
    """
    nvvm.wgmma_commit_group_sync_aligned(loc=loc, ip=ip)


@dsl_user_op
def wait_group(group, *, loc=None, ip=None) -> None:
    """
    See the `PTX documentation <https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-multiply-and-accumulate-instruction-wgmma-wait-group>`__.
    """
    nvvm.wgmma_wait_group_sync_aligned(group, loc=loc, ip=ip)
