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

from cutlass.cutlass_dsl import CuTeDSL, T, dsl_user_op

import cutlass._mlir.dialects.cute_nvgpu as _cute_nvgpu_ir
from cutlass._mlir.dialects import nvvm, scf
from cutlass._mlir import ir

from ..typing import Int, Int32
from ...impl_utils import check_value_in


@dsl_user_op
def make_warp_uniform(value: Int, *, loc=None, ip=None) -> Int32:
    """
    Creates a warp-uniform value from the given integer input.

    :param value: The integer to make warp uniform.
    :type value:  Int
    :return:      The warp-uniform value equal to the input.
    :rtype:       Int32
    """
    return Int32(
        _cute_nvgpu_ir.arch_make_warp_uniform(
            Int32(value).ir_value(loc=loc, ip=ip), loc=loc, ip=ip
        )
    )


class IfOpRegion:
    """
    A context manager for if Op.
    Automatically inserts `scf.yield([])` when exiting the context.
    """

    def __init__(self, block, *, loc=None, ip=None):
        self.block = block
        self.insert_point = ir.InsertionPoint(self.block)
        self.loc = loc
        self.ip = ip

    def __enter__(self):
        self.insert_point.__enter__()
        return self.block.arguments

    def __exit__(self, exc_type, exc_value, traceback):
        scf.yield_([], loc=self.loc, ip=self.ip)
        self.insert_point.__exit__(exc_type, exc_value, traceback)


@dsl_user_op
def elect_one(*, loc=None, ip=None) -> IfOpRegion:
    """
    Elects one thread within a warp.

    .. code-block:: python

        with elect_one():
            # Only one thread in the warp executes the code in this context
            pass
    """
    arch = CuTeDSL._get_dsl().envar.arch
    check_value_in(
        arch,
        [
            "sm_90",
            "sm_90a",
            "sm_100a",
            "sm_100f",
        ],
        "arch",
    )
    is_thread_leader = nvvm.elect_sync(T.bool())
    if_op = scf.IfOp(is_thread_leader, loc=loc, ip=ip)
    return IfOpRegion(if_op.then_block, loc=loc, ip=ip)
