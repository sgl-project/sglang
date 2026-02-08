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

"""
This module provides MLIR GPU Dialect helper functions
"""


from ..._mlir import ir
from ..._mlir.dialects import gpu, arith, scf
from ..._mlir.extras import types as T

from ..common import *

# =============================================================================
# GPU Dialect Helper functions
# =============================================================================


def create_async_token():
    token_ty = gpu.AsyncTokenType.get()
    token = gpu.wait(token_ty, [])
    return token


def printf(fmt, *args, threadNumber=-1):
    """Generate gpu.printf OP predicated on threadNumber"""
    type_formats = []
    for arg in args:
        ty_format = None
        if ir.IndexType.isinstance(arg.type):
            ty_format = "%llu"
        if ir.IntegerType.isinstance(arg.type):
            width = ir.IntegerType(arg.type).width
            if width == 64:
                ty_format = "%llu"
            elif width == 32:
                ty_format = "%d"
            elif width == 1:
                ty_format = "%i"
        if ir.F32Type.isinstance(arg.type):
            ty_format = "%f"
        if ty_format is None:
            raise DSLNotImplemented(arg.type)
        type_formats.append(ty_format)
    if threadNumber == -1:
        gpu.printf(fmt.format(*type_formats) + "\n", args)
    if threadNumber != -1:
        tidx = gpu.thread_id(gpu.Dimension.x)
        predicate = arith.cmpi(
            arith.CmpIPredicate.eq, tidx, arith.constant(_T.index(), threadNumber)
        )
        if_op = scf.IfOp(predicate)
        with ir.InsertionPoint(if_op.then_block):
            gpu.printf(fmt.format(*type_formats) + "\n", args)
            scf.yield_([])
