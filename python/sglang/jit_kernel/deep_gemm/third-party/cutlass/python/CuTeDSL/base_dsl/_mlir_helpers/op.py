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
This module provides MLIR's OP helper functions
"""


import inspect
from functools import wraps

from ..._mlir import ir


def dsl_user_op(opFunc):
    @wraps(opFunc)
    def wrapper(*args, **kwargs):
        loc = kwargs.pop("loc", None)
        if loc is None:
            frame = inspect.currentframe().f_back
            file_loc = ir.Location.file(frame.f_code.co_filename, frame.f_lineno, 0)
            loc = ir.Location.name(frame.f_code.co_name, childLoc=file_loc)
        res_or_list = opFunc(*args, **kwargs, loc=loc)
        return res_or_list

    return wrapper
