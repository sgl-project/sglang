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
This module provides a timing helper functions
"""
from functools import wraps

from .logger import log


# TODO: revisit this part when mlir timing manager is ready for pybind.
def timer(*dargs, **kwargs):
    enable = kwargs.get("enable", True)

    def decorator(func):
        @wraps(func)
        def func_wrapper(*args, **kwargs):
            if not enable:
                return func(*args, **kwargs)
            from time import time

            start = time()
            result = func(*args, **kwargs)
            end = time()

            # Convert time from seconds to us
            spend_us = (end - start) * 1e6

            # Determine the function type and format the log message
            if hasattr(func, "__name__"):
                func_name = func.__name__
                log_message = f"[JIT-TIMER] Function: {func_name} | Execution Time: {spend_us:.2f} µs"
            elif "CFunctionType" in str(type(func)):
                log_message = f"[JIT-TIMER] C API Function: {str(func)} | Execution Time: {spend_us:.2f} µs"
            else:
                log_message = f"[JIT-TIMER] Anonymous Function | Execution Time: {spend_us:.2f} µs"

            log().info(log_message)

            return result

        return func_wrapper

    if len(dargs) == 1 and callable(dargs[0]):
        return decorator(dargs[0])
    else:
        return decorator
