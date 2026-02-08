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
This module provides @lru_cache_ir
It extends functools.lru_cache with IR Context awareness.

Example usage:
from cutlass import ir
from lru_cache_ir import lru_cache_ir

@lru_cache_ir(ir, maxsize=128, typed=False)
def make_layout(...):
...

"""


from functools import lru_cache, wraps

from ..._mlir import ir  # type: ignore


def get_ir_context(func):
    """
    Return the context for given func called under ir.
    Currently the context includes MLIRContext and InsertionPoint.
    """
    try:
        if ir:
            return (ir.Context.current, ir.InsertionPoint.current)
        else:
            return None
    except ValueError:
        return None


def lru_cache_ir(maxsize=128, typed=True):
    """
    Applies an LRU cache to a given function, with awareness of IR context.

    Usage is similar to functools.lru_cache while taking `ir` as required argument.

    :param ir: The IR object from which to derive the context by `get_ir_context`
    :param maxsize: Max cache size, same as functools.lru_cache
    :param typed: Whether params are type-sensitive, default to True as IR is type-sensitive
    """

    def decorator(func):
        # Use functools.lru_cache with a custom wrapper to control the key generation
        @lru_cache(maxsize=maxsize, typed=typed)
        def cached_func(context, *args, **kwargs):
            return func(*args, **kwargs)

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Call the cached function with the context
                return cached_func(get_ir_context(func), *args, **kwargs)
            except (RuntimeError, TypeError):
                return func(*args, **kwargs)

        # Expose cache-related methods for introspection
        wrapper.cache_clear = cached_func.cache_clear
        wrapper.cache_info = cached_func.cache_info
        return wrapper

    return decorator
