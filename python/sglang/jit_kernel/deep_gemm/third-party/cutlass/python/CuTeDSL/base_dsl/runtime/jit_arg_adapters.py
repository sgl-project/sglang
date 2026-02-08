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
This module provides runtime utilities for JIT argument conversion in DSL.
"""

from functools import wraps
from typing import get_origin

# Local modules imports
from ..common import DSLRuntimeError
from ..typing import (
    Constexpr,
    Int32,
    Float32,
    Boolean,
)


def is_arg_spec_constexpr(arg_spec, arg_name, arg_index, owning_func):
    """
    Check if the argument spec is a constexpr.
    """

    def _is_reserved_python_func_arg(arg_index, arg_name, func):
        """
        Check if the argument is a reserved python function argument.
        """

        if arg_index != 0:
            return False

        if arg_name == "self":
            return True

        is_classmethod = isinstance(func, classmethod) or (
            hasattr(func, "__func__") and isinstance(func.__func__, classmethod)
        )
        return arg_name == "cls" and is_classmethod

    return (
        _is_reserved_python_func_arg(arg_index, arg_name, owning_func)
        or (isinstance(arg_spec, type) and issubclass(arg_spec, Constexpr))
        or (get_origin(arg_spec) is Constexpr)
    )


def is_argument_constexpr(arg, arg_spec, arg_name, arg_index, owning_func):
    """
    Check if the argument is a constexpr.
    """

    def _is_type_argument(arg, arg_annotation):
        """
        Check if the argument is a type argument like Type[X]
        """

        return isinstance(arg, type) and (
            arg_annotation is None or get_origin(arg_annotation) is type
        )

    return (
        is_arg_spec_constexpr(arg_spec, arg_name, arg_index, owning_func)
        or _is_type_argument(arg, arg_spec)
        or arg is None
    )


class JitArgAdapterRegistry:
    """
    A registry to keep track of the JIT argument adapters.

    An adapter is a callable that converts a Python type to a type with following protocols supported:
    - JitArgument
    - DynamicExpression
    The converted type can then be further processed by DSL to generate arguments for JIT functions.
    """

    # A dictionary with key=type and value=callable
    jit_arg_adapter_registry = {}

    @classmethod
    def register_jit_arg_adapter(cls, *dargs, **dkwargs):
        """
        Register a JIT argument adapter callable

        This can be used as a decorator on any callable like:

        @register_jit_arg_adapter(my_py_type)
        def my_adapter_for_my_py_type(arg):
            ...

        @register_jit_arg_adapter(my_py_type)
        class MyAdapterForMyPythonType:
            ...

        The adapters are registered per type. If a type is already registerd, an error will be raised.
        """

        def decorator(*dargs, **dkwargs):
            darg_python_ty = dargs[0]

            @wraps(darg_python_ty)
            def wrapper(*args, **kwargs):
                if len(args) != 1 or not callable(args[0]):
                    raise DSLRuntimeError(
                        "a callable must be provided for registering JIT argument adapter"
                    )
                adapter = args[0]

                if darg_python_ty in cls.jit_arg_adapter_registry:
                    raise DSLRuntimeError(
                        f"JIT argument adapter for {darg_python_ty} is already registered!",
                        context={
                            "Registered adapter": cls.jit_arg_adapter_registry[
                                darg_python_ty
                            ],
                            "Adapter to be registered": adapter,
                        },
                    )
                cls.jit_arg_adapter_registry[darg_python_ty] = adapter
                return adapter

            return wrapper

        if len(dargs) > 0:
            return decorator(*dargs, **dkwargs)
        else:
            raise DSLRuntimeError(
                "a Python type must be provided for registering JIT argument adapter"
            )

    @classmethod
    def get_registered_adapter(cls, ty):
        """
        Get the registered JIT argument adapter for the given type.
        """
        return cls.jit_arg_adapter_registry.get(ty, None)


# =============================================================================
# JIT Argument Adapters
# =============================================================================


@JitArgAdapterRegistry.register_jit_arg_adapter(int)
@JitArgAdapterRegistry.register_jit_arg_adapter(float)
@JitArgAdapterRegistry.register_jit_arg_adapter(bool)
def _convert_python_scalar(arg):
    """
    Convert a Python scalar to a DSL type.
    """
    conversion_map = {
        int: Int32,
        float: Float32,
        bool: Boolean,
    }
    return conversion_map.get(type(arg))(arg)


@JitArgAdapterRegistry.register_jit_arg_adapter(tuple)
@JitArgAdapterRegistry.register_jit_arg_adapter(list)
def _convert_python_sequence(arg):
    """
    Go through each element in the sequence and convert it to a type that can be
    further processed by DSL to generate the corresponding JIT argument(s).
    """
    adapted_arg = []
    for elem in arg:
        adapter = JitArgAdapterRegistry.get_registered_adapter(type(elem))
        if adapter is not None:
            converted_elem = adapter(elem)
            adapted_arg.append(converted_elem)
        else:
            # If no registered adapter is found, just return the original element
            adapted_arg.append(elem)

    assert len(adapted_arg) == len(arg)
    return type(arg)(adapted_arg)
