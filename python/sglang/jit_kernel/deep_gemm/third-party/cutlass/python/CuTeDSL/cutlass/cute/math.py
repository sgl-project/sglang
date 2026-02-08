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

from .core import TensorSSA
from .typing import Numeric
from cutlass._mlir.dialects import math, arith

from typing import Callable, Union


def _math_op(func: Callable, fastmath: bool, *args, **kwargs):
    """Dispatch the function to either a TensorSSA or a Numeric(Float).

    :param func: The function to dispatch
    :param args: The input tensor or scalar
    :param kwargs: The input tensor or scalar
    """
    arg_type = type(args[0])
    for arg in args:
        if not isinstance(arg, TensorSSA) and (
            not isinstance(arg, Numeric) or not type(arg).is_float
        ):
            raise TypeError(
                f"Expected a TensorSSA or Numeric(Float), but got {type(arg)}"
            )
        if not isinstance(arg, arg_type):
            raise TypeError(
                f"Expected all inputs to be of type {arg_type}, but got {type(arg)}"
            )

    fastmath_flag = arith.FastMathFlags.fast if fastmath else arith.FastMathFlags.none
    if isinstance(args[0], TensorSSA):
        return TensorSSA(
            func(*args, fastmath=fastmath_flag), args[0].shape, args[0].dtype
        )
    else:
        args = [a.ir_value() for a in args]
        return func(*args, fastmath=fastmath_flag)


def acos(
    a: Union[TensorSSA, Numeric], fastmath: bool = False
) -> Union[TensorSSA, Numeric]:
    """Compute element-wise arc cosine of the input tensor.

    :param a: Input tensor
    :type a: Union[TensorSSA, Numeric]
    :param fastmath: Enable fast math optimizations, defaults to False
    :type fastmath: bool, optional
    :return: Tensor containing the arc cosine of each element in input tensor
    :rtype: Union[TensorSSA, Numeric]

    Example:

    .. code-block::

        x = cute.make_fragment(layout)  # Create tensor
        y = x.load()  # Load values
        z = acos(y)  # Compute arc cosine
    """
    return _math_op(math.acos, fastmath, a)


def asin(
    a: Union[TensorSSA, Numeric], fastmath: bool = False
) -> Union[TensorSSA, Numeric]:
    """Compute element-wise arc sine of the input tensor.

    :param a: Input tensor
    :type a: Union[TensorSSA, Numeric]
    :param fastmath: Enable fast math optimizations, defaults to False
    :type fastmath: bool, optional
    :return: Tensor containing the arc sine of each element in input tensor
    :rtype: Union[TensorSSA, Numeric]

    Example:

    .. code-block::

        x = cute.make_fragment(layout)  # Create tensor
        y = x.load()  # Load values
        z = asin(y)  # Compute arc sine
    """
    return _math_op(math.asin, fastmath, a)


def atan(
    a: Union[TensorSSA, Numeric], fastmath: bool = False
) -> Union[TensorSSA, Numeric]:
    """Compute element-wise arc tangent of the input tensor.

    :param a: Input tensor
    :type a: Union[TensorSSA, Numeric]
    :param fastmath: Enable fast math optimizations, defaults to False
    :type fastmath: bool, optional
    :return: Tensor containing the arc tangent of each element in input tensor
    :rtype: Union[TensorSSA, Numeric]

    Example:

    .. code-block::

        x = cute.make_fragment(layout)  # Create tensor
        y = x.load()  # Load values
        z = atan(y)  # Compute arc tangent
    """
    raise NotImplementedError("atan is not implemented")
    return _math_op(math.atan, fastmath, a)


def atan2(
    a: Union[TensorSSA, Numeric], b: Union[TensorSSA, Numeric], fastmath: bool = False
) -> Union[TensorSSA, Numeric]:
    """Compute element-wise arc tangent of two tensors.

    Computes atan2(a, b) element-wise. The function atan2(a, b) is the angle in radians
    between the positive x-axis and the point given by the coordinates (b, a).

    :param a: First input tensor (y-coordinates)
    :type a: Union[TensorSSA, Numeric]
    :param b: Second input tensor (x-coordinates)
    :type b: Union[TensorSSA, Numeric]
    :param fastmath: Enable fast math optimizations, defaults to False
    :type fastmath: bool, optional
    :return: Tensor containing the arc tangent of a/b element-wise
    :rtype: Union[TensorSSA, Numeric]

    Example:

    .. code-block::

        y = cute.make_fragment(ptr1, layout).load()  # y coordinates
        x = cute.make_fragment(ptr2, layout).load()  # x coordinates
        theta = atan2(y, x)  # Compute angles
    """
    return _math_op(math.atan2, fastmath, a, b)


def cos(
    a: Union[TensorSSA, Numeric], fastmath: bool = False
) -> Union[TensorSSA, Numeric]:
    """Compute element-wise cosine of the input tensor.

    :param a: Input tensor (in radians)
    :type a: Union[TensorSSA, Numeric]
    :param fastmath: Enable fast math optimizations, defaults to False
    :type fastmath: bool, optional
    :return: Tensor containing the cosine of each element
    :rtype: Union[TensorSSA, Numeric]

    Example:

    .. code-block::

        x = cute.make_fragment(layout)  # Create tensor
        y = x.load()  # Load values
        z = cos(y)  # Compute cosine
    """
    return _math_op(math.cos, fastmath, a)


def erf(
    a: Union[TensorSSA, Numeric], fastmath: bool = False
) -> Union[TensorSSA, Numeric]:
    """Compute element-wise error function of the input tensor.

    The error function is defined as:
    erf(x) = 2/√π ∫[0 to x] exp(-t²) dt

    :param a: Input tensor
    :type a: Union[TensorSSA, Numeric]
    :param fastmath: Enable fast math optimizations, defaults to False
    :type fastmath: bool, optional
    :return: Tensor containing the error function value for each element
    :rtype: Union[TensorSSA, Numeric]

    Example:

    .. code-block::

        x = cute.make_fragment(layout)  # Create tensor
        y = x.load()  # Load values
        z = erf(y)  # Compute error function
    """
    return _math_op(math.erf, fastmath, a)


def exp(
    a: Union[TensorSSA, Numeric], fastmath: bool = False
) -> Union[TensorSSA, Numeric]:
    """Compute element-wise exponential of the input tensor.

    :param a: Input tensor
    :type a: Union[TensorSSA, Numeric]
    :param fastmath: Enable fast math optimizations, defaults to False
    :type fastmath: bool, optional
    :return: Tensor containing the exponential of each element
    :rtype: Union[TensorSSA, Numeric]

    Example:

    .. code-block::

        x = cute.make_fragment(layout)  # Create tensor
        y = x.load()  # Load values
        z = exp(y)  # Compute exponential
    """
    return _math_op(math.exp, fastmath, a)


def exp2(
    a: Union[TensorSSA, Numeric], fastmath: bool = False
) -> Union[TensorSSA, Numeric]:
    """Compute element-wise base-2 exponential of the input tensor.

    :param a: Input tensor
    :type a: Union[TensorSSA, Numeric]
    :param fastmath: Enable fast math optimizations, defaults to False
    :type fastmath: bool, optional
    :return: Tensor containing 2 raised to the power of each element
    :rtype: Union[TensorSSA, Numeric]

    Example:

    .. code-block::

        x = cute.make_fragment(layout)  # Create tensor
        y = x.load()  # Load values
        z = exp2(y)  # Compute 2^x
    """
    return _math_op(math.exp2, fastmath, a)


def log(
    a: Union[TensorSSA, Numeric], fastmath: bool = False
) -> Union[TensorSSA, Numeric]:
    """Compute element-wise natural logarithm of the input tensor.

    :param a: Input tensor
    :type a: Union[TensorSSA, Numeric]
    :param fastmath: Enable fast math optimizations, defaults to False
    :type fastmath: bool, optional
    :return: Tensor containing the natural logarithm of each element
    :rtype: Union[TensorSSA, Numeric]

    Example:

    .. code-block::

        x = cute.make_fragment(layout)  # Create tensor
        y = x.load()  # Load values
        z = log(y)  # Compute natural logarithm
    """
    return _math_op(math.log, fastmath, a)


def log2(
    a: Union[TensorSSA, Numeric], fastmath: bool = False
) -> Union[TensorSSA, Numeric]:
    """Compute element-wise base-2 logarithm of the input tensor.

    :param a: Input tensor
    :type a: Union[TensorSSA, Numeric]
    :param fastmath: Enable fast math optimizations, defaults to False
    :type fastmath: bool, optional
    :return: Tensor containing the base-2 logarithm of each element
    :rtype: Union[TensorSSA, Numeric]

    Example:

    .. code-block::

        x = cute.make_fragment(layout)  # Create tensor
        y = x.load()  # Load values
        z = log2(y)  # Compute log base 2
    """
    return _math_op(math.log2, fastmath, a)


def log10(
    a: Union[TensorSSA, Numeric], fastmath: bool = False
) -> Union[TensorSSA, Numeric]:
    """Compute element-wise base-10 logarithm of the input tensor.

    :param a: Input tensor
    :type a: Union[TensorSSA, Numeric]
    :param fastmath: Enable fast math optimizations, defaults to False
    :type fastmath: bool, optional
    :return: Tensor containing the base-10 logarithm of each element
    :rtype: Union[TensorSSA, Numeric]

    Example:

    .. code-block::

        x = cute.make_fragment(layout)  # Create tensor
        y = x.load()  # Load values
        z = log10(y)  # Compute log base 10
    """
    return _math_op(math.log10, fastmath, a)


def rsqrt(
    a: Union[TensorSSA, Numeric], fastmath: bool = False
) -> Union[TensorSSA, Numeric]:
    """Compute element-wise reciprocal square root of the input tensor.

    Computes 1/√x element-wise.

    :param a: Input tensor
    :type a: Union[TensorSSA, Numeric]
    :param fastmath: Enable fast math optimizations, defaults to False
    :type fastmath: bool, optional
    :return: Tensor containing the reciprocal square root of each element
    :rtype: Union[TensorSSA, Numeric]

    Example:

    .. code-block::

        x = cute.make_fragment(layout)  # Create tensor
        y = x.load()  # Load values
        z = rsqrt(y)  # Compute 1/√x
    """
    return _math_op(math.rsqrt, fastmath, a)


def sin(
    a: Union[TensorSSA, Numeric], fastmath: bool = False
) -> Union[TensorSSA, Numeric]:
    """Compute element-wise sine of the input tensor.

    :param a: Input tensor (in radians)
    :type a: Union[TensorSSA, Numeric]
    :param fastmath: Enable fast math optimizations, defaults to False
    :type fastmath: bool, optional
    :return: Tensor containing the sine of each element
    :rtype: Union[TensorSSA, Numeric]

    Example:

    .. code-block::

        x = cute.make_fragment(layout)  # Create tensor
        y = x.load()  # Load values
        z = sin(y)  # Compute sine
    """
    return _math_op(math.sin, fastmath, a)


def sqrt(
    a: Union[TensorSSA, Numeric], fastmath: bool = False
) -> Union[TensorSSA, Numeric]:
    """Compute element-wise square root of the input tensor.

    :param a: Input tensor
    :type a: Union[TensorSSA, Numeric]
    :param fastmath: Enable fast math optimizations, defaults to False
    :type fastmath: bool, optional
    :return: Tensor containing the square root of each element
    :rtype: Union[TensorSSA, Numeric]

    Example:

    .. code-block::

        x = cute.make_fragment(layout)  # Create tensor
        y = x.load()  # Load values
        z = sqrt(y)  # Compute square root
    """
    return _math_op(math.sqrt, fastmath, a)


def tan(
    a: Union[TensorSSA, Numeric], fastmath: bool = False
) -> Union[TensorSSA, Numeric]:
    """Compute element-wise tangent of the input tensor.

    :param a: Input tensor (in radians)
    :type a: Union[TensorSSA, Numeric]
    :param fastmath: Enable fast math optimizations, defaults to False
    :type fastmath: bool, optional
    :return: Tensor containing the tangent of each element
    :rtype: Union[TensorSSA, Numeric]

    Example:

    .. code-block::

        x = cute.make_fragment(layout)  # Create tensor
        y = x.load()  # Load values
        z = tan(y)  # Compute tangent
    """
    return _math_op(math.tan, fastmath, a)


def tanh(
    a: Union[TensorSSA, Numeric], fastmath: bool = False
) -> Union[TensorSSA, Numeric]:
    """Compute element-wise hyperbolic tangent of the input tensor.

    :param a: Input tensor
    :type a: Union[TensorSSA, Numeric]
    :param fastmath: Enable fast math optimizations, defaults to False
    :type fastmath: bool, optional
    :return: Tensor containing the hyperbolic tangent of each element
    :rtype: Union[TensorSSA, Numeric]

    Example:

    .. code-block::

        x = cute.make_fragment(layout)  # Create tensor
        y = x.load()  # Load values
        z = tanh(y)  # Compute hyperbolic tangent
    """
    return _math_op(math.tanh, fastmath, a)


__all__ = [
    "acos",
    "asin",
    "atan",
    "atan2",
    "cos",
    "erf",
    "exp",
    "exp2",
    "log",
    "log10",
    "log2",
    "rsqrt",
    "sin",
    "sqrt",
    "tan",
    "tanh",
]
