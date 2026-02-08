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

import copy as py_copy
from dataclasses import dataclass
import inspect
import math
import operator
from abc import ABC, abstractmethod
from functools import lru_cache, partial, reduce
from inspect import isclass
from itertools import chain
from typing import (
    Callable,
    Iterable,
    overload,
    List,
    Tuple,
    Union,
    Type,
    Any,
    Dict,
    Optional,
)
from enum import Enum, auto

from cutlass.cutlass_dsl import (
    const,
    T,
    lru_cache_ir,
    is_dynamic_expression,
    for_generate,
    yield_out,
    if_generate,
    extract_mlir_values,
    new_from_mlir_values,
    _binary_op_type_promote,
    not_,
    cutlass_arith,
    dsl_user_op,
)

from cutlass._mlir import ir
from cutlass._mlir.dialects._ods_common import get_op_result_or_op_results
from cutlass._mlir.dialects import cute as _cute_ir
from cutlass._mlir.dialects.cute import (
    ScaledBasis as _ScaledBasis,
    Ratio as _Ratio,
)

from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir
from cutlass._mlir.dialects import llvm, builtin, vector, arith

from .typing import (
    Numeric,
    Integer,
    NumericMeta,
    Boolean,
    Int32,
    Int8,
    Int16,
    Int32,
    Int64,
    Float32,
    TFloat32,
    Int,
    IntTuple,
    Shape,
    Stride,
    Coord,
    Layout,
    Tile,
    Tiler,
    XTuple,
    Tensor,
    Pointer,
    AddressSpace,
    as_numeric,
)


####################################################################################################
#
# Internal IntTuple helpers
#
####################################################################################################


def _get_typed_value(x):
    if isinstance(x, Integer):
        return (
            x.value.get_typed_value() if isinstance(x.value, IntValue) else x.ir_value()
        )
    else:
        return x


def _pack_x(x, packer, op, *, loc=None, ip=None) -> ir.Value:
    x = transform_leaf(_get_typed_value, x)
    res_ty, dyn_elems = packer(x)
    # <"0"> is deduced from type inference which should be removed for make_... operations
    dyn_elems = [t for t in dyn_elems if not is_static(t)]
    return op(res_ty, dyn_elems, loc=loc, ip=ip).result


def _pack_shape(shape: Shape, *, loc=None, ip=None) -> ir.Value:
    _check_shape(shape)
    return _pack_x(shape, _cute_ir.pack_shape, _cute_ir.MakeShapeOp, loc=loc, ip=ip)


def _pack_stride(stride: Stride, *, loc=None, ip=None) -> ir.Value:
    _check_stride(stride)
    # Convert basis elements to the base class before _pack_x
    stride = transform_leaf(
        lambda x: x.to(_cute_ir.ScaledBasis) if isinstance(x, ScaledBasis) else x,
        stride,
    )
    return _pack_x(stride, _cute_ir.pack_stride, _cute_ir.MakeStrideOp, loc=loc, ip=ip)


def _pack_coord(coord: Coord, *, loc=None, ip=None) -> ir.Value:
    _check_coord(coord)
    return _pack_x(coord, _cute_ir.pack_coord, _cute_ir.MakeCoordOp, loc=loc, ip=ip)


def _pack_int_tuple(int_tuple: IntTuple, *, loc=None, ip=None) -> ir.Value:
    _check_int_tuple(int_tuple)
    return _pack_x(
        int_tuple, _cute_ir.pack_int_tuple, _cute_ir.MakeIntTupleOp, loc=loc, ip=ip
    )


def _pack_tile(tile: Tile, *, loc=None, ip=None) -> ir.Value:
    _check_tile(tile)

    def expand_leaves(tile) -> list:
        leaves = []
        for e in tile:
            if isinstance(e, _Layout):
                leaves.extend(list(flatten_to_tuple(e.shape)))
                leaves.extend(list(flatten_to_tuple(e.stride)))
            else:
                leaves.append(e)
        return leaves

    layout_leaves = flatten_to_tuple(tile)
    dyn_elems = expand_leaves(layout_leaves)
    dyn_elems = [
        _get_typed_value(x) for x in dyn_elems if isinstance(x, (Integer, ir.Value))
    ]

    res_ty = _cute_ir.pack_tile(tile)
    return _cute_ir.make_tile(res_ty, dyn_elems, loc=loc, ip=ip)


def _unpack_x_tuple(t: Union[ir.Type, ir.Value], *, loc=None, ip=None) -> XTuple:
    # If t is an MLIR type, make sure it's static and make a Value
    if isinstance(t, ir.Type):
        if not _cute_ir.is_static(t):
            raise ValueError()
        t = _cute_ir.static(t)

    if isinstance(t, ir.Value):
        input_ty = t.type
        if t.type.rank == 0:
            # Handle this case separately, _cute_ir.get_leaves will return an Op in this case
            vals = []
        else:
            vals = _cute_ir.get_leaves(t, loc=loc, ip=ip)
            if not isinstance(vals, list):
                vals = [vals]
    else:
        raise TypeError(f"expects static type or value, but got {t}")

    # CuTe IR only supports Int32 for now. Need to support detection of other types
    res = _cute_ir.unpack_x_tuple(input_ty, vals)

    def post_process(x):
        if isinstance(x, _cute_ir.ScaledBasis):
            return ScaledBasis(post_process(x.get_value()), x.get_mode())
        elif isinstance(x, _cute_ir.Ratio):
            return Ratio(x.numerator, x.denominator)
        else:
            return x

    return transform_leaf(post_process, res)


####################################################################################################
# Validation helpers
####################################################################################################


def _check_shape(shape: Shape) -> None:
    if is_integer(shape):
        if isinstance(shape, int):
            if shape <= 0:
                raise ValueError(
                    f"Expected size in shape to be strictly positive, but got {shape}"
                )
        elif isinstance(shape, Integer):
            pass
        else:
            raise TypeError(f"Expected size be int or Integer, but got {type(shape)}")
    elif isinstance(shape, tuple):
        for s in shape:
            _check_shape(s)
    else:
        raise ValueError(
            f"Expected Shape, which is a positive integer or tuple of Shapes, but got {shape}"
        )


def _check_coord(coord: Coord) -> None:
    flat_coord = flatten_to_tuple(coord)
    if not all(is_integer(c) or c is None for c in flat_coord):
        raise ValueError(
            f"Expected Coord, whose leaves are integers or None, but got {coord}"
        )


def _check_stride(stride: Stride) -> None:
    flat_stride = flatten_to_tuple(stride)
    if not all(is_integer(s) or isinstance(s, ScaledBasis) for s in flat_stride):
        raise ValueError(
            f"Expected Stride, whose leaves are integers or ScaledBasis, but got {stride}"
        )


def _check_int_tuple(int_tuple: IntTuple) -> None:
    flat_int_tuple = flatten_to_tuple(int_tuple)
    if not all(is_integer(d) for d in flat_int_tuple):
        raise ValueError(
            f"Expected IntTuple, whose leaves are integers, but got {int_tuple}"
        )


def _check_tile(tile: Tile) -> None:
    flat_tile = flatten_to_tuple(tile)
    if not all(is_integer(t) or isinstance(t, _Layout) or t is None for t in flat_tile):
        raise ValueError(
            f"Expected Tile, whose leaves are integers or Layout or None, but got {tile}"
        )


####################################################################################################
#
# Core types
#
####################################################################################################


class IntValue(cutlass_arith.ArithValue):
    """Internal representation of constrained integer types with divisibility information.

    IntValue serves as a proxy for constrained integer types in the CuTe IR. Rather than
    directly storing values of IntTupleType with depth=0, it stores the result of the
    `cute.get_scalars` operation applied to such values.

    This class represents the following sequence of operations in the IR:
      %0 = ... : (...) -> !cute.int_tuple<"?">
      %1 = cute.get_scalars(%0) : (!cute.int_tuple<"?">) -> i32

    where the first operation produces a `cute.int_tuple<"?">` with depth=0 and rank=1. It
    automatically emit `cute.get_scalars` and track it.

    IntValue inherits behavior from ArithValue with the following extensions:
      * Overloaded operations that accept IntTupleType values to propagate divisibility information
      * Support for CuTe operations that utilize divisibility constraints

    API for interacting with IntValue:
      * get_typed_value() - Returns the value as an IntTupleType
      * get_divisibility() - Returns the divisibility constraint of the value
    """

    def __init__(self, v, signed=True):
        # Cute Constrained Int Type is always signed
        if isinstance(v, int):
            v = _pack_int_tuple(v)

        if isinstance(v.type, _cute_ir.IntTupleType):
            scalar_val = _cute_ir.get_scalars(v)
            super().__init__(scalar_val, True)
        else:
            super().__init__(v, True)

    def get_typed_value(self):
        if isinstance(self.type, ir.IntegerType):
            def_op = self.owner.operation
            if def_op.name == "cute.get_scalars":
                return def_op.operands[0]

        assert not isinstance(self.type, _cute_ir.IntTupleType)

        return _pack_int_tuple(self)

    @property
    def divisibility(self):
        if isinstance(self.get_typed_value().type, _cute_ir.IntTupleType):
            return self.get_typed_value().type.get_divisibility([0])
        else:
            return 1

    def __str__(self):
        if self.divisibility == 1:
            return f"?"
        else:
            return f"?{{div={self.divisibility}}}"

    def __repr__(self):
        parent_name = cutlass_arith.ArithValue.__name__
        return super().__str__().replace(parent_name, IntValue.__name__)

    def pretty_str(self):
        return self.__str__()

    @staticmethod
    def _binary_op(op):
        def wrapper(self, other, **kwargs):
            if isinstance(other, IntValue):
                other_val = other.get_typed_value()
            elif isinstance(other, ir.Value) and isinstance(
                other.type, _cute_ir.IntTupleType
            ):
                other_val = other
            elif isinstance(other, ir.Value) and isinstance(other.type, ir.IntegerType):
                other = cutlass_arith.int_to_int(other, Int32, **kwargs)
                other_val = _pack_int_tuple(other)
            elif isinstance(other, (int, bool)):
                other_val = _pack_int_tuple(int(other))
            else:
                # Dispatch to `__rmul__` of `other`
                return NotImplemented

            return IntValue(op(self, other_val, **kwargs))

        return wrapper

    @dsl_user_op
    @_binary_op
    def __add__(self, other, *, loc=None, ip=None):
        return _cute_ir.add_offset(self.get_typed_value(), other, loc=loc, ip=ip)

    @dsl_user_op
    @_binary_op
    def __sub__(self, other, *, loc=None, ip=None):
        return _cute_ir.tuple_sub(self.get_typed_value(), other, loc=loc, ip=ip)

    @dsl_user_op
    @_binary_op
    def __mul__(self, other, *, loc=None, ip=None):
        return _cute_ir.tuple_mul(self.get_typed_value(), other, loc=loc, ip=ip)

    @dsl_user_op
    @_binary_op
    def __floordiv__(self, other, *, loc=None, ip=None) -> "IntValue":
        return _cute_ir.tuple_div(self.get_typed_value(), other, loc=loc, ip=ip)

    @dsl_user_op
    @_binary_op
    def __mod__(self, other, *, loc=None, ip=None) -> cutlass_arith.ArithValue:
        return _cute_ir.tuple_mod(self.get_typed_value(), other, loc=loc, ip=ip)

    @dsl_user_op
    @_binary_op
    def __radd__(self, other, *, loc=None, ip=None) -> "IntValue":
        return _cute_ir.add_offset(other, self.get_typed_value(), loc=loc, ip=ip)

    @dsl_user_op
    @_binary_op
    def __rsub__(self, other, *, loc=None, ip=None) -> "IntValue":
        return _cute_ir.tuple_sub(other, self.get_typed_value(), loc=loc, ip=ip)

    @dsl_user_op
    @_binary_op
    def __rmul__(self, other, *, loc=None, ip=None):
        return _cute_ir.tuple_mul(other, self.get_typed_value(), loc=loc, ip=ip)

    @dsl_user_op
    @_binary_op
    def __rfloordiv__(self, other, *, loc=None, ip=None) -> "IntValue":
        return _cute_ir.tuple_div(other, self.get_typed_value(), loc=loc, ip=ip)

    @dsl_user_op
    @_binary_op
    def __rmod__(self, other, *, loc=None, ip=None) -> "IntValue":
        return _cute_ir.tuple_mod(other, self.get_typed_value(), loc=loc, ip=ip)


class Ratio(_Ratio):
    """A class representing a rational number as a ratio of two integers.

    Ratio is used in CuTe to represent exact fractional values that arise in
    tensor layout operations, particularly in composition operations where
    divisibility conditions may not be satisfied.

    :param numerator: The numerator of the ratio
    :type numerator: int
    :param denominator: The denominator of the ratio
    :type denominator: int
    :raises TypeError: If numerator or denominator are not integers
    """

    def __init__(self, numerator: int, denominator: int):
        if not isinstance(numerator, int) or not isinstance(denominator, int):
            raise TypeError(
                f"numerator and denominator must be integers, but got {numerator} and {denominator}"
            )
        super().__init__(numerator, denominator)

    def is_integral(self) -> bool:
        """Check if the ratio represents an integer value.

        :return: True if the numerator is divisible by the denominator
        :rtype: bool
        """
        return super().is_integral()

    def reduced(self) -> "Ratio":
        """Return a new Ratio with the numerator and denominator reduced to lowest terms.

        :return: A new Ratio in reduced form
        :rtype: Ratio
        """
        res = super().reduced()
        return Ratio(res.numerator, res.denominator)

    def __mul__(self, other):
        """Multiply this ratio by another ratio or an integer.

        :param other: The value to multiply by
        :type other: Union[Ratio, int]
        :return: A new ratio representing the product
        :rtype: Ratio
        :raises TypeError: If other is not a Ratio or int
        """
        if isinstance(other, Ratio):
            return Ratio(
                self.numerator * other.numerator,
                self.denominator * other.denominator,
            )
        elif isinstance(other, int):
            return Ratio(self.numerator * other, self.denominator)
        else:
            raise TypeError(f"Cannot multiply Ratio with {type(other)}")

    def __rmul__(self, other):
        """Right multiplication operation.

        :param other: The value to multiply by
        :type other: Union[Ratio, int]
        :return: A new ratio representing the product
        :rtype: Ratio
        """
        return self.__mul__(other)

    def __str__(self):
        """String representation of the ratio.

        :return: String in the format "numerator/denominator"
        :rtype: str
        """
        return super().__str__()

    def to(self, dtype):
        """Convert the ratio to another type.

        :param dtype: The target type for conversion
        :type dtype: type
        :return: The ratio converted to the specified type
        :raises TypeError: If conversion to the specified type is not supported
        """
        if dtype is Ratio:
            return self
        elif dtype is float:
            return self.numerator / self.denominator
        elif dtype is int:
            return self.numerator // self.denominator
        elif issubclass(dtype, _Ratio):
            return self
        else:
            raise TypeError(f"Cannot convert Ratio to {dtype}")


class ScaledBasis:
    """A class representing a scaled basis element in CuTe's layout algebra.

    ScaledBasis is used to represent elements in the layout algebra, particularly
    in the context of composition operations. It consists of a value (scale) and
    a mode that identifies mode of the basis element.

    :param value: The scale value
    :type value: Union[int, Integer, Ratio, ir.Value]
    :param mode: The mode identifying the basis element
    :type mode: Union[int, List[int]]
    :raises TypeError: If mode is not an integer or list of integers

    **Examples:**

    .. code-block:: python

        # Create a scaled basis with integer scale and mode
        sb1 = ScaledBasis(2, 0)  # 2 * E(0)

        # Create a scaled basis with a Ratio scale
        sb2 = ScaledBasis(Ratio(1, 2), 1)  # (1/2) * E(1)

        # Create a scaled basis with a list of modes
        sb3 = ScaledBasis(4, [0, 1])  # 4 * E([0, 1])

        # Scaled basis elements are commonly used in layout strides
        layout = make_layout((4, 8), stride=(ScaledBasis(2, 0), ScaledBasis(1, 1)))

        # This creates a layout with strides (2@0, 1@1) representing
        # a coordinate system where each dimension has its own basis

        # Example: Mapping coordinates to indices using the layout
        coord = (2, 3)
        idx = crd2idx(coord, layout)  # Maps (2, 3) to (4, 3)
    """

    def __init__(self, value, mode) -> None:
        if isinstance(mode, int):
            self._mode = [mode]
        else:
            if any(not isinstance(x, int) for x in mode):
                raise TypeError("Mode must be a list of integers")
            self._mode = mode

        self._value = value

    def is_static(self) -> bool:
        """Check if the value is statically known.

        :return: True if the value is not a dynamic expression
        :rtype: bool
        """
        return not is_dynamic_expression(self._value)

    def to(self, dtype):
        """Convert to another type.

        :param dtype: The target type for conversion
        :type dtype: type
        :return: The ScaledBasis converted to the specified type
        :raises TypeError: If conversion to the specified type is not supported
        """
        if dtype is ScaledBasis:
            return self
        elif dtype is _ScaledBasis:
            if isinstance(self._value, Ratio):
                scale = self._value
            elif isinstance(self._value, Integer):
                scale = self._value.ir_value()
            else:
                scale = self._value

            if isinstance(scale, IntValue):
                return _ScaledBasis(scale.get_typed_value(), self._mode)
            else:
                return _ScaledBasis(scale, self._mode)
        else:
            raise TypeError(f"Cannot convert ScaledBasis to {dtype}")

    def __str__(self):
        return f"{self.to(_ScaledBasis).__str__()}"

    def __hash__(self):
        if isinstance(self.mode, list):
            return hash((self.value, tuple(self.mode)))
        else:
            return hash((self.value, self.mode))

    @property
    def value(self):
        """Get the scale value.

        :return: The scale value
        """
        return self._value

    @property
    def mode(self) -> List[int]:
        """Get the mode identifying the basis element.

        :return: The mode as a list of integers
        :rtype: List[int]
        """
        return self._mode

    def __eq__(self, other):
        if isinstance(other, ScaledBasis):
            return self.value == other.value and self.mode == other.mode
        else:
            return False

    def __rmul__(self, scale: Union[Int, ir.Value, Ratio]) -> "ScaledBasis":
        """Right multiplication by a scale factor.

        This operation is used in layout algebra to scale basis elements,
        which is essential for operations like composition and partitioning.

        :param scale: The scale factor
        :type scale: Union[Int, ir.Value, Ratio]
        :return: A new scaled basis element
        :rtype: ScaledBasis
        :raises TypeError: If scale is not of a supported type
        :raises NotImplementedError: If scaling a basis element with a ratio value
        """
        if not isinstance(scale, (int, Integer, Ratio, ir.Value)):
            raise TypeError(
                f"scale must be an integer or a ratio, but got {type(scale)}"
            )
        if isinstance(self.value, Ratio):
            raise NotImplementedError(
                "scaling a basis element having a ratio is not supported"
            )

        value = self.value

        if not isinstance(value, (Integer, Ratio, int, cutlass_arith.ArithValue)):
            raise TypeError(f"Don't support {type(value)} for ScaledBasis")

        # Lift to IntValue type to preserve type info as much as possible
        if isinstance(scale, cutlass_arith.ArithValue):
            scale = IntValue(_pack_int_tuple(cutlass_arith.int_to_int(scale, Int32)))

        if isinstance(value, cutlass_arith.ArithValue):
            value = IntValue(_pack_int_tuple(cutlass_arith.int_to_int(value, Int32)))
        elif isinstance(value, Integer):
            value = value.ir_value()

        return ScaledBasis(scale * value, self.mode)  # type: ignore


def E(mode: Union[int, List[int]]) -> ScaledBasis:
    """Create a unit ScaledBasis element with the specified mode.

    This function creates a ScaledBasis with value 1 and the given mode.
    The mode represents the coordinate axis or dimension in the layout.

    :param mode: The mode (dimension) for the basis element, either a single integer or a list of integers
    :type mode: Union[int, List[int]]
    :return: A ScaledBasis with value 1 and the specified mode
    :rtype: ScaledBasis
    :raises TypeError: If mode is not an integer or a list

    **Examples:**

    .. code-block:: python

        # Create a basis element for the first dimension (mode 0)
        e0 = E(0)

        # Create a basis element for the second dimension (mode 1)
        e1 = E(1)

        # Create a basis element for a hierarchical dimension
        e_hier = E([0, 1])
    """
    if isinstance(mode, int):
        mode = [mode]

    if not isinstance(mode, list):
        raise TypeError(f"expects a list, got {type(mode)}")

    if not mode:
        return 1

    return ScaledBasis(1, mode)


def get_divisibility(x: Union[int, Integer]) -> int:
    if isinstance(x, int):
        return x

    if isinstance(x, Integer):
        x = x.value

    if isinstance(x, IntValue):
        return x.divisibility
    else:
        return 1


@ir.register_value_caster(_cute_ir.SwizzleType.get_static_typeid(), replace=True)
class Swizzle(ir.Value):
    """
    Swizzle is a transformation that permutes the elements of a layout.

    Swizzles are used to rearrange data elements to improve memory access patterns
    and computational efficiency.

    Swizzle is defined by three parameters:
    - MBase: The number of least-significant bits to keep constant
    - BBits: The number of bits in the mask
    - SShift: The distance to shift the mask

    The mask is applied to the least-significant bits of the layout.

    .. code-block::

        0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
                                      ^--^ MBase is the number of least-sig bits to keep constant
                         ^-^       ^-^     BBits is the number of bits in the mask
                           ^---------^     SShift is the distance to shift the YYY mask
                                              (pos shifts YYY to the right, neg shifts YYY to the left)

        e.g. Given
        0bxxxxxxxxxxxxxxxxYYxxxxxxxxxZZxxx

        the result is
        0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx where AA = ZZ `xor` YY

    """

    def __str__(self):
        # Cut off the MLIR type's string for making pretty_str more concise
        return self.type.__str__()[15 : 15 + 8]


@ir.register_value_caster(_cute_ir.LayoutType.get_static_typeid(), replace=True)
class _Layout(Layout):
    """Layout is CuTe's core abstraction for representing tensor layouts.

    A Layout maps from a logical coordinate space to an index space, defined by a
    pair of (Shape, Stride). The Shape defines the abstract dimensions of the Layout,
    while the Stride defines how coordinates within the Shape map to linear indices.

    Layouts present a common interface to multidimensional array access that abstracts
    away the details of how array elements are organized in memory. This allows algorithms
    to be written generically, so that layouts can change without requiring code changes.

    CuTe layouts are inherently hierarchical, constructed from smaller, nested layouts
    that can represent complex mappings required by GPU tensor instructions. They support
    a rich algebra of operations including concatenation, coalescence, composition,
    complement, and inversion.

    :ivar shape: An IntTuple representing the dimensions of the layout.
    :ivar stride: An IntTuple representing the strides of the layout.
    :ivar max_alignment: The maximum alignment of the layout.

    **Examples:**

    .. code-block:: python

        # Creating a layout with shape (4,8) and default stride (layout left / "column major")
        layout = cute.make_layout((4, 8))

        # Creating a layout with explicit shape and stride
        layout = cute.make_layout((4, 8), stride=(8, 1))

        # Accessing a specific coordinate: (2, 3) -> 2 * 8 + 3 * 1 = 19
        idx = cute.crd2idx((2, 3), layout)
    """

    def __init__(self, op_result) -> None:
        """Initialize a Layout object.

        :param op_result: The operation result value to wrap.
        """
        super().__init__(op_result)

    def __str__(self) -> str:
        """Return a string representation of the layout.

        :return: A string in the format "shape:stride".
        """
        return f"{pretty_str(self.shape)}:{pretty_str(self.stride)}"

    @property
    def shape(self, *, loc=None, ip=None) -> Shape:
        """Get the shape of the layout.

        The shape defines the dimensions and structure of the layout's
        coordinate space.

        :param loc: Optional location information for debugging.
        :param ip: Optional insertion point for IR generation.
        :return: The hierarchical shape of the layout.
        """
        return _unpack_x_tuple(_cute_ir.get_shape(self, loc=loc, ip=ip), loc=loc, ip=ip)

    @property
    def stride(self, *, loc=None, ip=None) -> Stride:
        """Get the stride of the layout.

        The stride defines how coordinates map to linear indices in memory.

        :param loc: Optional location information for debugging.
        :param ip: Optional insertion point for IR generation.
        :return: The hierarchical stride of the layout.
        """
        return _unpack_x_tuple(
            _cute_ir.get_stride(self, loc=loc, ip=ip), loc=loc, ip=ip
        )

    @property
    def max_alignment(self) -> int:
        """Get the maximum alignment of the layout.

        :return: The maximum alignment in bytes.
        """
        return self.type.max_alignment

    def __eq__(self, other) -> Union[bool, Boolean]:
        """Check if this layout is equal to another layout.

        Two layouts are equal if they have the same shape and stride.

        :param other: The layout to compare with.
        :return: True if layouts are equal, False otherwise.
            May return an IR value for dynamic layouts.
        """
        if isinstance(other, Layout):
            if is_static(self.type) and is_static(other.type):
                return self.type == other.type
            return Boolean(_cute_ir.equal(self, other))
        else:
            return False

    def __req__(self, other) -> Union[bool, Boolean]:
        """Reflected equality check.

        :param other: The layout to compare with.
        :return: Result of other.__eq__(self).
        """
        if isinstance(other, Layout):
            return other.__eq__(self)
        return False

    def __ne__(self, other) -> Union[bool, Boolean]:
        """Check if this layout is not equal to another layout.

        :param other: The layout to compare with.
        :return: True if layouts are not equal, False otherwise.
        """
        if isinstance(other, Layout):
            if is_static(self.type) and is_static(other.type):
                return self.type != other.type
            return Boolean(not_(_cute_ir.equal(self, other)))
        else:
            return True

    def __rne__(self, other) -> Union[bool, Boolean]:
        """Reflected inequality check.

        :param other: The layout to compare with.
        :return: Result of other.__ne__(self).
        """
        if isinstance(other, Layout):
            return other.__ne__(self)
        return False

    def __getitem__(self, idx: int) -> Layout:
        """
        Top-level `get` to provide a syntax similar to `tuple`.
        """
        return get(self, mode=[idx])

    @dsl_user_op
    def __call__(self, coord: Coord, loc=None, ip=None) -> IntTuple:
        return crd2idx(coord, self, loc=loc, ip=ip)

    @dsl_user_op
    def get_hier_coord(self, idx, *, loc=None, ip=None) -> Coord:
        """Get the hierarchical coordinate corresponding to a linear index.

        This method maps from a linear index back to the logical coordinate
        in the layout's coordinate space.

        :param idx: The linear index to convert.
        :return: The hierarchical coordinate corresponding to the index.

        **Examples:**

        .. code-block:: python

            layout = make_layout((4, 8), stride=(8, 1))

            # map linear index back to coordinate: 5 -> (1, 1)
            coord = get_hier_coord(5, layout)
        """
        idx_val = Int32(idx).ir_value()
        crd = _cute_ir.get_hier_coord(idx_val, self, loc=loc, ip=ip)
        return _unpack_x_tuple(crd)

    @dsl_user_op
    def get_flat_coord(self, idx, *, loc=None, ip=None) -> Coord:
        idx_val = Int32(idx).ir_value()
        res = _cute_ir.get_flat_coord(idx_val, self, loc=loc, ip=ip)
        return _unpack_x_tuple(res, loc=loc, ip=ip)


@ir.register_value_caster(_cute_ir.ComposedLayoutType.get_static_typeid(), replace=True)
class ComposedLayout(ir.Value):
    r"""ComposedLayout represents the functional composition of layouts in CuTe.

    A ComposedLayout is formed by the composition of three components:
    inner o offset o outer, where:

    - inner: The inner layout or swizzle that is applied last
    - offset: An integer tuple representing a coordinate offset
    - outer: The outer layout that is applied first

    ComposedLayout implements the functional composition operation where:

    .. math::

        R(c) := (inner \\circ offset \\circ outer)(c) := inner(offset + outer(c))

    This composition allows for complex transformations of coordinates and indices,
    enabling operations like tiling, partitioning, and reshaping of data.

    :ivar inner: The inner layout or swizzle component
    :ivar offset: The coordinate offset applied between inner and outer layouts
    :ivar outer: The outer layout component
    :ivar max_alignment: The maximum alignment of the composed layout

    **Examples:**

    .. code-block:: python

        # Create a composed layout with inner layout, offset, and outer layout

        # inner layout: (4, 8):(1, 4)
        inner_layout = make_layout((4, 8))

        offset = (0, 0)

        # outer layout: (2, 2):(1@0, 1@1)
        outer_layout = make_layout((2, 2), stride=(1 * E(0), 1 * E(1)))

        # composed layout: (inner o offset o outer)
        composed = make_composed_layout(inner_layout, offset, outer_layout)

        # Accessing components of the composed layout
        inner = composed.inner
        offset = composed.offset
        outer = composed.outer

        # map coordinate (0, 1) to linear index
        #  - outer(0, 1) = (0, 1)
        #  - offset + outer(0, 1) = (0, 1)
        #  - inner(0, 1) = 0 * 1 + 1 * 4 = 4
        idx = crd2idx((0, 1), composed)

        # Composition is used in many tiling operations
        # For example, in logical_product, raked_product, and blocked_product
    """

    def __init__(self, value) -> None:
        """Initialize a ComposedLayout object.

        :param value: The operation result value to wrap.
        """
        super().__init__(value)

    def __str__(self) -> str:
        return f"{pretty_str(self.inner)} o {pretty_str(self.offset)} o {pretty_str(self.outer)}"

    @property
    def inner(self, *, loc=None, ip=None) -> Union[Swizzle, Layout]:
        return _cute_ir.composed_get_inner(self, loc=loc, ip=ip)

    @property
    def offset(self, *, loc=None, ip=None) -> IntTuple:
        return _unpack_x_tuple(_cute_ir.composed_get_offset(self, loc=loc, ip=ip))

    @property
    def outer(self, *, loc=None, ip=None) -> Layout:
        return _cute_ir.composed_get_outer(self, loc=loc, ip=ip)

    @property
    def shape(self, *, loc=None, ip=None) -> Shape:
        return _unpack_x_tuple(_cute_ir.get_shape(self, loc=loc, ip=ip), loc=loc, ip=ip)

    @property
    def max_alignment(self) -> int:
        return self.type.max_alignment

    def __eq__(self, other) -> Union[bool, Boolean]:
        if isinstance(other, ComposedLayout):
            if is_static(self.type) and is_static(other.type):
                return self.type == other.type
            else:
                raise NotImplementedError(
                    f"runtime comparison of composed layouts is not supported, got `{self}` and `{other}`"
                )
        else:
            return False

    def __req__(self, other) -> Union[bool, Boolean]:
        if isinstance(other, ComposedLayout):
            return Boolean(other.__eq__(self))
        return False

    def __ne__(self, other) -> Union[bool, Boolean]:
        return not self.__eq__(other)

    def __rne__(self, other) -> Union[bool, Boolean]:
        if isinstance(other, ComposedLayout):
            return other.__ne__(self)
        return False

    def __getitem__(self, idx: int) -> "ComposedLayout":
        """
        Top-level `get` to provide a syntax similar to `tuple`.
        """
        return get(self, mode=[idx])

    @dsl_user_op
    def __call__(self, coord: Coord, loc=None, ip=None) -> IntTuple:
        return crd2idx(coord, self, loc=loc, ip=ip)


@ir.register_value_caster(_cute_ir.PtrType.get_static_typeid(), replace=True)
class _Pointer(Pointer):
    """
    A pointer class representing a memory address with specific properties.

    Pointers are a fundamental type of iterator/engine that support random-access operations.
    They can be offset by elements of a layout's codomain and dereferenced to produce values.

    :param value: The MLIR operation result value to initialize the pointer with
    :type value: ir.Value

    :ivar type: The MLIR type of the pointer
    :vartype type: Type
    :ivar value_type: The type of value this pointer points to
    :vartype value_type: Type
    :ivar memspace: The memory space where the pointer data resides (e.g., gmem, smem, rmem)
    :vartype memspace: AddressSpace

    :note: When composed with a layout, a pointer forms a tensor: T = E ∘ L, where E is the pointer
           and L is the layout. The tensor evaluates the layout by mapping a coordinate c to the
           codomain, offsets the pointer accordingly, and dereferences the result:
           T(c) = (E ∘ L)(c) = *(E + L(c))
    """

    def __init__(self, value) -> None:
        assert isinstance(value, ir.Value)
        self.value = ir.Value(value)

    def __str__(self) -> str:
        # Cut off the MLIR type's string for making pretty_str more concise
        return self.type.__str__()[6:]

    def __get_mlir_types__(self):
        return [self.value.type]

    def __extract_mlir_values__(self):
        return [self.value]

    def __new_from_mlir_values__(self, values):
        # Only expecting single value of _Pointer instance or ir.Value
        # In this context, a _Pointer instance is an encapsulated ir.Value which is automatically created
        # by value caster for cute.ptr typed values
        assert len(values) == 1, f"Expected 1 value, but got {len(values)}"
        assert isinstance(
            values[0], (_Pointer, ir.Value)
        ), f"Expected _Pointer or ir.Value, but got {type(values[0])}"
        return _Pointer(
            values[0] if isinstance(values[0], ir.Value) else values[0].value
        )

    @property
    @lru_cache_ir()
    def dtype(self) -> Type[Numeric]:
        return Numeric.from_mlir_type(self.value.type.value_type)

    @property
    def alignment(self) -> int:
        return self.type.alignment

    @property
    def max_alignment(self) -> int:
        return self.type.max_alignment

    @property
    @lru_cache_ir()
    def memspace(self) -> AddressSpace:
        return AddressSpace(self.type.address_space)

    # Make it behave as if it inherited from ir.Value
    @property
    @lru_cache_ir()
    def type(self) -> ir.Type:
        return self.value.type

    # Only use if you absolutely need to get the LLVM pointer Value
    @property
    @lru_cache_ir()
    def llvm_ptr(self, *, loc=None, ip=None) -> ir.Value:
        """
        Get the LLVM pointer representation of this pointer.

        :param loc: The source location for the operation, defaults to None
        :type loc: Location, optional
        :param ip: The insertion point for the operation, defaults to None
        :type ip: InsertionPoint, optional
        :return: The LLVM pointer representation
        :rtype: ir.Value
        """
        llvm_ptr_ty = llvm.PointerType.get(self.memspace.value)
        return builtin.unrealized_conversion_cast(
            [llvm_ptr_ty], [self.value], loc=loc, ip=ip
        )

    def __add__(self, offset: IntTuple) -> Pointer:
        """
        Offset the pointer by elements of a layout's codomain.

        :param offset: The offset to add to the pointer
        :type offset: IntTuple
        :return: A new pointer offset by the specified amount
        :rtype: ir.Value
        """
        offset = _pack_int_tuple(offset)
        return _cute_ir.add_offset(self.value, offset=offset)

    @dsl_user_op
    def toint(self, *, loc=None, ip=None):
        if self.memspace in (AddressSpace.gmem, AddressSpace.generic):
            res_type = Int64
        else:
            res_type = Int32

        return res_type(
            _cute_ir.ptrtoint(res_type.mlir_type, self.value, loc=loc, ip=ip)
        )

    @dsl_user_op
    def align(self, min_align: int, *, loc=None, ip=None) -> Pointer:
        """
        Align a pointer to a specified byte alignment.

        :param min_align: The minimum byte alignment requirement. Must be a power of 2.
        :type min_align: int
        :param loc: The source location for the operation, defaults to None
        :type loc: Location, optional
        :param ip: The insertion point for the operation, defaults to None
        :type ip: InsertionPoint, optional
        :return: The aligned new pointer that satisfies alignment request.
        :rtype: Pointer
        :raises ValueError: If the alignment is not a power of 2.
        :raises TypeError: If pointer is in tmem address space.
        """

        if (min_align & (min_align - 1)) != 0:
            raise ValueError("Alignment must be a power of 2")

        assert isinstance(self.type, _cute_ir.PtrType)
        if self.memspace is AddressSpace.tmem:
            raise ValueError("aligning a TMEM pointer is not supported")

        if min_align <= self.alignment:
            return self

        dtype = Numeric.from_mlir_type(self.type.value_type)
        # Convert pointer to integer
        address_int = self.toint(loc=loc, ip=ip)
        # Align the address
        aligned_address = (address_int + min_align - 1) & ~(min_align - 1)

        return make_ptr(
            dtype,
            aligned_address,
            self.memspace,
            assumed_align=min_align,
            loc=loc,
            ip=ip,
        )


@ir.register_value_caster(_cute_ir.MemRefType.get_static_typeid(), replace=True)
@ir.register_value_caster(_cute_ir.CoordTensorType.get_static_typeid(), replace=True)
@ir.register_value_caster(
    _cute_nvgpu_ir.SmemDescViewType.get_static_typeid(), replace=True
)
class _Tensor(Tensor):
    """A tensor class representing the composition of an iterator (engine) with a layout.

    A tensor evaluates the layout by mapping a coordinate to the codomain, offsets the
    iterator accordingly, and dereferences the result to obtain the tensor's value.
    Formally: T(c) = (E ∘ L)(c) = *(E + L(c)), where E is the iterator/engine and L is the layout.

    :param value: The MLIR operation result value to initialize the tensor with
    :type value: ir.Value
    :param dtype: The user specified data type of the tensor elements. It could be \
        different from the underlying dtype in the iterator. The default is None.
    :type dtype: Type[Numeric], optional

    Attributes:
        iterator: The pointer or iterator (engine) component of the tensor
        layout: The layout component defining the mapping from coordinates to offsets
        shape: The shape of the tensor, inherited from the layout
        stride: The stride of the tensor, inherited from the layout
        element_type: The data type of the tensor elements
        memspace: The memory space where the tensor data resides

    Notes:
        - The tensor supports both direct element access via coordinates and slicing operations
        - Load/store operations are only supported for specific memory spaces (rmem, smem, gmem, generic)
        - For composed layouts, stride information is not directly accessible
        - Dynamic layouts do not support vector load/store operations

    **Examples:**

    .. code-block:: python

        # Create a tensor with shape (4,8) in row-major layout
        tensor = make_tensor(ptr, make_layout(shape=(4,8), stride=(8,1)))

        # Access individual element
        val = tensor[0, 0]    # or val = tensor[(0, 0)]

        # Slice operation - get first column
        subtensor = tensor[None, 0]  # or subtensor = tensor[(None, 0)]
    """

    def __init__(self, value, dtype: Optional[Type[Numeric]] = None):
        self._dtype = dtype
        if isinstance(value, ir.Value):
            self.value = value
        elif isinstance(value, _Tensor):
            self.value = value.value
        else:
            raise TypeError(f"Expected ir.Value or core._Tensor, got {type(value)}")

        # Set iterator
        iter_val = _cute_ir.get_iter(self.value)
        if isinstance(iter_val, Pointer):
            self._iterator = iter_val
        elif isinstance(iter_val.type, _cute_ir.IntTupleType):
            self._iterator = _unpack_x_tuple(iter_val)
        elif isinstance(iter_val, ir.Value):
            # Example: SMEM descriptor iterator, not well supported today
            self._iterator = iter_val
        else:
            raise TypeError(f"unsupported iterator type, got {type(iter_val)}")

        # Set dtype
        if self._dtype is None:
            if is_int_tuple(self.iterator):
                self._dtype = IntTuple
            elif isinstance(self.iterator, Pointer):
                self._dtype = self.iterator.value_type
            elif isinstance(self.type, _cute_nvgpu_ir.SmemDescViewType):
                # SmemDescViewType do not need dtype
                self._dtype = None
            else:
                raise TypeError(f"unsupported iterator type, got {type(self.iterator)}")

    def __str__(self):
        return f"tensor<{pretty_str(self.iterator)} o {pretty_str(self.layout)}>"

    def __extract_mlir_values__(self):
        return [self.value]

    def __new_from_mlir_values__(self, values):
        # Only expecting single value of _Tensor or ir.Value
        # In this context, a _Tensor instance is an encapsulated ir.Value which is automatically created
        # by value caster for MemRef/CoordTensor/SmemDescView typed values
        assert len(values) == 1, f"Expected 1 value, but got {len(values)}"
        assert isinstance(
            values[0], (_Tensor, ir.Value)
        ), f"Expected _Tensor or ir.Value, but got {type(values[0])}"
        return _Tensor(
            values[0] if isinstance(values[0], ir.Value) else values[0].value,
            dtype=self.element_type,
        )

    # Cheat to let `Type(_Tensor())` to return cute.Tensor
    @property
    def __class__(self) -> Type[Tensor]:
        return Tensor

    # Make it behave as if it inherited from ir.Value
    @property
    @lru_cache_ir()
    def type(self) -> ir.Type:
        return self.value.type

    @dsl_user_op
    def __getitem__(
        self, crd: Coord, *, loc=None, ip=None
    ) -> Union[Tensor, Numeric, IntTuple]:
        """Access or slice tensor elements using coordinates.

        This method implements
        * tensor evaluation T(c) = *(E + L(c)) when `c` is a coordinate without slicing, or
        * tensor slicing operations T(c) = make_tensor(E + L(c), slice(L, c))
        where E is the iterator/engine and L is the layout

        :param crd: Coordinate or slice specification for accessing tensor elements
        :type crd: Coord
        :param loc: Source location for MLIR operation tracking, defaults to None
        :type loc: Optional[Location]
        :param ip: Insertion point for MLIR operation, defaults to None
        :type ip: Optional[InsertionPoint]
        :return: Tensor element value or sliced subtensor
        :rtype: Union[Tensor, ir.Value, IntTuple]

        :raises ValueError: If coordinate access is invalid for the tensor layout

        **Examples:**

        .. code-block:: python

            # Create a tensor with pointer iterator
            ptr = make_ptr(cutlass.Float32, 0, cutlass.AddressSpace.gmem)
            layout = make_layout((64, 128))  # leftmost mode is major
            tensor = make_tensor(ptr, layout)  # Tensor using pointer iterator

            # Direct element access loads from memory
            val = tensor[0]  # Loads element at offset 0
            val = tensor[1]  # Loads element at offset 4 (4bytes per Float32)
            val = tensor[(0, 1)]  # Loads element at offset 64

            # Create a coord tensor
            layout = make_layout((64, 128), stride=(1 * E(0), 1 * E(1)))
            tensor = make_tensor((128, 128), layout)

            # Direct element access
            val = tensor[0]  # Returns (128, 128)
            val = tensor[(0, 1)]  # Returns (128, 129)

            # Slice access
            sliced = view[(3, None)]  # Returns tensor slice

        .. note::
            Sub-byte types like Float4E2M1FN and Float6E3M2FN are not supported for scalar
            dereference operations. Attempting to set individual elements of tensors with
            these element types will result in errors.

        **Examples:**

        .. code-block:: python

            # Unsupported operations with sub-byte types:
            ptr = make_ptr(cutlass.Float4E2M1FN, 0, cutlass.AddressSpace.gmem)
            tensor = make_tensor(ptr, layout)
            # The following will raise an error:
            val = tensor[0]  # Error: sub-byte scalar dereference not supported

            # Similarly for other sub-byte types:
            ptr = make_ptr(cutlass.Float6E3M2FN, 0, cutlass.AddressSpace.gmem)
            tensor = make_tensor(ptr, layout)
            val = tensor[0]  # Error: sub-byte scalar dereference not supported
        """
        if has_underscore(crd):
            return slice_(self.value, crd)
        elif isinstance(self.type, _cute_ir.CoordTensorType):
            res = _cute_ir.get_iter(slice_(self, crd).value, loc=loc, ip=ip)
            return _unpack_x_tuple(res)
        else:
            self._check_can_load_store()
            self._check_can_dereference()

            crd_val = _pack_coord(crd, loc=loc, ip=ip)
            data_val = _cute_ir.memref_load(self.value, crd_val, loc=loc, ip=ip)
            return self.element_type(data_val)

    def _cvt_to_dest(self, data: Union["TensorSSA", Numeric], *, loc=None, ip=None):
        orig_dtype = data.dtype
        # Implicit upcast to wider type
        if (
            data.dtype.is_same_kind(self.element_type)
            and self.element_type.width >= data.dtype.width
        ):
            data = data.to(self.element_type, loc=loc, ip=ip)  # type: ignore

        if data.dtype.width != self.element_type.width:
            raise ValueError(
                f"Type mismatch, store {orig_dtype} (-> {data.dtype}) "
                f"to Tensor with element type {self.element_type}"
            )

        if data.dtype is Boolean and self.element_type is Boolean:
            # Boolean Numeric and Boolean TensorSSA both hold i1 value, but we need int8 value store to memory
            val = data.ir_value_int8()
        else:
            val = data.ir_value()
        return val

    @dsl_user_op
    def __setitem__(
        self,
        crd: Coord,
        data: Union[int, float, ir.Value, Numeric, "TensorSSA"],
        *,
        loc=None,
        ip=None,
    ) -> None:
        """Set tensor elements at specified coordinates.

        Assigns values to tensor elements through direct coordinate access or slice assignment.
        For slice assignment, the value must be a TensorSSA with matching shape.

        :param crd: Coordinate or slice specification for tensor element assignment
        :type crd: Coord
        :param data: Value to assign - can be scalar or TensorSSA for slice assignment
        :type data: Union[int, float, ir.Value, Numeric, TensorSSA]
        :param loc: Source location for MLIR operation tracking, defaults to None
        :type loc: Optional[Location]
        :param ip: Insertion point for MLIR operation, defaults to None
        :type ip: Optional[InsertionPoint]

        :raises ValueError: If tensor type doesn't support load/store operations
        :raises ValueError: If slice assignment value is not a TensorSSA
        :raises ValueError: If value type doesn't match tensor element type
        :raises NotImplementedError: If value type is not supported

        .. note::
            Sub-byte types like Float4E2M1FN and Float6E3M2FN are not supported for scalar
            dereference operations. Attempting to set individual elements of tensors with
            these element types will result in errors.

        **Examples:**

        .. code-block:: python

            # Unsupported operations with sub-byte types:
            ptr = make_ptr(cutlass.Float4E2M1FN, 0, cutlass.AddressSpace.gmem)
            tensor = make_tensor(ptr, layout)
            # The following will raise an error:
            tensor[0] = 1.0  # Error: sub-byte scalar dereference not supported

            # Similarly for other sub-byte types:
            ptr = make_ptr(cutlass.Float6E3M2FN, 0, cutlass.AddressSpace.gmem)
            tensor = make_tensor(ptr, layout)
            tensor[0] = 0.5  # Error: sub-byte scalar dereference not supported
        """
        self._check_can_load_store()

        # convert scalar type
        if not has_underscore(crd):
            self._check_can_dereference()
            # First, convert ir.Value to Numeric
            if isinstance(data, ir.Value):
                data = as_numeric(data)
            elif isinstance(data, (int, float, bool)):
                data = as_numeric(data)

            if not isinstance(data, Numeric):
                raise ValueError(f"unsupported data type: {type(data)}")

            # Implicit upcast to wider type
            val = self._cvt_to_dest(data, loc=loc, ip=ip)
            if val.type != self.type.value_type:
                raise ValueError(
                    f"type mismatch, store {val.type} to {self.element_type}"
                )

            crd_val = _pack_coord(crd, loc=loc, ip=ip)
            _cute_ir.memref_store(self.value, crd_val, val, loc=loc, ip=ip)
        else:
            if not isinstance(data, TensorSSA):
                raise ValueError(f"expects TensorSSA, but got {data}")

            self.__getitem__(crd).store(data, loc=loc, ip=ip)  # type: ignore

    @property
    def __class__(self) -> Type[Tensor]:
        return Tensor

    # Make it behave as if it inherited from ir.Value
    @property
    @lru_cache_ir()
    def type(self) -> ir.Type:
        return self.value.type

    @property
    def iterator(self) -> Union[Pointer, IntTuple]:
        return self._iterator

    @property
    def layout(self) -> Layout:
        return _cute_ir.get_layout(self.value)

    @property
    def shape(self) -> Shape:
        return self.layout.shape

    @property
    def stride(self) -> Stride:
        if isinstance(self.type, _cute_ir.ComposedLayoutType):
            raise ValueError(f"can't get stride from composed layout")
        return self.layout.stride

    @property
    def leading_dim(self) -> Union[int, Tuple[int], None]:
        """Get the leading dimension of this Tensor.

        :return: The index or indices of the first mode (from left to right) with stride 1
        :rtype: Union[int, Tuple[int], None]
        :returns:
            - int: Single leading dimension index if found
            - Tuple[int]: Tuple of indices for nested leading dimensions
            - None: If no leading dimension is found

        :postcondition: ``get(self.stride(), mode=self.leading_dim()) == 1 if self.leading_dim() != None else True``
        """
        return leading_dim(self.shape, self.stride)

    @property
    @lru_cache_ir()
    def element_type(self) -> Union[Type[Numeric], Type[IntTuple]]:
        return self._dtype

    @property
    @lru_cache_ir()
    def memspace(self) -> AddressSpace:
        if isinstance(self.iterator, Pointer):
            return self.iterator.memspace

        raise ValueError(f"{self} doesn't have memspace")

    @dsl_user_op
    def load(self, *, loc=None, ip=None) -> "TensorSSA":
        """Load tensor elements as a vector.

        Loads all elements of the tensor into a vector representation, assuming the tensor
        has a static shape and is in a memory space that supports load operations.

        :param loc: Source location for MLIR operation tracking, defaults to None
        :type loc: Optional[Location]
        :param ip: Insertion point for MLIR operation, defaults to None
        :type ip: Optional[InsertionPoint]
        :return: Vector representation of tensor elements
        :rtype: TensorSSA

        :raises ValueError: If tensor has dynamic layout
        :raises ValueError: If tensor memory space doesn't support load operations
        """
        if not is_static(self.shape):
            raise ValueError("dynamic layout doesn't support load")

        self._check_can_load_store()

        res_vect = _cute_ir.memref_load_vec(self.value, row_major=True, loc=loc, ip=ip)
        if self.element_type is Boolean:
            assert (
                res_vect.type.element_type == T.i8()
            ), f"Boolean tensor must be stored as i8 in memory, but got {res_vect.type.element_type}"
            zeros = full_like(self, 0, Int8, loc=loc, ip=ip)
            res_vect = arith.cmpi(
                arith.CmpIPredicate.ne, res_vect, zeros, loc=loc, ip=ip
            )
        return TensorSSA(res_vect, self.shape, self.element_type)

    @dsl_user_op
    def store(self, data: "TensorSSA", *, loc=None, ip=None):
        """Store vector data into tensor.

        Stores vector data into the tensor, assuming matching shapes and a memory space
        that supports store operations.

        :param data: Vector data to store into tensor
        :type data: TensorSSA
        :param loc: Source location for MLIR operation tracking, defaults to None
        :type loc: Optional[Location]
        :param ip: Insertion point for MLIR operation, defaults to None
        :type ip: Optional[InsertionPoint]

        :raises ValueError: If tensor has dynamic layout
        :raises ValueError: If tensor memory space doesn't support store operations
        :raises ValueError: If data shape doesn't match tensor shape
        """
        if not isinstance(data, TensorSSA):
            raise ValueError(f"Expects TensorSSA, but got {type(data)}")

        if not is_static(self.shape):
            raise ValueError("Dynamic layout doesn't support vectorized store")

        self._check_can_load_store()

        n_elems = size(self.shape, loc=loc, ip=ip)
        if n_elems != size(data.shape, loc=loc, ip=ip):
            raise ValueError(
                f"lhs and rhs must have the same shape, but got {self.shape} and {data.shape}"
            )

        elem_mlir_type = cutlass_arith.element_type(data.dtype.mlir_type)
        if cutlass_arith.is_narrow_precision(elem_mlir_type):
            if elem_mlir_type.width * n_elems % 32 != 0:
                raise ValueError(
                    f"narrow precision type must be 32-bit aligned vector, but got {elem_mlir_type} with {n_elems} elements"
                )

        # Implicit upcast to wider type
        new_data = self._cvt_to_dest(data, loc=loc, ip=ip)

        return _cute_ir.memref_store_vec(
            new_data, self.value, row_major=True, loc=loc, ip=ip
        )

    @dsl_user_op
    def fill(self, value: Numeric, *, loc=None, ip=None) -> None:
        """Fill tensor with a constant value.

        Fills all elements of the tensor with the specified value, assuming static size
        and supported memory space.

        :param value: Value to fill tensor with
        :type value: Union[int, float]
        :param loc: Source location for MLIR operation tracking, defaults to None
        :type loc: Optional[Location]
        :param ip: Insertion point for MLIR operation, defaults to None
        :type ip: Optional[InsertionPoint]

        :raises NotImplementedError: If tensor has dynamic size

        **Examples:**

        .. code-block:: python

            # Create tensor from numpy array
            b = np.random.randn(4, 8).astype(np.float32)
            tensor = from_dlpack(b)

            # Fill tensor with constant value
            tensor.fill(0.5)  # All elements become 0.5
        """
        self._check_can_load_store()

        sz = size(self, loc=loc, ip=ip)
        if type(sz) is not int:
            raise NotImplementedError(f"dynamic size is not supported: {self.type}")

        # Should we cast to destination type even with narrow cast?
        dst_type = self.element_type
        value = dst_type(value)

        self[None] = full(self.shape, fill_value=value, dtype=dst_type, loc=loc, ip=ip)

    def _check_can_load_store(self):
        if not isinstance(self.type, _cute_ir.MemRefType) or not self.memspace in (
            AddressSpace.rmem,
            AddressSpace.smem,
            AddressSpace.gmem,
            AddressSpace.generic,
        ):
            raise ValueError(f"{self} doesn't support load and store")

    def _check_can_dereference(self):
        # Check for sub-byte types and raise error if needed
        if self.element_type.width % 8 != 0 and self.element_type is not Boolean:
            raise ValueError(
                f"Sub-byte scalar dereference not supported for type {self.element_type}"
            )


@dsl_user_op
def print_tensor(
    tensor: Union[Tensor, "TensorSSA"], *, verbose: bool = False, loc=None, ip=None
):
    """Print content of the tensor in human readable format.

    Outputs the tensor data in a structured format showing both metadata
    and the actual data values. The output includes tensor type information,
    layout details, and a formatted array representation of the values.

    :param tensor: The tensor to print
    :type tensor: Tensor
    :param verbose: If True, includes additional debug information in the output
    :type verbose: bool
    :param loc: Source location where it's called, defaults to None
    :type loc: source location, optional
    :param ip: Insertion pointer for IR generation, defaults to None
    :type ip: insertion pointer, optional
    :raises NotImplementedError: If the tensor type doesn't support trivial dereferencing

    **Example output:**

    .. code-block:: text

        tensor(raw_ptr<@..., Float32, generic, align(4)> o (8,5):(5,1), data=
               [[-0.4326, -0.5434,  0.1238,  0.7132,  0.8042],
                [-0.8462,  0.9871,  0.4389,  0.7298,  0.6948],
                [ 0.3426,  0.5856,  0.1541,  0.2923,  0.6976],
                [-0.1649,  0.8811,  0.1788,  0.1404,  0.2568],
                [-0.2944,  0.8593,  0.4171,  0.8998,  0.1766],
                [ 0.8814,  0.7919,  0.7390,  0.4566,  0.1576],
                [ 0.9159,  0.7577,  0.6918,  0.0754,  0.0591],
                [ 0.6551,  0.1626,  0.1189,  0.0292,  0.8655]])
    """
    if isinstance(tensor, TensorSSA):
        tmp = make_fragment(tensor.shape, tensor.dtype)
        tmp.store(tensor)
        tensor = tmp

    if not isinstance(tensor.type, _cute_ir.MemRefType):
        raise NotImplementedError(
            f"printing {tensor} is not supported because it doesn't support trivial dereferencing. "
            f"Coordinate Tensor will be supported in the future."
        )

    tensor._check_can_load_store()  # type: ignore

    if tensor.element_type.is_integer:
        signed = tensor.element_type.signed
    else:
        signed = False

    _cute_ir.print_view(tensor.value, verbose=verbose, is_signed=signed, loc=loc, ip=ip)


####################################################################################################
#
# Core API
#
####################################################################################################


#
# Utilties
#


@lru_cache_ir()
def is_integer(a) -> bool:
    """Check if an object is static integer or dynamic integer"""
    return isinstance(a, (int, Integer)) or (
        isinstance(a, ir.Value)
        and isinstance(a.type, (ir.IntegerType, _cute_ir.ConstrainedIntType))
    )


def is_valid_leaf(a) -> bool:
    """
    Returns whether `a` has a type that is valid for a CuTe tuple's leaf.
    """
    return (
        is_integer(a)
        or (a is None)
        or isinstance(a, (ScaledBasis, Layout, ComposedLayout))
    )


def is_int_tuple(a) -> bool:
    if isinstance(a, tuple):
        return all([is_int_tuple(x) for x in a])
    else:
        return is_integer(a)


def is_static(x: Union[ir.Type, ir.Value, XTuple]) -> bool:
    """Check if a value is statically known at compile time.

    In CuTe, static values are those whose values are known at compile time,
    as opposed to dynamic values which are only known at runtime.

    :param x: The value to check
    :type x: Union[ir.Type, ir.Value, XTuple]
    :return: True if the value is static, False otherwise
    :rtype: bool
    :raises TypeError: If an unsupported type is provided
    """
    if isinstance(x, ir.Type):
        return _cute_ir.is_static(x)
    elif isinstance(x, tuple):
        return all(is_static(a) for a in x)
    # Can it be a static int?
    elif isinstance(x, Numeric):
        return False
    elif is_dynamic_expression(x):
        return _cute_ir.is_static(x.type)
    elif isinstance(x, (bool, int, float)) or x is None:
        return True
    elif isinstance(x, ScaledBasis):
        return x.is_static()
    else:
        raise TypeError(f"unsupported type {x}")


def has_underscore(a: XTuple) -> bool:
    if type(a) is tuple:
        return any([has_underscore(x) for x in a])
    else:
        return a is None


def has_scaled_basis(a: XTuple) -> bool:
    """Check if a tuple or its nested elements contain ScaledBasis objects.

    ScaledBasis objects are fundamental components in CuTe layouts,
    representing the basis vectors of coordinate systems.

    :param a: The tuple to check
    :type a: XTuple
    :return: True if the tuple contains ScaledBasis objects, False otherwise
    :rtype: bool
    """
    if type(a) is tuple:
        return any([has_scaled_basis(x) for x in a])
    else:
        return isinstance(a, ScaledBasis)


def _tuple_str(t: tuple) -> str:
    """
    Constructs a string representation of a python tuple without calling __repr__ on its elements.
    """

    def construct_inner_str(t) -> str:
        if not isinstance(t, tuple):
            return pretty_str(t)
        res = ""
        l = len(t)
        for i in range(l):
            res += pretty_str(t[i])
            if i < l - 1:
                res += ","
        return res

    res = "(" + construct_inner_str(t) + ")"
    return res


def pretty_str(arg) -> str:
    """
    Constructs a concise readable pretty string.
    """
    if isinstance(arg, tuple):
        # _tuple_str for tuples
        return _tuple_str(arg)
    elif arg is None:
        # We interpret None as underscores for slicers
        return "_"
    else:
        # Fallback to __str__
        return arg.__str__()


@dsl_user_op
def printf(*args, loc=None, ip=None) -> None:
    """
    Print a value or a list of values.

    It supports c-style printf format as well:

    .. code-block:: python

        a = cute.make_layout(shape=(10, 10), stride=(10, 1))
        b = cutlass.Float32(1.234)
        cute.printf(a, b)
        cute.printf("a={}, b={}", a, b)
        cute.printf("a={}, b=%.2f", a, b)

    :param args: List of values to print
    :type args: list
    :param loc: Source location where it's called, defaults to None
    :type loc: source location, optional
    :param ip: Insertion pointer, defaults to None
    :type ip: insertion pointer, optional
    :raises ValueError: If no arguments are provided or if an unsupported argument type is passed
    """

    if len(args) == 0:
        raise ValueError("expects at least one argument to print")

    if isinstance(args[0], str):
        fmt = args[0] + "\n"
        args = args[1:]
    else:
        fmt = "{}" + ", {}" * (len(args) - 1) + "\n"

    def process_arg(arg):
        arg0 = arg.value if isinstance(arg, Numeric) else arg

        if isinstance(arg0, ir.Value):
            return arg0
        elif isinstance(arg0, bool):
            return const(arg0, Boolean)
        elif isinstance(arg0, int):
            return const(arg0, Int32)
        elif isinstance(arg0, float):
            return const(arg0, Float32)
        elif has_underscore(arg0):
            # Assume it's a coordinate
            return _pack_coord(arg0)
        elif has_scaled_basis(arg0):
            # Assume it's a stride
            return _pack_stride(arg0)
        elif isinstance(arg0, tuple):
            # Assume it's an int_tuple
            return _pack_int_tuple(arg0)
        elif isinstance(arg0, (_Tensor, _Pointer)):
            return arg0.value
        else:
            raise TypeError(f"unsupported argument type in printf, got {type(arg)}")

    args = [process_arg(a) for a in args]
    _cute_ir.print_(args, fmt=fmt, loc=loc, ip=ip)


@dsl_user_op
def front(input, *, loc=None, ip=None):
    """Recursively get the first element of input.

    This function traverses a hierarchical structure (like a layout or tensor)
    and returns the first element at the deepest level. It's particularly useful
    for accessing the first stride value in a layout to determine properties like
    majorness.

    :param input: The hierarchical structure to traverse
    :type input: Union[Tensor, Layout, Stride]
    :param loc: Source location where it's called, defaults to None
    :type loc: source location, optional
    :param ip: Insertion pointer for IR generation, defaults to None
    :type ip: insertion pointer, optional
    :return: The first element at the deepest level of the input structure
    :rtype: Union[int, float, bool, ir.Value]
    """
    if rank(input) == 1 and depth(input) == 0:
        return input
    else:
        return front(get(input, mode=[0], loc=loc, ip=ip), loc=loc, ip=ip)


@dsl_user_op
def is_major(mode, stride: Stride, *, loc=None, ip=None) -> bool:
    """
    Check whether a mode in stride is the major mode.
    """
    first_stride = front(get(stride, mode=[mode], loc=loc, ip=ip), loc=loc, ip=ip)
    if is_dynamic_expression(first_stride):
        return False
    return True if first_stride == 1 else False


def leading_dim(shape: Shape, stride: Stride) -> Union[int, Tuple[int, ...], None]:
    """
    Find the leading dimension of a shape and stride.

    :param shape: The shape of the tensor or layout
    :type shape: Shape
    :param stride: The stride of the tensor or layout
    :type stride: Stride
    :return: The leading dimension index or indices
    :rtype: Union[int, Tuple[int, ...], None]

    The return value depends on the stride pattern:

        * If a single leading dimension is found, returns an integer index
        * If nested leading dimensions are found, returns a tuple of indices
        * If no leading dimension is found, returns None
    """

    def pred_fn(val, pos):
        # skip dynamic values which can't be compared
        # find the candidate target val, stride at this position is 1
        if (not is_dynamic_expression(val)) and (val == 1):
            # extract the shape at this position
            mode = [pos] if isinstance(pos, int) else list(pos)
            s = get(shape, mode)
            if is_dynamic_expression(s) or s != 1:
                # shape at this position is dynamic value or not 1
                # we found the leading dimension
                return True
        return False

    return find_if(stride, pred_fn=pred_fn)


@dsl_user_op
def find_if(
    t: Union[tuple, ir.Value, int],
    pred_fn: Callable[[int, Tuple[int, ...]], bool],
    *,
    loc=None,
    ip=None,
) -> Union[int, Tuple[int, ...], None]:
    """Find the first position in t where pred_fn(val, pos) returns True.

    :param t: The search space
    :type t: Union[tuple, ir.Value, int]
    :param pred_fn: A callable object (lambda, function, etc.) that predicates the value and position in t.
                    It takes the current leaf value and position, returns True if the value or position is satisfied.
    :type pred_fn: Callable[[int, Tuple[int, ...]], bool]
    :return: Index if found at top level, tuple of indices showing nested position, or None if not found
    :rtype: Union[int, Tuple[int, ...], None]

    **Examples:**

    .. code-block:: python

        # Find the first position of x in t
        t = (3, 4)
        find_if(t, pred_fn=lambda val, pos: val == x)

    .. code-block:: python

        # find the leading dimension
        shape = (3, 4)
        stride = (4, 1)
        # Find value 1 in stride where the corresponding shape is not 1
        def pred_fn(val, pos):
            mode = [pos] if isinstance(pos, int) else list(pos)
            return val == 1 and get(shape, mode) != 1
        find_if(stride, pred_fn=pred_fn)
    """

    def _find_if_impl(curr, pos, *, loc=None, ip=None):
        if isinstance(curr, tuple):
            # Recursively search nested tuple
            for i in range(rank(curr)):
                sub_curr = get(curr, mode=[i], loc=loc, ip=ip)
                sub_pos = (pos, i) if isinstance(pos, int) else pos + (i,)
                res_pos = _find_if_impl(sub_curr, sub_pos, loc=loc, ip=ip)
                if res_pos is not None:
                    return res_pos
        else:
            # For leaf values, check if it matches x
            if pred_fn(curr, pos):
                return pos
        return None

    def _check_pred_fn():
        if not callable(pred_fn):
            raise TypeError(f"pred_fn must be callable, but got {type(pred_fn)}")
        signature = inspect.signature(pred_fn)
        if len(signature.parameters) != 2:
            raise ValueError(
                f"pred_fn must have two parameters (value, pos), but got {len(signature.parameters)}"
            )

    _check_pred_fn()

    for i in range(rank(t)):
        curr = get(t, mode=[i], loc=loc, ip=ip)
        res_pos = _find_if_impl(curr, i, loc=loc, ip=ip)
        if res_pos is not None:
            return res_pos
    return None


@dsl_user_op
def find(
    t: Union[tuple, ir.Value, int],
    x: int,
    *,
    loc=None,
    ip=None,
) -> Union[int, Tuple[int, ...], None]:
    """Find the first position of a value ``x`` in a hierarchical structure ``t``.

    Searches for the first occurrence of x in t, optionally excluding positions
    where a comparison value matches. The search can traverse nested structures
    and returns either a single index or a tuple of indices for nested positions.

    :param t: The search space
    :type t: Union[tuple, ir.Value, int]
    :param x: The static integer x to search for
    :type x: int
    :return: Index if found at top level, tuple of indices showing nested position, or None if not found
    :rtype: Union[int, Tuple[int, ...], None]
    """
    if not isinstance(x, int):
        raise TypeError(f"find() requires a static x to search for, but got {x}")

    def pred_fn(val, pos):
        # Skip dynamic values which can't be compared
        return not is_dynamic_expression(val) and val == x

    return find_if(t, pred_fn=pred_fn, loc=loc, ip=ip)


def transform_leaf(f, *args):
    """
    Apply a function to the leaf nodes of nested tuple structures.

    This function traverses nested tuple structures in parallel and applies the function f
    to corresponding leaf nodes. All input tuples must have the same nested structure.

    :param f: Function to apply to leaf nodes
    :type f: Callable
    :param args: One or more nested tuple structures with matching profiles
    :return: A new nested tuple with the same structure as the inputs, but with leaf values transformed by f
    :raises TypeError: If the input tuples have different nested structures

    Example:

    .. code-block:: python

        >>> transform_leaf(lambda x: x + 1, (1, 2))
        (2, 3)
        >>> transform_leaf(lambda x, y: x + y, (1, 2), (3, 4))
        (4, 6)
        >>> transform_leaf(lambda x: x * 2, ((1, 2), (3, 4)))
        ((2, 4), (6, 8))
    """
    if all(isinstance(t, tuple) for t in args):
        return tuple(transform_leaf(f, *_args) for _args in zip(*args))
    elif all(not isinstance(t, tuple) for t in args):
        return f(*args)
    else:
        raise TypeError(f"profile of input tuples doesn't match: {args}")


@dsl_user_op
def assume(src, divby=None, *, loc=None, ip=None):
    if divby is None:
        return src

    if isinstance(src, Integer):
        width = type(src).width
        src_val = src.ir_value()
    else:
        width = src.type.width
        src_val = src

    res_ty = _cute_ir.ConstrainedIntType.get(divby, width)
    assumed_val = _cute_ir.assume(res_ty, src_val, loc=loc, ip=ip)
    return type(src)(IntValue(_pack_int_tuple(assumed_val, loc=loc, ip=ip)))


@dsl_user_op
def make_swizzle(b, m, s, *, loc=None, ip=None):
    # canonicalize to <0, 4, 3> for identity swizzle (as compiler assumes <0, 4, 3>)
    if b == 0:
        m, s = 4, 3
    ty = ir.Type.parse(f'!cute.swizzle<"S<{b},{m},{s}>">')
    return Swizzle(_cute_ir.static(ty, loc=loc, ip=ip))


#
# Tuple API (also used by layouts and tensors)
#


def depth(a: Union[XTuple, Layout, "ComposedLayout"]) -> int:
    """Returns the depth (nesting level) of a tuple, layout, or tensor.

    The depth of a tuple is the maximum depth of its elements plus 1.
    For an empty tuple, the depth is 1. For layouts and tensors, the depth
    is determined by the depth of their shape. For non-tuple values (e.g., integers),
    the depth is considered 0.

    :param a: The object whose depth is to be determined
    :type a: Union[XTuple, Layout, ComposedLayout, Tensor, Any]
    :return: The depth of the input object
    :rtype: int

    Example:

    .. code-block:: python

        >>> depth(1)
        0
        >>> depth((1, 2))
        1
        >>> depth(((1, 2), (3, 4)))
        2
    """
    if type(a) is tuple:
        if not a:
            return 1
        return max(depth(x) for x in a) + 1
    elif isinstance(a, (Layout, ComposedLayout, Tensor)):
        return depth(a.shape)
    else:
        return 0


@lru_cache_ir()
def rank(a: Union[XTuple, Layout, "ComposedLayout"]) -> int:
    """Returns the rank (dimensionality) of a tuple, layout, or tensor.

    The rank of a tuple is its length. For layouts and tensors, the rank is
    determined by the rank of their shape. For non-tuple values (e.g., integers),
    the rank is considered 1 for convenience.

    :param a: The object whose rank is to be determined
    :type a: Union[XTuple, Layout, ComposedLayout, Tensor, Any]
    :return: The rank of the input object
    :rtype: int

    This function is used in layout algebra to determine the dimensionality
    of tensors and layouts for operations like slicing and evaluation.
    """
    if isinstance(a, tuple):
        return len(a)
    elif isinstance(a, (Layout, ComposedLayout, Tensor)):
        return rank(a.shape)
    elif depth(a) == 0:
        return 1
    else:
        raise TypeError(f"unsupported type in rank, got {type(a)}")


def is_congruent(
    a: Union[XTuple, Layout, ComposedLayout, Tensor],
    b: Union[XTuple, Layout, ComposedLayout, Tensor],
) -> bool:
    """
    Returns whether a is congruent to b.

    Congruence is an equivalence relation between hierarchical structures.

    Two objects are congruent if:
    * They have the same rank, AND
    * They are both non-tuple values, OR
    * They are both tuples AND all corresponding elements are congruent.

    Congruence requires type matching at each level -- scalar values match with
    scalar values, and tuples match with tuples of the same rank.

    :param a: First object to compare
    :type a: Union[XTuple, Layout, ComposedLayout, Tensor]
    :param b: Second object to compare
    :type b: Union[XTuple, Layout, ComposedLayout, Tensor]
    :return: True if a and b are congruent, False otherwise
    :rtype: bool
    """
    if isinstance(a, (Layout, ComposedLayout, Tensor)):
        a = a.shape
    if isinstance(b, (Layout, ComposedLayout, Tensor)):
        b = b.shape
    if isinstance(a, tuple) and isinstance(b, tuple):
        return (len(a) == len(b)) and all(is_congruent(x, y) for x, y in zip(a, b))
    if isinstance(a, tuple) or isinstance(b, tuple):
        return False
    return True


def is_weakly_congruent(
    a: Union[XTuple, Layout, ComposedLayout, Tensor],
    b: Union[XTuple, Layout, ComposedLayout, Tensor],
) -> bool:
    """
    Returns whether a is weakly congruent to b.

    Weak congruence is a partial order on hierarchical structures.

    Object X is weakly congruent to object Y if:
    * X is a non-tuple value, OR
    * X and Y are both tuples of the same rank AND all corresponding elements are weakly congruent.

    Weak congruence allows scalar values to match with tuples, making it useful
    for determining whether an object has a hierarchical structure "up to" another.

    :param a: First object to compare
    :type a: Union[XTuple, Layout, ComposedLayout, Tensor]
    :param b: Second object to compare
    :type b: Union[XTuple, Layout, ComposedLayout, Tensor]
    :return: True if a and b are weakly congruent, False otherwise
    :rtype: bool
    """
    if isinstance(a, (Layout, ComposedLayout, Tensor)):
        a = a.shape
    if isinstance(b, (Layout, ComposedLayout, Tensor)):
        b = b.shape
    if not isinstance(a, tuple):
        return True
    if isinstance(a, tuple) and isinstance(b, tuple):
        return (len(a) == len(b)) and all(
            is_weakly_congruent(x, y) for x, y in zip(a, b)
        )
    if isinstance(a, tuple) or isinstance(b, tuple):
        return False
    return True


@overload
def get(input: Shape, mode, *, loc=None, ip=None) -> Shape: ...
@overload
def get(input: Stride, mode, *, loc=None, ip=None) -> Stride: ...
@overload
def get(input: Coord, mode, *, loc=None, ip=None) -> Coord: ...
@overload
def get(input: IntTuple, mode, *, loc=None, ip=None) -> IntTuple: ...
@overload
def get(input: Tile, mode, *, loc=None, ip=None) -> Tile: ...
@overload
def get(input: Layout, mode, *, loc=None, ip=None) -> Layout: ...
@overload
def get(input: ComposedLayout, mode, *, loc=None, ip=None) -> ComposedLayout: ...


@dsl_user_op
def get(input, mode: List[int], *, loc=None, ip=None):
    """Extract a specific element or sub-layout from a layout or tuple.

    This function recursively traverses the input according to the mode indices,
    extracting the element at the specified path. For layouts, this operation
    corresponds to extracting a specific sub-layout.

    :param input: The input layout or tuple to extract from
    :type input: Layout, ComposedLayout, tuple
    :param mode: Indices specifying the path to traverse for extraction
    :type mode: List[int]
    :param loc: Source location for MLIR, defaults to None
    :type loc: optional
    :param ip: Insertion point, defaults to None
    :type ip: optional
    :return: The extracted element or sub-layout
    :rtype: Layout, ComposedLayout, or element type
    :raises ValueError: If any index in mode is out of range
    :raises TypeError: If mode contains non-integer elements or if input has unsupported type

    :postcondition: ``get(t, mode=find(x,t)) == x if find(x,t) != None else True``

    **Examples:**

    .. code-block:: python

        layout = make_layout(((4, 8), (16, 1), 8), stride=((1, 4), (32, 0), 512))
        sub_layout = get(layout, mode=[0, 1])   # 8:4
        sub_layout = get(layout, mode=[1])      # (16, 1):(32, 0)
    """
    # Empty mode returns input and terminates the recursive call
    if not mode:
        return input

    if rank(input) <= mode[0]:
        raise ValueError(
            f"elements in mode must be less than rank({input}), got {mode}"
        )

    if depth(input) == 0:
        return input
    elif isinstance(input, tuple):
        if not isinstance(mode[0], int):
            raise TypeError(
                f"invalid element in mode, expects int, got {type(mode[0])}"
            )
        return get(input[mode[0]], mode=mode[1:])
    else:
        if not isinstance(input, (Layout, ComposedLayout)):
            raise TypeError(f"unsupported type of input, got {type(input)}")
        return _cute_ir.get(
            input.type.get_op_res_type(mode=mode), input, mode=mode, loc=loc, ip=ip
        )


@overload
def select(input: Shape, mode, *, loc=None, ip=None) -> Shape: ...
@overload
def select(input: Stride, mode, *, loc=None, ip=None) -> Stride: ...
@overload
def select(input: Coord, mode, *, loc=None, ip=None) -> Coord: ...
@overload
def select(input: IntTuple, mode, *, loc=None, ip=None) -> IntTuple: ...
@overload
def select(input: Tile, mode, *, loc=None, ip=None) -> Tile: ...
@overload
def select(input: Layout, mode, *, loc=None, ip=None) -> Layout: ...
@overload
def select(input: ComposedLayout, mode, *, loc=None, ip=None) -> ComposedLayout: ...


@dsl_user_op
def select(input, mode: List[int], *, loc=None, ip=None):
    """Select modes from input.

    :param input: Input to select from
    :type input: Layout, ComposedLayout, tuple
    :param mode: Indices specifying which dimensions or elements to select
    :type mode: List[int]
    :param loc: Source location for MLIR, defaults to None
    :type loc: optional
    :param ip: Insertion point, defaults to None
    :type ip: optional
    :return: A new instance with selected dimensions/elements
    :rtype: Layout, ComposedLayout, tuple
    :raises ValueError: If any index in mode is out of range
    :raises TypeError: If the input type is invalid

    **Examples:**

    .. code-block:: python

        # Select specific dimensions from a layout
        layout = make_layout((4, 8, 16), stride=(32, 4, 1))
        selected = select(layout, mode=[0, 2])  # Select mode 0 and mode 2
        # Result: (4, 16):(32, 1)

        # Select elements from a tuple
        t = (1, 2, 3, 4, 5)
        selected = select(t, mode=[0, 2, 4])  # Select mode 0, mode 2, and mode 4
        # Result: (1, 3, 5)
    """
    if any((not isinstance(i, int)) or (i >= rank(input)) for i in mode):
        raise ValueError(
            f"invalid mode element for input of rank {rank(input)}, got {mode=}"
        )

    if isinstance(input, tuple):
        return tuple(input[i] for i in mode)

    if not isinstance(input, (Layout, ComposedLayout)):
        raise TypeError(f"unsupported type of input, got {type(input)}")

    return _cute_ir.select(input, mode=mode, loc=loc, ip=ip)


@overload
def group_modes(input: Shape, begin: int, end: int, *, loc=None, ip=None) -> Shape: ...
@overload
def group_modes(
    input: Stride, begin: int, end: int, *, loc=None, ip=None
) -> Stride: ...
@overload
def group_modes(input: Coord, begin: int, end: int, *, loc=None, ip=None) -> Coord: ...
@overload
def group_modes(
    input: IntTuple, begin: int, end: int, *, loc=None, ip=None
) -> IntTuple: ...
@overload
def group_modes(input: Tile, begin: int, end: int, *, loc=None, ip=None) -> Tile: ...
@overload
def group_modes(
    input: Layout, begin: int, end: int, *, loc=None, ip=None
) -> Layout: ...
@overload
def group_modes(
    input: ComposedLayout, begin: int, end: int, *, loc=None, ip=None
) -> ComposedLayout: ...
@overload
def group_modes(
    input: Tensor, begin: int, end: int, *, loc=None, ip=None
) -> Tensor: ...


@dsl_user_op
def group_modes(input, begin: int, end: int = -1, *, loc=None, ip=None):
    """Group modes of a hierarchical tuple or layout into a single mode.

    This function groups a range of modes from the input object into a single mode,
    creating a hierarchical structure. For tuples, it creates a nested tuple containing
    the specified range of elements. For layouts and other CuTe objects, it creates
    a hierarchical representation where the specified modes are grouped together.

    :param input: Input object to group modes from (layout, tuple, etc.)
    :type input: Layout, ComposedLayout, tuple, Shape, Stride, etc.
    :param beg: Beginning index of the range to group (inclusive)
    :type beg: int
    :param end: Ending index of the range to group (exclusive)
    :type end: int
    :param loc: Source location for MLIR, defaults to None
    :type loc: optional
    :param ip: Insertion point, defaults to None
    :type ip: optional
    :return: A new object with the specified modes grouped
    :rtype: Same type as input with modified structure

    **Examples:**

    .. code-block:: python

        # Group modes in a tuple
        t = (2, 3, 4, 5)
        grouped = group_modes(t, 1, 3)  # (2, (3, 4), 5)

        # Group modes in a layout
        layout = make_layout((2, 3, 4, 5))
        grouped_layout = group_modes(layout, 1, 3)  # Layout with shape (2, (3, 4), 5)

        # Group modes in a shape
        shape = make_shape(2, 3, 4, 5)
        grouped_shape = group_modes(shape, 0, 2)  # Shape ((2, 3), 4, 5)
    """
    if depth(input) == 0 and is_integer(input):
        return (input,)
    if isinstance(input, tuple):
        return (*input[:begin], (input[begin:end]), *input[end:])
    return _cute_ir.group_modes(
        input.value if isinstance(input, Tensor) else input, begin, end, loc=loc, ip=ip
    )


@overload
def slice_(src: Shape, coord: Coord, *, loc=None, ip=None) -> Shape: ...
@overload
def slice_(src: Stride, coord: Coord, *, loc=None, ip=None) -> Stride: ...
@overload
def slice_(src: Coord, coord: Coord, *, loc=None, ip=None) -> Coord: ...
@overload
def slice_(src: IntTuple, coord: Coord, *, loc=None, ip=None) -> IntTuple: ...
@overload
def slice_(src: Tile, coord: Coord, *, loc=None, ip=None) -> Tile: ...
@overload
def slice_(src: Layout, coord: Coord, *, loc=None, ip=None) -> Layout: ...
@overload
def slice_(
    src: ComposedLayout, coord: Coord, *, loc=None, ip=None
) -> ComposedLayout: ...
@overload
def slice_(src: Tensor, coord: Coord, *, loc=None, ip=None) -> Tensor: ...


@dsl_user_op
def slice_(src, coord: Coord, *, loc=None, ip=None):
    """Perform a slice operation on a source object using the given coordinate.

    This function implements CuTe's slicing operation which extracts a subset of elements
    from a source object (tensor, layout, etc.) based on a coordinate pattern. The slice
    operation preserves the structure of the source while selecting specific elements.

    :param src: Source object to be sliced (tensor, layout, tuple, etc.)
    :type src: Union[Tensor, Layout, IntTuple, Value]
    :param coord: Coordinate pattern specifying which elements to select
    :type coord: Coord
    :param loc: Source location information, defaults to None
    :type loc: Optional[Location]
    :param ip: Insertion point for IR generation, defaults to None
    :type ip: Optional[InsertionPoint]
    :return: A new object containing the sliced elements
    :rtype: Union[Tensor, Layout, IntTuple, tuple]
    :raises ValueError: If the coordinate pattern is incompatible with source

    **Examples:**

    .. code-block:: python

        # Layout slicing
        layout = make_layout((4,4))

        # Select 1st index of first mode and keep all elements in second mode
        sub_layout = slice_(layout, (1, None))

    .. code-block:: python

        # Basic tensor slicing
        tensor = make_tensor(...)           # Create a 2D tensor

        # Select 1st index of first mode and keep all elements in second mode
        sliced = slice_(tensor, (1, None))

    .. code-block:: python

        # Select 2nd index of second mode and keep all elements in first mode
        sliced = slice_(tensor, (None, 2))

    Note:
        - `None` represents keeping all elements in that mode
        - Slicing preserves the layout/structure of the original object
        - Can be used for:
          * Extracting sub-tensors/sub-layouts
          * Creating views into data
          * Selecting specific patterns of elements
    """

    def lift_slice(a, b):
        if isinstance(a, tuple):
            if (not isinstance(b, tuple)) or (len(a) != len(b)):
                raise ValueError("coord must be weakly congruent to src in slice_")
            return reduce(
                lambda p, q: p + q, (lift_slice(x, y) for x, y in zip(a, b)), ()
            )
        elif a is None:
            return (b,)
        else:
            return ()

    if is_integer(src) or isinstance(src, tuple):
        if isinstance(coord, tuple):
            if (not isinstance(src, tuple)) or (len(coord) != len(src)):
                raise ValueError("coord must be weakly congruent to src in slice_")
            return reduce(
                lambda p, q: p + q, (lift_slice(x, y) for x, y in zip(coord, src)), ()
            )
        elif coord is None:
            return src
        else:
            return ()

    res_type = None
    if isinstance(src, Tensor):
        res_type = src.element_type
        src = src.value
    coord_val = _pack_coord(coord, loc=loc, ip=ip)
    res = _cute_ir.slice(input=src, coord=coord_val, loc=loc, ip=ip)
    return _Tensor(res, dtype=res_type) if isinstance(res, _Tensor) else res


@overload
def dice(src: Shape, coord: Coord, *, loc=None, ip=None) -> Shape: ...
@overload
def dice(src: Stride, coord: Coord, *, loc=None, ip=None) -> Stride: ...
@overload
def dice(src: Coord, coord: Coord, *, loc=None, ip=None) -> Coord: ...
@overload
def dice(src: IntTuple, coord: Coord, *, loc=None, ip=None) -> IntTuple: ...
@overload
def dice(src: Tile, coord: Coord, *, loc=None, ip=None) -> Tile: ...
@overload
def dice(src: Layout, coord: Coord, *, loc=None, ip=None) -> Layout: ...
@overload
def dice(src: ComposedLayout, coord: Coord, *, loc=None, ip=None) -> ComposedLayout: ...


@dsl_user_op
@lru_cache_ir()
def dice(src, dicer, *, loc=None, ip=None):
    """Keep modes in input when it is paired with an integer in dicer.

    This function performs dicing operation on the input based on the dicer coordinate.
    Dicing is a fundamental operation in CuTe that allows selecting specific modes from
    a tensor or layout based on a coordinate pattern.

    :param dicer: A static coordinate indicating how to dice the input
    :type dicer: Coord
    :param input: The operand to be diced on
    :type input: Union[IntTuple, Shape, Stride, Coord, Layout, ComposedLayout]
    :param loc: Source location information, defaults to None
    :type loc: Optional[Location]
    :param ip: Insertion point for IR generation, defaults to None
    :type ip: Optional[InsertionPoint]
    :return: The diced result with selected modes from the input
    :rtype: Union[IntTuple, Shape, Stride, Coord, Layout, ComposedLayout]
    :raises TypeError: If dicer has an unsupported type
    :raises ValueError: If input is not provided

    **Examples:**

    .. code-block:: python

        # Basic dicing of a layout
        layout = make_layout((32,16,8))

        # Keep only first and last modes
        diced = dice((1,None,1), layout)

    Note:
        - The dicer coordinate must be static
        - Use underscore (_) to remove a mode
    """
    if not is_static(dicer):
        raise ValueError(f"expects dicer to be static, but got {dicer}")

    def lift_dice(a, b):
        if isinstance(a, tuple):
            if (not isinstance(b, tuple)) or (len(a) != len(b)):
                raise ValueError("dicer must be weakly congruent to input in dice")
            return reduce(
                lambda p, q: p + q, (lift_dice(x, y) for x, y in zip(a, b)), ()
            )
        elif a is None:
            return ()
        else:
            return (b,)

    if is_integer(src) or isinstance(src, tuple):
        if isinstance(dicer, tuple):
            if (not isinstance(src, tuple)) or (len(dicer) != len(src)):
                raise ValueError("dicer must be weakly congruent to src in dice")
            return reduce(
                lambda p, q: p + q, (lift_dice(x, y) for x, y in zip(dicer, src)), ()
            )
        elif dicer is None:
            return ()
        else:
            return src

    dicer_val = _pack_coord(dicer, loc=loc, ip=ip)
    return _cute_ir.dice(src, dicer_val.type.attribute, loc=loc, ip=ip)


def wrap(x) -> tuple:
    """
    Wraps the input into a tuple if not a tuple.
    """
    if isinstance(x, tuple):
        return x
    return (x,)


def _extend(func, input, elem, up_to_rank, loc, ip):
    if input is None:
        raise ValueError(f"No input provided for input")

    if isinstance(input, (Layout, ComposedLayout)):
        if elem is None:
            elem = make_layout(1)
        elif not isinstance(elem, Layout):
            raise TypeError(f"Input type of elem ({type(elem)}) is not accepted!")
        N = rank(input) + 1 if up_to_rank is None else up_to_rank
        return func(N, input, elem, loc=loc, ip=ip)

    if is_valid_leaf(input) or isinstance(input, tuple):
        if elem is None:
            elem = 1
        if (not isinstance(elem, tuple)) and (not is_valid_leaf(elem)):
            raise TypeError(f"Input type of elem ({type(elem)}) is not accepted!")

        input = wrap(input)
        repeat_cnt = 1 if up_to_rank is None else up_to_rank - rank(input)
        if repeat_cnt == 0:
            return input
        elif repeat_cnt < 0:
            raise ValueError(f"up_to_rank must be >= rank(input)")
        else:
            if func is _cute_ir.prepend_to_rank:
                return (elem,) * repeat_cnt + input
            else:
                return input + (elem,) * repeat_cnt

    raise TypeError(f"invalid type for input, got {type(input)}")


@overload
def prepend(
    input: Shape, elem: Shape, up_to_rank=None, *, loc=None, ip=None
) -> Shape: ...
@overload
def prepend(
    input: Stride, elem: Stride, up_to_rank=None, *, loc=None, ip=None
) -> Stride: ...
@overload
def prepend(
    input: Coord, elem: Coord, up_to_rank=None, *, loc=None, ip=None
) -> Coord: ...
@overload
def prepend(
    input: IntTuple, elem: IntTuple, up_to_rank=None, *, loc=None, ip=None
) -> IntTuple: ...
@overload
def prepend(input: Tile, elem: Tile, up_to_rank=None, *, loc=None, ip=None) -> Tile: ...
@overload
def prepend(
    input: Layout, elem: Layout, up_to_rank=None, *, loc=None, ip=None
) -> Layout: ...
@overload
def prepend(
    input: ComposedLayout, elem: Layout, up_to_rank=None, *, loc=None, ip=None
) -> ComposedLayout: ...


@dsl_user_op
def prepend(input, elem, up_to_rank: Union[None, int] = None, *, loc=None, ip=None):
    """Extend input to rank up_to_rank by prepending elem in front of input.

    This function extends the input object by prepending elements to reach a desired rank.
    It supports various CuTe types including shapes, layouts, tensors etc.

    :param input: Source to be prepended to
    :type input: Union[Shape, Stride, Coord, IntTuple, Tile, Layout, ComposedLayout, Tensor]
    :param elem: Element to prepend to input
    :type elem: Union[Shape, Stride, Coord, IntTuple, Tile, Layout]
    :param up_to_rank: The target rank after extension, defaults to None
    :type up_to_rank: Union[None, int], optional
    :param loc: Source location for MLIR, defaults to None
    :type loc: Optional[Location]
    :param ip: Insertion point, defaults to None
    :type ip: Optional[InsertionPoint]
    :return: The extended result with prepended elements
    :rtype: Union[Shape, Stride, Coord, IntTuple, Tile, Layout, ComposedLayout, Tensor]
    :raises ValueError: If up_to_rank is less than input's current rank
    :raises TypeError: If input or elem has unsupported type

    **Examples:**

    .. code-block:: python

        # Prepend to a Shape
        shape = (4,4)
        prepend(shape, 2)                   # Returns (2,4,4)

        # Prepend to a Layout
        layout = make_layout((8,8))
        prepend(layout, make_layout((2,)))  # Returns (2,8,8):(1,1,8)

        # Prepend with target rank
        coord = (1,1)
        prepend(coord, 0, up_to_rank=4)     # Returns (0,0,1,1)
    """
    return _extend(_cute_ir.prepend_to_rank, input, elem, up_to_rank, loc=loc, ip=ip)


@overload
def append(
    input: Shape, elem: Shape, up_to_rank=None, *, loc=None, ip=None
) -> Shape: ...
@overload
def append(
    input: Stride, elem: Stride, up_to_rank=None, *, loc=None, ip=None
) -> Stride: ...
@overload
def append(
    input: Coord, elem: Coord, up_to_rank=None, *, loc=None, ip=None
) -> Coord: ...
@overload
def append(
    input: IntTuple, elem: IntTuple, up_to_rank=None, *, loc=None, ip=None
) -> IntTuple: ...
@overload
def append(input: Tile, elem: Tile, up_to_rank=None, *, loc=None, ip=None) -> Tile: ...
@overload
def append(
    input: Layout, elem: Layout, up_to_rank=None, *, loc=None, ip=None
) -> Layout: ...
@overload
def append(
    input: ComposedLayout, elem: Layout, up_to_rank=None, *, loc=None, ip=None
) -> ComposedLayout: ...


@dsl_user_op
def append(input, elem, up_to_rank: Union[None, int] = None, *, loc=None, ip=None):
    """Extend input to rank up_to_rank by appending elem to the end of input.

    This function extends the input object by appending elements to reach a desired rank.
    It supports various CuTe types including shapes, layouts, tensors etc.

    :param input: Source to be appended to
    :type input: Union[Shape, Stride, Coord, IntTuple, Tile, Layout, ComposedLayout, Tensor]
    :param elem: Element to append to input
    :type elem: Union[Shape, Stride, Coord, IntTuple, Tile, Layout]
    :param up_to_rank: The target rank after extension, defaults to None
    :type up_to_rank: Union[None, int], optional
    :param loc: Source location for MLIR, defaults to None
    :type loc: Optional[Location]
    :param ip: Insertion point, defaults to None
    :type ip: Optional[InsertionPoint]
    :return: The extended result with appended elements
    :rtype: Union[Shape, Stride, Coord, IntTuple, Tile, Layout, ComposedLayout, Tensor]
    :raises ValueError: If up_to_rank is less than input's current rank
    :raises TypeError: If input or elem has unsupported type

    **Examples:**

    .. code-block:: python

        # Append to a Shape
        shape = (4,4)
        append(shape, 2)                   # Returns (4,4,2)

        # Append to a Layout
        layout = make_layout((8,8))
        append(layout, make_layout((2,)))  # Returns (8,8,2):(1,8,1)

        # Append with target rank
        coord = (1,1)
        append(coord, 0, up_to_rank=4)     # Returns (1,1,0,0)

    Note:
        - The function preserves the structure of the input while extending it
        - Can be used to extend tensors, layouts, shapes and other CuTe types
        - When up_to_rank is specified, fills remaining positions with elem
        - Useful for tensor reshaping and layout transformations
    """
    return _extend(_cute_ir.append_to_rank, input, elem, up_to_rank, loc=loc, ip=ip)


@dsl_user_op
def prepend_ones(
    t: Tensor, up_to_rank: Union[None, int] = None, *, loc=None, ip=None
) -> Tensor:
    return make_tensor(
        t.iterator, prepend(t.layout, make_layout(1), up_to_rank), loc=loc, ip=ip
    )


@dsl_user_op
def append_ones(
    t: Tensor, up_to_rank: Union[None, int] = None, *, loc=None, ip=None
) -> Tensor:
    return make_tensor(
        t.iterator, append(t.layout, make_layout(1), up_to_rank), loc=loc, ip=ip
    )


def repeat_like(x, target):
    """Creates an object congruent to target and filled with x.

    This function recursively creates a nested tuple structure that matches the structure
    of the target, with each leaf node filled with the value x.

    :param x: The value to fill the resulting structure with
    :type x: Any
    :param target: The structure to mimic
    :type target: Union[tuple, Any]
    :return: A structure matching target but filled with x
    :rtype: Union[tuple, Any]

    **Examples:**

    .. code-block:: python

        repeat_like(0, (1, 2, 3))      # Returns (0, 0, 0)
        repeat_like(1, ((1, 2), 3))    # Returns ((1, 1), 1)
        repeat_like(2, 5)              # Returns 2
    """
    if not isinstance(target, tuple):
        return x
    if not target:
        return ()
    if len(target) == 1:
        return (repeat_like(x, target[0]),)
    return tuple(repeat_like(x, t) for t in target)


def flatten_to_tuple(a: Union[IntTuple, Coord, Shape, Stride]) -> tuple:
    """Flattens a potentially nested tuple structure into a flat tuple.

    This function recursively traverses the input structure and flattens it into
    a single-level tuple, preserving the order of elements.

    :param a: The structure to flatten
    :type a: Union[IntTuple, Coord, Shape, Stride]
    :return: A flattened tuple containing all elements from the input
    :rtype: tuple

    **Examples:**

    .. code-block:: python

        flatten_to_tuple((1, 2, 3))       # Returns (1, 2, 3)
        flatten_to_tuple(((1, 2), 3))     # Returns (1, 2, 3)
        flatten_to_tuple((1, (2, (3,))))  # Returns (1, 2, 3)
    """
    if not isinstance(a, tuple):
        return wrap(a)
    else:
        return tuple(chain.from_iterable(tuple(flatten_to_tuple(x) for x in a)))


@overload
def flatten(a: Union[IntTuple, Coord, Shape, Stride]) -> IntTuple: ...
@overload
def flatten(a: Tensor) -> Tensor: ...
@overload
def flatten(a: Layout) -> Layout: ...


def flatten(a):
    """Flattens a CuTe data structure into a simpler form.

    For tuples, this function flattens the structure into a single-level tuple.
    For layouts, it returns a new layout with flattened shape and stride.
    For tensors, it returns a new tensor with flattened layout.
    For other types, it returns the input unchanged.

    :param a: The structure to flatten
    :type a: Union[IntTuple, Coord, Shape, Stride, Layout, Tensor]
    :return: The flattened structure
    :rtype: Union[tuple, Any]

    **Examples:**

    .. code-block:: python

        flatten((1, 2, 3))                      # Returns (1, 2, 3)
        flatten(((1, 2), (3, 4)))               # Returns (1, 2, 3, 4)
        flatten(5)                              # Returns 5
        flatten(Layout(shape, stride))          # Returns Layout(flatten(shape), flatten(stride))
        flatten(Tensor(layout))                 # Returns Tensor(flatten(layout))

    """
    if isinstance(a, Tensor):
        return make_tensor(a.iterator, flatten(a.layout))
    elif isinstance(a, Layout):
        return make_layout(flatten(a.shape), stride=flatten(a.stride))
    elif isinstance(a, tuple):
        return flatten_to_tuple(a)
    else:
        return a


def unflatten(
    sequence: Union[Tuple[Any, ...], List[Any], Iterable[Any]], profile: XTuple
) -> XTuple:
    """Unflatten a flat tuple into a nested tuple structure according to a profile.

    This function transforms a flat sequence of elements into a nested tuple structure
    that matches the structure defined by the profile parameter. It traverses the profile
    structure and populates it with elements from the sequence.

    sequence must be long enough to fill the profile. Raises RuntimeError if it is not.

    :param sequence: A flat sequence of elements to be restructured
    :type sequence: Union[Tuple[Any, ...], List[Any], Iterable[Any]]
    :param profile: A nested tuple structure that defines the shape of the output
    :type profile: XTuple
    :return: A nested tuple with the same structure as profile but containing elements from sequence
    :rtype: XTuple

    Example:
        >>> unflatten([1, 2, 3, 4], ((0, 0), (0, 0)))
        ((1, 2), (3, 4))
    """

    def _make_generator():
        for element in sequence:
            yield element

    xs = _make_generator()
    return transform_leaf(lambda _: next(xs), profile)


@dsl_user_op
def elem_less(
    lhs: Union[Shape, IntTuple, Coord],
    rhs: Union[Shape, IntTuple, Coord],
    *,
    loc=None,
    ip=None,
):
    lhs_val = _pack_coord(lhs, loc=loc, ip=ip)
    rhs_val = _pack_coord(rhs, loc=loc, ip=ip)
    return Boolean(_cute_ir.elem_less(lhs_val, rhs_val, loc=loc, ip=ip))


@overload
def filter_zeros(
    input: Layout, *, target_profile=None, loc=None, ip=None
) -> Layout: ...
@overload
def filter_zeros(
    input: Tensor, *, target_profile=None, loc=None, ip=None
) -> Tensor: ...


@dsl_user_op
def filter_zeros(input, *, target_profile=None, loc=None, ip=None):
    """Filter out zeros from a layout or tensor.

    This function removes zero-stride dimensions from a layout or tensor.
    Refer to https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/02_layout_algebra.md
    for more layout algebra operations.

    :param input: The input layout or tensor to filter
    :type input: Layout or Tensor
    :param target_profile: Target profile for the filtered result, defaults to None
    :type target_profile: optional
    :param loc: Source location for MLIR, defaults to None
    :type loc: optional
    :param ip: Insertion point, defaults to None
    :type ip: optional
    :return: The filtered layout or tensor with zeros removed
    :rtype: Layout or Tensor
    :raises TypeError: If input is not a Layout or Tensor
    """
    if not isinstance(input, (Layout, Tensor)):
        raise TypeError(f"Expect layout or tensor as input but got {type(input)=}")
    if isinstance(input, Tensor):
        input = input.value
    return _cute_ir.filter_zeros(input, target_profile=target_profile, loc=loc, ip=ip)


@dsl_user_op
def filter(input: Union[Layout, Tensor], *, loc=None, ip=None):
    """Filter a layout or tensor.

    This function filters a layout or tensor according to CuTe's filtering rules.

    :param input: The input layout or tensor to filter
    :type input: Layout or Tensor
    :param loc: Source location for MLIR, defaults to None
    :type loc: optional
    :param ip: Insertion point, defaults to None
    :type ip: optional
    :return: The filtered layout or tensor
    :rtype: Layout or Tensor
    :raises TypeError: If input is not a Layout or Tensor
    """
    if not isinstance(input, (Layout, Tensor)):
        raise TypeError(f"Expect layout or tensor as input but got {type(input)=}")
    if isinstance(input, _Tensor):
        input = input.value
    return _cute_ir.filter(input, loc=loc, ip=ip)


@dsl_user_op
def product(a: Union[IntTuple, Shape], *, loc=None, ip=None):
    """Return product of the given IntTuple or Shape.

    Computes the product of all elements in the input tuple or shape.
    Returns static value if type is static.

    :param a: The input tuple or shape
    :type a: IntTuple or Shape
    :param loc: Source location for MLIR, defaults to None
    :type loc: optional
    :param ip: Insertion point, defaults to None
    :type ip: optional
    :return: Static product of IntTuple or Shape if static, otherwise a Value
    :rtype: int or Value
    :raises TypeError: If input is not an IntTuple or Shape
    """
    if is_integer(a):
        return a
    if isinstance(a, tuple):
        a_val = _pack_int_tuple(a, loc=loc, ip=ip)
        res = _cute_ir.tuple_product(a_val, loc=loc, ip=ip)
        return _unpack_x_tuple(res, loc=loc, ip=ip)
    else:
        raise TypeError(f"expects IntTuple or Shape, but got {type(a)}")


@overload
def product_like(
    a: IntTuple, target_profile: XTuple, *, loc=None, ip=None
) -> IntTuple: ...
@overload
def product_like(a: Shape, target_profile: XTuple, *, loc=None, ip=None) -> Shape: ...


@dsl_user_op
def product_like(
    a: Union[IntTuple, Shape], target_profile: XTuple, *, loc=None, ip=None
):
    """Return product of the given IntTuple or Shape at leaves of `target_profile`.

    This function computes products according to the structure defined by target_profile.

    :param a: The input tuple or shape
    :type a: IntTuple or Shape
    :param target_profile: The profile that guides how products are computed
    :type target_profile: XTuple
    :param loc: Source location for MLIR, defaults to None
    :type loc: optional
    :param ip: Insertion point, defaults to None
    :type ip: optional
    :return: The resulting tuple with products computed according to target_profile
    :rtype: IntTuple or Shape
    :raises TypeError: If inputs have incompatible types
    :raises ValueError: If inputs have incompatible shapes
    """
    # Perform product at leaf of `target_profile`
    if not isinstance(target_profile, tuple):
        return product(a, loc=loc, ip=ip)
    else:
        if not isinstance(a, tuple):
            raise TypeError(f"expects `a` tuple but got {a}")

        if len(a) != len(target_profile):
            raise ValueError(f"expects `a` and `guide` have the same rank")

        return tuple(
            product_like(x, g, loc=loc, ip=ip) for x, g in zip(a, target_profile)
        )


@overload
def product_each(a: IntTuple, *, loc=None, ip=None) -> IntTuple: ...
@overload
def product_each(a: Shape, *, loc=None, ip=None) -> Shape: ...


@dsl_user_op
def product_each(a, *, loc=None, ip=None):
    """Compute products for each component of the input.

    Returns a rank(a) tuple `result` such that get(result, mode=[i]) == product(get(a, mode=[i]))

    :param a: The input tuple or shape
    :type a: IntTuple or Shape
    :param loc: Source location for MLIR, defaults to None
    :type loc: optional
    :param ip: Insertion point, defaults to None
    :type ip: optional
    :return: A tuple containing products for each component
    :rtype: tuple
    :raises TypeError: If input is not an IntTuple or Shape
    """
    if is_integer(a):
        return a
    if isinstance(a, tuple):
        if not a:
            return 1
        else:
            a_val = _pack_int_tuple(a, loc=loc, ip=ip)
            res = _cute_ir.tuple_product_each(a_val, loc=loc, ip=ip)
            return _unpack_x_tuple(res, loc=loc, ip=ip)
    else:
        raise TypeError(f"expects IntTuple or Shape, but got {type(a)}")


@dsl_user_op
def size(
    a: Union[IntTuple, Shape, Layout, ComposedLayout, Tensor],
    mode: List[int] = [],
    *,
    loc=None,
    ip=None,
) -> Int:
    """Return size of domain of layout or tensor.

    Computes the size (number of elements) in the domain of a layout or tensor.
    For layouts, this corresponds to the shape of the coordinate space.
    See https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/01_layout.md
    for more details on layout domains.

    :param a: The input object whose size to compute
    :type a: IntTuple, Shape, Layout, ComposedLayout or Tensor
    :param mode: List of mode(s) for size calculation. If empty, computes total size, defaults to []
    :type mode: list of int, optional
    :param loc: Source location for MLIR, defaults to None
    :type loc: optional
    :param ip: Insertion point, defaults to None
    :type ip: optional
    :return: Static size of layout or tensor if static, otherwise a Value
    :rtype: int or Value
    :raises ValueError: If mode contains non-integer elements
    """
    if any(not isinstance(m, int) for m in mode):
        raise ValueError(f"expects integer elements in mode, but got {mode}")

    if isinstance(a, (TiledMma, TiledCopy)):
        return a.size
    a_val = None
    if not isinstance(a, (Layout, ComposedLayout, Tensor)):
        a_val = _pack_int_tuple(a, loc=loc, ip=ip)
    elif isinstance(a, Tensor):
        a_val = a.value
    else:
        a_val = a

    res = _cute_ir.size(a_val, mode=mode, loc=loc, ip=ip)
    return _unpack_x_tuple(res, loc=loc, ip=ip)  # type: ignore


@dsl_user_op
def shape_div(lhs: Shape, rhs: Shape, *, loc=None, ip=None) -> Shape:
    """Perform element-wise division of shapes.

    This function performs element-wise division between two shapes.

    :param lhs: Left-hand side shape
    :type lhs: Shape
    :param rhs: Right-hand side shape
    :type rhs: Shape
    :param loc: Source location for MLIR, defaults to None
    :type loc: optional
    :param ip: Insertion point, defaults to None
    :type ip: optional
    :return: The result of element-wise division
    :rtype: Shape
    """
    lhs = _pack_shape(lhs, loc=loc, ip=ip)
    rhs = _pack_shape(rhs, loc=loc, ip=ip)
    res = _cute_ir.shape_div(lhs, rhs, loc=loc, ip=ip)
    return _unpack_x_tuple(res, loc=loc, ip=ip)


@dsl_user_op
def ceil_div(input: Shape, tiler: Tiler, *, loc=None, ip=None) -> Shape:
    """
    Compute the ceiling division of a target shape by a tiling specification.

    This function computes the number of tiles required to cover the target domain.
    It is equivalent to the second mode of `zipped_divide(input, tiler)`.

    :param input: A tuple of integers representing the dimensions of the target domain.
    :type input: Shape
    :param tiler: The tiling specification.
    :type tiler: Union[Layout, Shape, Tile]
    :param loc: Optional location information for IR diagnostics.
    :type loc: optional
    :param ip: Optional instruction pointer or context for underlying IR functions.
    :type ip: optional
    :return: A tuple of integers representing the number of tiles required along each dimension,
             i.e. the result of the ceiling division of the input dimensions by the tiler dimensions.
    :rtype: Shape

    Example:

    .. code-block:: python

        import cutlass.cute as cute
        @cute.jit
        def foo():
            input = (10, 6)
            tiler = (3, 4)
            result = cute.ceil_div(input, tiler)
            print(result)  # Outputs: (4, 2)
    """
    input_val = _pack_shape(input, loc=loc, ip=ip)
    tiler_val = _pack_tile(tiler, loc=loc, ip=ip)
    res = _cute_ir.ceil_div(input=input_val, tiler=tiler_val, loc=loc, ip=ip)
    return _unpack_x_tuple(res, loc=loc, ip=ip)


def round_up(a: IntTuple, b: IntTuple) -> IntTuple:
    """
    Rounds up elements of a using elements of b.
    """
    if isinstance(a, tuple):
        if not a:
            raise ValueError(f"inputs cannot be empty")
        if not isinstance(b, tuple):
            raise TypeError(
                f"expects both inputs to be tuple, but got {type(a)} and {type(b)}"
            )
        if rank(a) < rank(b):
            raise ValueError(
                f"expects rank(a) to be greater or equal than rank(b), but got {a}, {b}"
            )
        b = append(b, 1, rank(a))
        return tuple(round_up(x, y) for x, y in zip(a, b))
    return ((a + b - 1) // b) * b


#
# Layout API (also used by tensors)
#


@dsl_user_op
def make_layout(
    shape: Shape, *, stride: Union[Stride, None] = None, loc=None, ip=None
) -> Layout:
    """Create a CuTe Layout object from shape and optional stride information.

    A Layout in CuTe represents the mapping between logical and physical coordinates of a tensor.
    This function creates a Layout object that defines how tensor elements are arranged in memory.

    :param shape: Shape of the layout defining the size of each mode
    :type shape: Shape
    :param stride: Optional stride values for each mode, defaults to None
    :type stride: Union[Stride, None]
    :param loc: Source location information, defaults to None
    :type loc: Optional[Location]
    :param ip: Insertion point for IR generation, defaults to None
    :type ip: Optional[InsertionPoint]
    :return: A new Layout object with the specified shape and stride
    :rtype: Layout

    **Examples:**

    .. code-block:: python

        # Create a 2D compact left-most layout with shape (4,4)
        layout = make_layout((4,4))                     # compact left-most layout

        # Create a left-most layout with custom strides
        layout = make_layout((4,4), stride=(1,4))       # left-most layout with strides (1,4)

        # Create a layout for a 3D tensor
        layout = make_layout((32,16,8))                 # left-most layout

        # Create a layout with custom strides
        layout = make_layout((2,2,2), stride=(4,1,2))   # layout with strides (4,1,2)

    Note:
        - If stride is not provided, a default compact left-most stride is computed based on the shape
        - The resulting layout maps logical coordinates to physical memory locations
        - The layout object can be used for tensor creation and memory access patterns
        - Strides can be used to implement:
          * Row-major vs column-major layouts
          * Padding and alignment
          * Blocked/tiled memory arrangements
          * Interleaved data formats
        - Stride is keyword only argument to improve readability, e.g.
          * make_layout((3,4), (1,4)) can be confusing with make_layout(((3,4), (1,4)))
          * make_layout((3,4), stride=(1,4)) is more readable
    """
    if stride is not None and not is_congruent(shape, stride):
        raise ValueError(f"shape and stride must be congruent")

    shape_val = _pack_shape(shape, loc=loc, ip=ip)
    if stride is not None:
        stride_val = _pack_stride(stride, loc=loc, ip=ip)
        layout_ty = _cute_ir.LayoutType.get(shape_val, stride_val)
    else:
        stride_val = None
        layout_ty = _cute_ir.LayoutType.get(shape_val)

    return _cute_ir.make_layout(
        layout_ty, shape=shape_val, stride=stride_val, loc=loc, ip=ip
    )


@dsl_user_op
def make_identity_layout(shape: Shape, *, loc=None, ip=None) -> Layout:
    """Create an identity layout with the given shape.

    An identity layout maps logical coordinates directly to themselves without any transformation.
    This is equivalent to a layout with stride (1@0,1@1,...,1@(N-1)).

    :param shape: The shape of the layout
    :type shape: Shape
    :param loc: Source location information, defaults to None
    :type loc: Optional[Location]
    :param ip: Insertion point for IR generation, defaults to None
    :type ip: Optional[InsertionPoint]
    :return: A new identity Layout object with the specified shape
    :rtype: Layout

    **Examples:**

    .. code-block:: python

        # Create a 2D identity layout with shape (4,4)
        layout = make_identity_layout((4,4))     # stride=(1@0,1@1)

        # Create a 3D identity layout
        layout = make_identity_layout((32,16,8)) # stride=(1@0,1@1,1@2)

    Note:
        - An identity layout is a special case where each coordinate maps to itself
        - Useful for direct coordinate mapping without any transformation
    """
    if not is_int_tuple(shape):
        raise TypeError(f"expects a shape input, got {type(shape)}")
    shape_val = _pack_shape(shape, loc=loc, ip=ip)
    return _cute_ir.make_identity_layout(shape_val, loc=loc, ip=ip)


@dsl_user_op
def make_ordered_layout(shape: Shape, order: Shape, *, loc=None, ip=None) -> Layout:
    """Create a layout with a specific ordering of dimensions.

    This function creates a layout where the dimensions are ordered according to the
    specified order parameter, allowing for custom dimension ordering in the layout.

    :param shape: The shape of the layout
    :type shape: Shape
    :param order: The ordering of dimensions
    :type order: Shape
    :param loc: Source location information, defaults to None
    :type loc: Optional[Location]
    :param ip: Insertion point for IR generation, defaults to None
    :type ip: Optional[InsertionPoint]
    :return: A new Layout object with the specified shape and dimension ordering
    :rtype: Layout

    **Examples:**

    .. code-block:: python

        # Create a row-major layout
        layout = make_ordered_layout((4,4), order=(1,0))

        # Create a column-major layout
        layout = make_ordered_layout((4,4), order=(0,1))         # stride=(1,4)

        # Create a layout with custom dimension ordering for a 3D tensor
        layout = make_ordered_layout((32,16,8), order=(2,0,1))   # stride=(128,1,16)

    Note:
        - The order parameter specifies the ordering of dimensions from fastest-varying to slowest-varying
        - For a 2D tensor, (0,1) creates a column-major layout, while (1,0) creates a row-major layout
        - The length of order must match the rank of the shape
    """
    shape_val = _pack_shape(shape, loc=loc, ip=ip)
    order_val = _pack_int_tuple(order, loc=loc, ip=ip)
    return _cute_ir.make_ordered_layout(
        shape=shape_val, order=order_val, loc=loc, ip=ip
    )


@dsl_user_op
def make_composed_layout(
    inner, offset: IntTuple, outer: Layout, *, loc=None, ip=None
) -> ComposedLayout:
    """Create a composed layout by composing an inner transformation with an outer layout.

    A composed layout applies a sequence of transformations
    to coordinates. The composition is defined as (inner ∘ offset ∘ outer), where the operations
    are applied from right to left.

    :param inner: The inner transformation (can be a Layout or Swizzle)
    :type inner: Union[Layout, Swizzle]
    :param offset: An integral offset applied between transformations
    :type offset: IntTuple
    :param outer: The outer (right-most) layout that is applied first
    :type outer: Layout
    :param loc: Source location information, defaults to None
    :type loc: Optional[Location]
    :param ip: Insertion point for IR generation, defaults to None
    :type ip: Optional[InsertionPoint]
    :return: A new ComposedLayout representing the composition
    :rtype: ComposedLayout

    **Examples:**

    .. code-block:: python

        # Create a basic layout
        inner = make_layout(...)
        outer = make_layout((4,4), stride=(E(0), E(1)))

        # Create a composed layout with an offset
        composed = make_composed_layout(inner, (2,0), outer)

    Note:
        - The composition applies transformations in the order: outer → offset → inner
        - The stride divisibility condition must be satisfied for valid composition
        - Certain compositions (like Swizzle with scaled basis) are invalid and will raise errors
        - Composed layouts inherit many properties from the outer layout
    """
    if not isinstance(outer, Layout):
        raise TypeError(
            f"expects the outer (or right-most or effectively visible) layout to be an affine layout, but got {outer}"
        )
    if isinstance(inner, Swizzle) and has_scaled_basis(outer.stride):
        raise TypeError(f"invalid composition {inner} o {offset} o {outer}")
    offset_val = _pack_int_tuple(offset, loc=loc, ip=ip)
    return _cute_ir.make_composed_layout(inner, offset_val, outer, loc=loc, ip=ip)


@dsl_user_op
def cosize(
    a: Union[Layout, ComposedLayout, Tensor], mode: List[int] = [], *, loc=None, ip=None
):
    """Return size of codomain of layout or tensor. Return static value if type is static.

    :param a: Layout, ComposedLayout, or Tensor object
    :type a: Union[Layout, ComposedLayout, Tensor]
    :param mode: List of mode(s) for cosize calculation
    :type mode: List[int], optional
    :param loc: Location information for diagnostics, defaults to None
    :type loc: optional
    :param ip: Instruction pointer for diagnostics, defaults to None
    :type ip: optional
    :return: Static size of layout or tensor (fast fold) if static, or a dynamic Value
    :rtype: Union[int, Value]
    """
    if any(not is_static(m) for m in mode):
        raise ValueError(f"expects static mode, but got {mode}")

    if isinstance(a, _Tensor):
        a = a.value
    res = _cute_ir.cosize(a, mode=mode, loc=loc, ip=ip)
    return _unpack_x_tuple(res, loc=loc, ip=ip)


@dsl_user_op
def size_in_bytes(
    dtype: Type[Numeric], layout: Union[Layout, ComposedLayout], *, loc=None, ip=None
):
    """Calculate the size in bytes based on its data type and layout.

    :param dtype: The DSL numeric data type
    :type dtype: Type[Numeric]
    :param layout: The layout of the elements. If None, the function returns 0
    :type layout: Layout, optional
    :param loc: Location information for diagnostics, defaults to None
    :type loc: optional
    :param ip: Instruction pointer for diagnostics, defaults to None
    :type ip: optional
    :return: The total size in bytes. Returns 0 if the layout is None
    :rtype: int
    """
    if not isinstance(dtype, NumericMeta):
        raise TypeError(f"dtype must be a Numeric, but got {dtype}")

    if layout is None:
        return 0
    elif isinstance(layout, ComposedLayout):
        if not isinstance(layout.inner, Swizzle):
            raise TypeError(
                f"invalid composed layout {layout}, inner must be a Swizzle"
            )
        else:
            return cosize(layout.outer, loc=loc, ip=ip) * dtype.width // 8
    else:
        return cosize(layout, loc=loc, ip=ip) * dtype.width // 8


@dsl_user_op
def coalesce(input, *, target_profile: Coord = None, loc=None, ip=None):
    if target_profile:
        profile_val = _pack_coord(target_profile, loc=loc, ip=ip)
        return _cute_ir.coalesce(input, target_profile=profile_val, loc=loc, ip=ip)
    else:
        return _cute_ir.coalesce(input, loc=loc, ip=ip)


@dsl_user_op
def crd2idx(coord: Coord, layout, *, loc=None, ip=None):
    """
    Convert a multi-dimensional coordinate into a value using the specified layout.

    This function computes the inner product of the flattened coordinate and stride:

        index = sum(flatten(coord)[i] * flatten(stride)[i] for i in range(len(coord)))

    :param coord: A tuple or list representing the multi-dimensional coordinate
                  (e.g., (i, j) for a 2D layout).
    :type coord: Coord
    :param layout: A layout object that defines the memory storage layout, including shape and stride,
                   used to compute the inner product.
    :type layout: Layout or ComposedLayout
    :param loc: Optional location information for IR diagnostics.
    :type loc: optional
    :param ip: Optional instruction pointer or context for underlying IR functions.
    :type ip: optional
    :returns: The result of applying the layout transformation to the provided coordinate.
    :rtype: Any type that the layout maps to

    Example:

    .. code-block:: python

        import cutlass.cute as cute
        @cute.jit
        def foo():
            L = cute.make_layout((5, 4), stride=(4, 1))
            idx = cute.crd2idx((2, 3), L)
            # Computed as: 2 * 4 + 3 = 11
            print(idx)
        foo()  # Expected output: 11
    """
    coord_val = _pack_coord(coord, loc=loc, ip=ip)
    if isinstance(layout, (tuple, int)):
        layout = make_layout(layout, loc=loc, ip=ip)

    res = _cute_ir.crd2idx(coord_val, layout, loc=loc, ip=ip)
    return _unpack_x_tuple(res, loc=loc, ip=ip)


@dsl_user_op
def recast_layout(new_type_bits, old_type_bits, src_layout, *, loc=None, ip=None):
    return _cute_ir.recast_layout(
        new_type_bits, old_type_bits, src_layout, loc=loc, ip=ip
    )


@dsl_user_op
def slice_and_offset(coord, src, *, loc=None, ip=None):
    layout = slice_(src, coord, loc=loc, ip=ip)
    offset = crd2idx(coord, src, loc=loc, ip=ip)
    return layout, offset


@dsl_user_op
@lru_cache_ir()
def shape(
    input: Union[Shape, Tensor, Layout, Tile], *, mode=None, loc=None, ip=None
) -> Shape:
    """Returns the shape of a tensor, layout or tiler.

    For shapes, this function is identical to get.

    This function extracts the shape information from the input object. For tensors and layouts,
    it returns their internal shape property. For tilers, it unpacks the shape from the tile
    representation.

    :param input: The object to extract shape from
    :type input: Union[Tensor, Layout, Tile]
    :param mode: Optional mode selector to extract specific dimensions from the shape
    :type mode: Optional[int]
    :param loc: Source location for MLIR operation tracking
    :type loc: Optional[Location]
    :param ip: Insertion point for MLIR operation
    :type ip: Optional[InsertionPoint]
    :return: The shape of the input object, optionally filtered by mode
    :rtype: Shape

    Example:

    .. code-block:: python

        # Get shape of a layout
        l0 = cute.make_layout((2, 3, 4))
        s0 = cute.shape(l0)  # => (2, 3, 4)

        # Get shape of a hierarchical tiler
        l1 = cute.make_layout(1)
        s1 = cute.shape((l0, l1))  # => ((2, 3, 4), 1)

        # Get specific mode from a shape
        s2 = cute.shape(l0, mode=0)  # => 2
    """
    if is_int_tuple(input):
        return get(input, mode=mode)

    if isinstance(input, (Tensor, Layout)):
        shp = input.shape
    else:
        val = _cute_ir.get_shape(_pack_tile(input, loc=loc, ip=ip))
        shp = _unpack_x_tuple(val, loc=loc, ip=ip)
    return get(shp, mode=mode)


#
# Pointer API
#


@dsl_user_op
def recast_ptr(
    ptr: Pointer,
    swizzle_=None,
    dtype: Optional[Type[Numeric]] = None,
    loc=None,
    ip=None,
) -> Pointer:
    if dtype is not None:
        if not isclass(dtype) or not issubclass(dtype, Numeric):
            raise TypeError(f"dtype must be a type of Numeric, but got {dtype}")
        dtype = dtype.mlir_type

    value_type = ptr.type.value_type if dtype is None else dtype
    swizzle = swizzle_.type.attribute if swizzle_ is not None else None
    res_ty = _cute_ir.PtrType.get(value_type, ptr.memspace, ptr.alignment, swizzle)
    return _cute_ir.recast_iter(res_ty, ptr.value, loc=loc, ip=ip)


@dsl_user_op
def make_ptr(
    dtype: Union[Type[Numeric], None],
    value,
    mem_space: AddressSpace = AddressSpace.generic,
    *,
    assumed_align=None,
    loc=None,
    ip=None,
) -> Pointer:
    if dtype is None or not isinstance(dtype, NumericMeta):
        raise TypeError(f"expects dtype to be a type of Numeric, but got {dtype}")

    if not isinstance(mem_space, AddressSpace):
        raise TypeError(f"expects mem_space to be an AddressSpace, but got {mem_space}")

    if isinstance(value, ir.Value) and llvm.PointerType.isinstance(value.type):
        value = llvm.ptrtoint(T.i64(), value)

    if not is_integer(value):
        raise TypeError(f"expects integer value, but got {type(value)}")
    value = Int32(value) if mem_space == AddressSpace.tmem else Int64(value)

    bytes_per_elt = max(1, dtype.width // 8)
    if assumed_align is None:
        assumed_align = bytes_per_elt

    if bytes_per_elt % assumed_align != 0 and assumed_align % bytes_per_elt != 0:
        raise ValueError(
            f"{bytes_per_elt=} is not a multiple of {assumed_align=} and vice versa."
        )

    aligned_ty = _cute_ir.ConstrainedIntType.get(assumed_align, type(value).width)
    aligned_intptr = _cute_ir.assume(aligned_ty, value.ir_value(), loc=loc, ip=ip)

    data_ty = T.i8() if dtype is None else dtype.mlir_type
    ptr_ty = _cute_ir.PtrType.get(data_ty, mem_space, assumed_align)
    return _cute_ir.inttoptr(ptr_ty, aligned_intptr, loc=loc, ip=ip)


#
# Tensor API
#


@dsl_user_op
def make_tensor(
    iterator, layout: Union[Shape, Layout, ComposedLayout], *, loc=None, ip=None
) -> Tensor:
    """Creates a tensor by composing an engine (iterator/pointer) with a layout.

    A tensor is defined as T = E ∘ L, where E is an engine (array, pointer, or counting iterator)
    and L is a layout that maps logical coordinates to physical offsets. The tensor
    evaluates coordinates by applying the layout mapping and dereferencing the engine
    at the resulting offset.

    :param iterator: Engine component (pointer, iterator, or counting iterator) that provides
                    data access capabilities
    :type iterator: Union[Pointer, IntTuple]
    :param layout: Layout component that defines the mapping from logical coordinates to
                  physical offsets
    :type layout: Union[Shape, Layout, ComposedLayout]
    :param loc: Source location for MLIR operation tracking, defaults to None
    :type loc: Optional[Location]
    :param ip: Insertion point for MLIR operation, defaults to None
    :type ip: Optional[InsertionPoint]
    :return: A tensor object representing the composition E ∘ L
    :rtype: Tensor

    :raises ValueError: If iterator type is not supported

    **Examples:**

    .. code-block:: python

        # Create a tensor with row-major layout
        layout = make_layout((64, 128), stride=(128, 1))
        tensor = make_tensor(ptr, layout)

        # Create a tensor with hierarchical layout
        layout = make_layout(((128, 8), (1, 4, 1)), stride=((32, 1), (0, 8, 4096)))
        tensor = make_tensor(smem_ptr, layout)

        # Create a coord tensor
        layout = make_layout(2, stride=16 * E(0))
        tensor = make_tensor(5, layout)

    Notes:
        - The engine (iterator) must support random access operations
        - Common engine types include raw pointers, arrays, and random-access iterators
        - The layout defines both the shape (logical dimensions) and stride (physical mapping)
        - Supports both direct coordinate evaluation T(c) and partial evaluation (slicing)
    """
    if not isinstance(layout, (Layout, ComposedLayout)):
        layout = make_layout(layout, loc=loc, ip=ip)
    elif isinstance(layout, ComposedLayout) and layout.type.is_normal_layout:
        layout = layout.outer

    ty = None
    if is_integer(iterator) or isinstance(iterator, tuple):
        iterator = _pack_int_tuple(iterator, loc=loc, ip=ip)
        ty = _cute_ir.CoordTensorType.get(iterator.type, layout.type)
    elif isinstance(iterator, Pointer):
        iterator = iterator.value
        ty = _cute_ir.MemRefType.get(iterator.type, layout.type)
    else:
        raise TypeError(f"unsupported iterator type, got {type(iterator)}")

    return _cute_ir.make_view(result=ty, iter=iterator, layout=layout, loc=loc, ip=ip)


@dsl_user_op
def make_identity_tensor(shape: Shape, *, loc=None, ip=None) -> Tensor:
    """Creates an identity tensor with the given shape.

    An identity tensor maps each coordinate to itself, effectively creating a counting
    sequence within the shape's bounds. This is useful for generating coordinate indices
    or creating reference tensors for layout transformations.

    :param shape: The shape defining the tensor's dimensions. Can be a simple integer
                 sequence or a hierarchical structure ((m,n),(p,q))
    :type shape: Shape
    :param loc: Source location for MLIR operation tracking, defaults to None
    :type loc: Optional[Location]
    :param ip: Insertion point for MLIR operation, defaults to None
    :type ip: Optional[InsertionPoint]
    :return: A tensor that maps each coordinate to itself
    :rtype: Tensor

    **Examples:**

    .. code-block:: python

        # Create a simple 1D coord tensor
        tensor = make_identity_tensor(6)  # [0,1,2,3,4,5]

        # Create a 2D coord tensor
        tensor = make_identity_tensor((3,2))  # [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1)]

        # Create hierarchical coord tensor
        tensor = make_identity_tensor(((2,1),3))
        # [((0,0),0),((1,0),0),((0,0),1),((1,0),1),((0,0),2),((1,0),2)]

    Notes:
        - The shape parameter follows CuTe's IntTuple concept
        - Coordinates are ordered colexicographically
        - Useful for generating reference coordinates in layout transformations
    """
    shape_val = _pack_shape(shape, loc=loc, ip=ip)
    return _cute_ir.make_identity_tensor(shape_val, loc=loc, ip=ip)


@dsl_user_op
def make_fragment(
    layout_or_shape: Union[Layout, Shape],
    dtype: Type[Numeric],
    *,
    loc=None,
    ip=None,
) -> Tensor:
    if not issubclass(dtype, Numeric):
        raise TypeError(f"value_type must be a type of Numeric, but got {type(dtype)}")
    elem_ty = dtype.mlir_type if dtype is not Boolean else T.i8()

    # Alignment for register memory is useless(?), pick-up large enough number
    # to allow .128 (> 16B) load store
    alignment = 32
    layout = None
    if not isinstance(layout_or_shape, Layout):
        layout = make_layout(layout_or_shape, loc=loc, ip=ip)
    else:
        layout = layout_or_shape

    ptr_ty = _cute_ir.PtrType.get(elem_ty, AddressSpace.rmem, alignment)
    res_ty = _cute_ir.MemRefType.get(ptr_ty, layout.type)
    tensor = _cute_ir.memref_alloca(res_ty, layout=layout, loc=loc, ip=ip)
    return _Tensor(tensor.value, dtype)


@overload
def make_fragment_like(
    src: Tensor, dtype: Optional[Type[Numeric]], *, loc=None, ip=None
) -> Tensor: ...


@overload
def make_fragment_like(src: Layout, *, loc=None, ip=None) -> Layout: ...


@overload
def make_fragment_like(src: ComposedLayout, *, loc=None, ip=None) -> ComposedLayout: ...


@dsl_user_op
def make_fragment_like(src, dtype=None, *, loc=None, ip=None):
    """Create tensor with a compact layout in the same shape as the source on stack.

    This function either creates a fragment tensor with compact layout in
    same shape as the source layout or a new layout with the same shape as the source.
    The strides of the new layout follow the order induced by the source's strides, with a
    special handling of the 0th mode: it is always stride-1 and generated in column-major order
    (LayoutLeft).

    :param src: The source layout or tensor whose shape will be matched
    :type src: Union[Layout, ComposedLayout, Tensor]
    :param dtype: The element type for the fragment tensor, defaults to None
    :type dtype: Type[Numeric], optional
    :param loc: Source location for MLIR operations, defaults to None
    :type loc: Location, optional
    :param ip: Insertion point for MLIR operations, defaults to None
    :type ip: InsertionPoint, optional

    :return: A new layout or fragment tensor with matching shape
    :rtype: Union[Layout, Tensor]

    **Examples:**

    Creating a rmem tensor from a tensor:

    .. code-block:: python

        smem_tensor = cute.make_tensor(smem_ptr, layout)
        frag_tensor = cute.make_fragment_like(smem_tensor, cutlass.Float32)
        # frag_tensor will be a register-backed tensor with the same shape

    Creating a fragment with a different element type:

    .. code-block:: python

        tensor = cute.make_tensor(gmem_ptr, layout)
        bool_frag = cute.make_fragment_like(tensor, cutlass.Boolean)
        # bool_frag will be a register-backed tensor with Boolean elements

    **Notes**

    - When used with a Tensor, if a type is provided, it will create a new
      fragment tensor with that element type.
    - For layouts with ScaledBasis strides, the function creates a fragment
      from the shape only.
    - This function is commonly used in GEMM and other tensor operations to
      create register storage for intermediate results.

    """
    if isinstance(src, (Layout, ComposedLayout)):
        new_layout = None
        # Create base fragment layout
        if isinstance(src, Layout) and has_scaled_basis(src.stride):
            # For scaled basis strides, create fragment from shape only
            new_layout = _cute_ir.make_fragment_like(
                make_layout(src.shape), loc=loc, ip=ip
            )
        else:
            # Otherwise use full source layout
            new_layout = _cute_ir.make_fragment_like(src, loc=loc, ip=ip)
        if dtype is not None:
            # call make_fragment to convert layout to tensor
            return make_fragment(new_layout, dtype, loc=loc, ip=ip)
        else:
            return new_layout
    elif isinstance(src, Tensor):
        if isinstance(src.type, _cute_ir.CoordTensorType):
            if dtype is None:
                raise ValueError(
                    "dtype must be provided when src is a coordinate tensor"
                )

            new_layout = _cute_ir.make_fragment_like(
                make_layout(src.shape), loc=loc, ip=ip
            )
            return make_fragment(new_layout, dtype, loc=loc, ip=ip)
        else:
            dtype = src.element_type if dtype is None else dtype
            ty = dtype.mlir_type if dtype is not Boolean else T.i8()
            new_tensor = _cute_ir.make_fragment_like(
                src.value, elem_type=ty, loc=loc, ip=ip
            )
            return _Tensor(new_tensor.value, dtype)
    else:
        raise TypeError(
            f"src must be a Layout or ComposedLayout or tensor, got {type(src)}"
        )


@dsl_user_op
def recast_tensor(
    src: Tensor, dtype: Type[Numeric], swizzle_=None, *, loc=None, ip=None
):
    if not isclass(dtype) or not issubclass(dtype, Numeric):
        raise TypeError(f"dtype must be a type of Numeric, but got {dtype}")

    if dtype is Boolean:
        dst_width = 8
    else:
        dst_width = dtype.width

    if src.element_type is Boolean:
        src_width = 8
    else:
        src_width = src.element_type.width

    src_iter = recast_ptr(src.iterator, dtype=dtype, loc=loc, ip=ip)
    src_layout = recast_layout(dst_width, src_width, src.layout, loc=loc, ip=ip)
    return make_tensor(src_iter, src_layout, loc=loc, ip=ip)


@dsl_user_op
def domain_offset(coord: Coord, tensor: Tensor, *, loc=None, ip=None) -> Tensor:
    offset = crd2idx(coord, tensor.layout, loc=loc, ip=ip)
    if isinstance(tensor.iterator, Pointer):
        return make_tensor(tensor.iterator + offset, tensor.layout)
    elif is_integer(tensor.iterator) or isinstance(tensor.iterator, tuple):
        new_iter = _cute_ir.add_offset(
            _pack_int_tuple(tensor.iterator), _pack_int_tuple(offset)
        )
        return make_tensor(_unpack_x_tuple(new_iter), tensor.layout)
    else:
        raise ValueError(f"unsupported tensor for domain_offset, got {tensor}")


#
# Layout algebra
#


@overload
def composition(
    lhs: Layout, rhs: Union[Layout, Shape, Tile], *, loc=None, ip=None
) -> Layout: ...


@overload
def composition(
    lhs: Tensor, rhs: Union[Layout, Shape, Tile], *, loc=None, ip=None
) -> Tensor: ...


@dsl_user_op
def composition(lhs, rhs: Union[Layout, Shape, Tile], *, loc=None, ip=None):
    """
    Compose two layout representations using the CuTe layout algebra.

    Compose a left-hand layout (or tensor) with a right-hand operand into a new layout R, such that
    for every coordinate c in the domain of the right-hand operand, the composed layout satisfies:

        R(c) = A(B(c))

    where A is the left-hand operand provided as ``lhs`` and B is the right-hand operand provided as
    ``rhs``. In this formulation, B defines the coordinate domain while A applies its transformation to
    B's output, and the resulting layout R inherits the stride and shape adjustments from A.

    Satisfies:
        cute.shape(cute.composition(lhs, rhs)) is compatible with cute.shape(rhs)

    :param lhs: The left-hand operand representing the transformation to be applied.
    :type lhs: Layout or Tensor
    :param rhs: The right-hand operand defining the coordinate domain. If provided as an int or tuple,
                it will be converted to a tile layout.
    :type rhs: Layout, Shape, or Tile, or int or tuple
    :param loc: Optional location information for IR diagnostics.
    :type loc: optional
    :param ip: Optional instruction pointer or context for underlying IR functions.
    :type ip: optional
    :returns: A new composed layout R, such that for all coordinates c in the domain of ``rhs``,
              R(c) = lhs(rhs(c)).
    :rtype: Layout or Tensor

    Example:

    .. code-block:: python

        import cutlass.cute as cute
        @cute.jit
        def foo():
            # Create a layout that maps (i,j) to i*4 + j
            L1 = cute.make_layout((2, 3), stride=(4, 1))
            # Create a layout that maps (i,j) to i*3 + j
            L2 = cute.make_layout((3, 4), stride=(3, 1))
            # Compose L1 and L2
            L3 = cute.composition(L1, L2)
            # L3 now maps coordinates through L2 then L1
    """
    rhs_val = rhs
    if not isinstance(rhs, Layout) and isinstance(rhs, (int, tuple)):
        rhs_val = _pack_tile(rhs, loc=loc, ip=ip)
    if isinstance(lhs, _Tensor):
        lhs = lhs.value
    return _cute_ir.composition(lhs, rhs_val, loc=loc, ip=ip)


@dsl_user_op
def complement(
    input: Layout, cotarget: Union[Layout, Shape], *, loc=None, ip=None
) -> Layout:
    """
    Compute the complement layout of the input layout with respect to the cotarget.

    The complement of a layout A with respect to cotarget n is a layout A* such that
    for every k in Z_n and c in the domain of A, there exists a unique c* in the domain
    of A* where k = A(c) + A*(c*).

    This operation is useful for creating layouts that partition a space in complementary ways,
    such as row and column layouts that together cover a matrix.

    :param input: The layout to compute the complement of
    :type input: Layout
    :param cotarget: The target layout or shape that defines the codomain
    :type cotarget: Union[Layout, Shape]
    :param loc: Optional location information for IR diagnostics
    :type loc: optional
    :param ip: Optional instruction pointer or context for underlying IR functions
    :type ip: optional
    :returns: The complement layout
    :rtype: Layout

    Example:

    .. code-block:: python

        import cutlass.cute as cute
        @cute.jit
        def foo():
            # Create a right-major layout for a 4x4 matrix
            row_layout = cute.make_layout((4, 4), stride=(4, 1))
            # Create a left-major layout that complements the row layout
            col_layout = cute.complement(row_layout, 16)
            # The two layouts are complementary under 16
    """
    if isinstance(cotarget, Layout):
        return _cute_ir.complement(input, cotarget=cotarget, loc=loc, ip=ip)
    else:
        cotarget_val = _pack_shape(cotarget, loc=loc, ip=ip)
        return _cute_ir.complement(input, cotarget=cotarget_val, loc=loc, ip=ip)


@dsl_user_op
def right_inverse(input: Layout, *, loc=None, ip=None) -> Layout:
    if not isinstance(input, Layout):
        raise TypeError(f"expects input of type Layout, but got {type(input)}")
    return _cute_ir.right_inverse(input=input, loc=loc, ip=ip)


@dsl_user_op
def left_inverse(input: Layout, *, loc=None, ip=None) -> Layout:
    if not isinstance(input, Layout):
        raise TypeError(f"expects input of type Layout, but got {type(input)}")
    return _cute_ir.left_inverse(input=input, loc=loc, ip=ip)


@overload
def logical_product(block: Layout, tiler: Layout, *, loc=None, ip=None) -> Layout: ...
@overload
def logical_product(
    block: ComposedLayout, tiler: Layout, *, loc=None, ip=None
) -> ComposedLayout: ...


@dsl_user_op
def logical_product(block, tiler: Layout, *, loc=None, ip=None):
    return _cute_ir.logical_product(input=block, tiler=tiler, loc=loc, ip=ip)


@overload
def zipped_product(block: Layout, tiler: Layout, *, loc=None, ip=None) -> Layout: ...
@overload
def zipped_product(
    block: ComposedLayout, tiler: Layout, *, loc=None, ip=None
) -> ComposedLayout: ...


@dsl_user_op
def zipped_product(block, tiler: Layout, *, loc=None, ip=None):
    return _cute_ir.zipped_product(input=block, tiler=tiler, loc=loc, ip=ip)


@overload
def tiled_product(block: Layout, tiler: Layout, *, loc=None, ip=None) -> Layout: ...
@overload
def tiled_product(
    block: ComposedLayout, tiler: Layout, *, loc=None, ip=None
) -> ComposedLayout: ...


@dsl_user_op
def tiled_product(block, tiler: Layout, *, loc=None, ip=None):
    return _cute_ir.tiled_product(input=block, tiler=tiler, loc=loc, ip=ip)


@overload
def flat_product(block: Layout, tiler: Layout, *, loc=None, ip=None) -> Layout: ...
@overload
def flat_product(
    block: ComposedLayout, tiler: Layout, *, loc=None, ip=None
) -> ComposedLayout: ...


@dsl_user_op
def flat_product(block, tiler: Layout, *, loc=None, ip=None):
    return _cute_ir.flat_product(input=block, tiler=tiler, loc=loc, ip=ip)


@overload
def raked_product(block: Layout, tiler: Layout, *, loc=None, ip=None) -> Layout: ...
@overload
def raked_product(
    block: ComposedLayout, tiler: Layout, *, loc=None, ip=None
) -> ComposedLayout: ...


@dsl_user_op
def raked_product(block, tiler: Layout, *, loc=None, ip=None):
    return _cute_ir.raked_product(input=block, tiler=tiler, loc=loc, ip=ip)


@overload
def blocked_product(block: Layout, tiler: Layout, *, loc=None, ip=None) -> Layout: ...
@overload
def blocked_product(
    block: ComposedLayout, tiler: Layout, *, loc=None, ip=None
) -> ComposedLayout: ...


@dsl_user_op
def blocked_product(block, tiler: Layout, *, loc=None, ip=None):
    return _cute_ir.blocked_product(input=block, tiler=tiler, loc=loc, ip=ip)


@overload
def logical_divide(target: Layout, tiler: Tiler, *, loc=None, ip=None) -> Layout: ...
@overload
def logical_divide(target: Tensor, tiler: Tiler, *, loc=None, ip=None) -> Tensor: ...


@dsl_user_op
def logical_divide(target, tiler: Tiler, *, loc=None, ip=None):
    res_type = None
    if isinstance(target, _Tensor):
        res_type = target.element_type
        target = target.value
    if isinstance(tiler, tuple):
        tiler = _pack_tile(tiler, loc=loc, ip=ip)
    res = _cute_ir.logical_divide(input=target, tiler=tiler, loc=loc, ip=ip)
    return _Tensor(res, dtype=res_type) if isinstance(res, _Tensor) else res


@overload
def zipped_divide(target: Layout, tiler: Tiler, *, loc=None, ip=None) -> Layout: ...
@overload
def zipped_divide(target: Tensor, tiler: Tiler, *, loc=None, ip=None) -> Tensor: ...


@dsl_user_op
def zipped_divide(target, tiler: Tiler, *, loc=None, ip=None):
    res_type = None
    if isinstance(target, _Tensor):
        res_type = target.element_type
        target = target.value
    if isinstance(tiler, tuple):
        tiler = _pack_tile(tiler, loc=loc, ip=ip)
    res = _cute_ir.zipped_divide(input=target, tiler=tiler, loc=loc, ip=ip)
    return _Tensor(res, dtype=res_type) if isinstance(res, _Tensor) else res


@overload
def tiled_divide(target: Layout, tiler: Tiler, *, loc=None, ip=None) -> Layout: ...
@overload
def tiled_divide(target: Tensor, tiler: Tiler, *, loc=None, ip=None) -> Tensor: ...


@dsl_user_op
def tiled_divide(target, tiler: Tiler, *, loc=None, ip=None):
    res_type = None
    if isinstance(target, _Tensor):
        res_type = target.element_type
        target = target.value
    if isinstance(tiler, tuple):
        tiler = _pack_tile(tiler, loc=loc, ip=ip)
    res = _cute_ir.tiled_divide(input=target, tiler=tiler, loc=loc, ip=ip)
    return _Tensor(res, dtype=res_type) if isinstance(res, _Tensor) else res


@overload
def flat_divide(target: Layout, tiler: Tiler, *, loc=None, ip=None) -> Layout: ...
@overload
def flat_divide(target: Tensor, tiler: Tiler, *, loc=None, ip=None) -> Tensor: ...


@dsl_user_op
def flat_divide(target, tiler: Tiler, *, loc=None, ip=None):
    res_type = None
    if isinstance(target, _Tensor):
        res_type = target.element_type
        target = target.value
    if isinstance(tiler, tuple):
        tiler = _pack_tile(tiler, loc=loc, ip=ip)
    res = _cute_ir.flat_divide(input=target, tiler=tiler, loc=loc, ip=ip)
    return _Tensor(res, dtype=res_type) if isinstance(res, _Tensor) else res


#
# Higher-level utilties
#


@dsl_user_op
def max_common_layout(
    a: Union[Layout, Tensor], b: Union[Layout, Tensor], *, loc=None, ip=None
) -> Layout:
    a_layout = a.layout if isinstance(a, _Tensor) else a
    b_layout = b.layout if isinstance(b, _Tensor) else b

    inv_b = right_inverse(b_layout, loc=loc, ip=ip)
    common = coalesce(composition(a_layout, inv_b, loc=loc, ip=ip), loc=loc, ip=ip)

    # some_ir_value == 1 generates a new IR Value which evaluates to True!
    s = get(common.shape, mode=[0], loc=loc, ip=ip)
    d = get(common.stride, mode=[0], loc=loc, ip=ip)
    # Keep only the static identity component of the common layout
    if isinstance(s, int) and isinstance(d, int) and d == 1:
        # Truncate to the size of the contiguous vector (static stride-1 mode)
        return composition(inv_b, get(common, mode=[0], loc=loc, ip=ip), loc=loc, ip=ip)
    else:
        return make_layout(1, stride=0, loc=loc, ip=ip)


@dsl_user_op
def max_common_vector(
    a: Union[Layout, Tensor], b: Union[Layout, Tensor], *, loc=None, ip=None
) -> int:
    a_layout = a.layout if isinstance(a, _Tensor) else a
    b_layout = b.layout if isinstance(b, _Tensor) else b

    inv_b = right_inverse(b_layout, loc=loc, ip=ip)
    common = coalesce(composition(a_layout, inv_b, loc=loc, ip=ip), loc=loc, ip=ip)

    # Keep only the static identity component of the common layout
    if (
        is_static(get(common.shape, mode=[0], loc=loc, ip=ip))
        and get(common.stride, mode=[0], loc=loc, ip=ip) == 1
    ):
        # Truncate to the size of the contiguous vector (static stride-1 mode)
        return get(common.shape, mode=[0], loc=loc, ip=ip)
    else:
        return 1


@dsl_user_op
def tile_to_shape(
    atom: Union[Layout, ComposedLayout],
    trg_shape: Shape,
    order: Shape,
    *,
    loc=None,
    ip=None,
) -> Union[Layout, ComposedLayout]:
    trg_shape = _pack_shape(shape(trg_shape), loc=loc, ip=ip)
    order = _pack_int_tuple(order, loc=loc, ip=ip)
    return _cute_ir.tile_to_shape(atom, trg_shape, order, loc=loc, ip=ip)


@dsl_user_op
def local_partition(
    target: Tensor,
    tiler: Union[Layout, Shape],
    index: Union[int, Numeric],
    proj: XTuple = 1,
    *,
    loc=None,
    ip=None,
) -> Tensor:
    if isinstance(index, cutlass_arith.ArithValue):
        index_val = index
    else:
        index_val = index.ir_value()
    if index_val.type.width > 32:
        raise NotImplementedError(
            f"Index value should be 32-bit or smaller integer type, but got {index_val.type}"
        )
    return _cute_ir.local_partition(
        input=target.value, tiler=dice(tiler, proj), index=index_val, loc=loc, ip=ip
    )


@dsl_user_op
def local_tile(
    input: Tensor,
    tiler: Union[Layout, Shape],
    coord: Coord,
    proj: XTuple = None,
    *,
    loc=None,
    ip=None,
) -> Tensor:
    tiler_val = _pack_shape(tiler, loc=loc, ip=ip)
    coord_val = _pack_coord(coord, loc=loc, ip=ip)
    if proj is not None:
        if not isinstance(proj, tuple):
            raise TypeError(f"Expects tuple for proj, but got {type(proj)}")
        proj_val = _pack_coord(proj, loc=loc, ip=ip)
        proj = proj_val.type.attribute

    return _cute_ir.local_tile(
        input=input.value,
        tile=tiler_val,
        static_tile=None,
        coord=coord_val,
        static_coord=None,
        proj=proj,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def make_layout_image_mask(
    lay: Layout, coord: Coord, mode: int, *, loc=None, ip=None
) -> Int16:
    """
    Makes a 16-bit integer mask of the image of a layout sliced at a given mode
    and accounting for the offset given by the input coordinate for the other modes.
    """
    if not is_static(lay):
        raise ValueError(
            f"make_layout_image_mask requires the layout to be static, but got {pretty_str(lay)}"
        )
    r = rank(lay)
    if rank(coord) != r:
        raise ValueError(
            f"the rank of the coordinate must be equal to the one of the layout, but got {pretty_str(coord)}"
        )
    if mode > r or mode < 0:
        raise ValueError(f"expects `mode` to be in [0,rank(lay)), but got {mode}")
    # Given that we require the layout to be static, we can check that the mask fits in 16 bits
    # This might be too conservative but safe
    if cosize(lay) > 16:
        raise ValueError("the mask may not fit into a 16-bit integer")

    # Replace the mode to keep with _ in the coordinate
    slicer = tuple(None if idx == mode else x for idx, x in enumerate(coord))
    # Slice the layout with the slicer above and keep track of the offset
    sliced_lay, offset = slice_and_offset(slicer, lay, loc=loc, ip=ip)
    # Given that we replace only one mode with _, the rank of the slice should be 1
    assert rank(sliced_lay) == 1

    # Create the mask of the image
    mcast_mask = Int16(0)
    for i in range(size(sliced_lay)):
        mcast_mask = mcast_mask | (1 << sliced_lay(i))
    mcast_mask <<= offset
    return Int16(mcast_mask)


####################################################################################################
#
# Atom
#
####################################################################################################


class Op(ABC):
    """
    Operation abstract base class.
    """

    pass


class MmaOp(Op):
    """
    MMA Operation abstract base class.
    """

    @abstractmethod
    def _make_trait(self, *, loc=None, ip=None, **kwargs):
        pass


class CopyOp(Op):
    """
    Copy Operation abstract base class.
    """

    @abstractmethod
    def _make_trait(
        self, copy_internal_type: Type[Numeric], *, loc=None, ip=None, **kwargs
    ):
        pass


class Trait(ABC):
    """
    Trait abstract base class.

    Traits are internal-only classes used by Atoms that wrap the underlying IR Value. The Python
    user should only interact with Ops and Atoms.
    """

    def __init__(self, value: ir.Value) -> None:
        self.value = value

    def __extract_mlir_values__(self):
        return [self.value]

    def __new_from_mlir_values__(self, values):
        return self.__class__(values[0])

    def set(self, field, value, *, loc=None, ip=None) -> None:
        raise NotImplementedError(
            "set not implemented, the requesting Atom has likely no runtime state"
        )

    def unpack(self, *, loc=None, ip=None, **kwargs) -> ir.Value:
        return self.value


class Atom(ABC):
    """
    Atom base class.

    An Atom is the composition of

    - a MMA or Copy Operation;
    - an internal MMA or Copy Trait.

    An Operation is a pure Python class that is used to model a specific MMA or Copy instruction.
    The Trait wraps the underlying IR Value and provides access to the metadata of the instruction
    encoded using CuTe Layouts. When the Trait can be constructed straighforwardly from an
    Operation, the ``make_mma_atom`` or ``make_copy_atom`` API should be used. There are cases where
    constructing the metadata is not trivial and requires more information, for example to determine
    the number of bytes copied per TMA instruction ("the TMA vector length"). In such cases,
    dedicated helper functions are provided with an appropriate API such that the Atom is
    constructed internally in an optimal fashion for the user.
    """

    def __init__(self, op: Op, trait: Trait) -> None:
        self._op = op
        self._trait = trait

    def __extract_mlir_values__(self):
        return extract_mlir_values(self._trait)

    def __new_from_mlir_values__(self, values):
        return self.__class__(self.op, new_from_mlir_values(self._trait, values))

    @property
    def op(self) -> Op:
        return self._op

    @property
    def type(self):
        return self._trait.value.type

    @dsl_user_op
    def set(self, modifier, value, *, loc=None, ip=None) -> None:
        """
        Sets runtime fields of the Atom.

        Some Atoms have runtime state, for example a tcgen05 MMA Atom


        .. code-block:: python

            tiled_mma = cute.make_tiled_mma(some_tcgen05_mma_op)
            tiled_mma.set(cute.nvgpu.tcgen05.Field.ACCUMULATE, True)

        The ``set`` method provides a way to the user to modify such runtime state. Modifiable
        fields are provided by arch-specific enumerations, for example ``tcgen05.Field``. The Atom
        instance internally validates the field as well as the value provided by the user to set
        the field to.
        """
        self._trait.set(modifier, value, loc=loc, ip=ip)

    def _unpack(self, *, loc=None, ip=None, **kwargs) -> ir.Value:
        return self._trait.unpack(loc=loc, ip=ip, **kwargs)


####################################################################################################
#
# MMA Atoms, TiledMma, and ThrMma
#
####################################################################################################


class MmaAtom(Atom):
    """
    The MMA Atom class.
    """

    def __str__(self) -> str:
        res = "MMA Atom\n"
        res += "  ThrID:       " + pretty_str(self.thr_id) + "\n"
        res += "  Shape MNK:   " + pretty_str(self.shape_mnk) + "\n"
        res += "  TV Layout A: " + pretty_str(self.tv_layout_A) + "\n"
        res += "  TV Layout B: " + pretty_str(self.tv_layout_B) + "\n"
        res += "  TV Layout C: " + pretty_str(self.tv_layout_C)
        return res

    #
    # Properties
    #

    @property
    def thr_id(self) -> Layout:
        return _cute_ir.static(self._trait.value.type.thr_id)

    @property
    def shape_mnk(self) -> Shape:
        return _unpack_x_tuple(self._trait.value.type.shape_mnk)

    @property
    def tv_layout_A(self) -> Layout:
        return _cute_ir.static(self._trait.value.type.layout_a_tv)

    @property
    def tv_layout_B(self) -> Layout:
        return _cute_ir.static(self._trait.value.type.layout_b_tv)

    @property
    def tv_layout_C(self) -> Layout:
        return _cute_ir.static(self._trait.value.type.layout_c_tv)

    #
    # make_fragment
    #

    @dsl_user_op
    def make_fragment_A(self, input, *, loc=None, ip=None):
        # input could be memref/shape/layout for tmem based fragment
        if isinstance(input, _Tensor):
            if self.op is not None:
                self.op._verify_fragment_A(input, loc=loc, ip=ip)
            input = input.value
        if isinstance(input, tuple):
            input = _pack_shape(input, loc=loc, ip=ip)
        return _cute_ir.mma_make_fragment(
            _cute_ir.MmaOperand.A,
            self._trait.value,
            input,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def make_fragment_B(self, input, *, loc=None, ip=None):
        if isinstance(input, _Tensor):
            if self.op is not None:
                self.op._verify_fragment_B(input, loc=loc, ip=ip)
            input = input.value
        return _cute_ir.mma_make_fragment(
            _cute_ir.MmaOperand.B,
            self._trait.value,
            input,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def make_fragment_C(self, input, *, loc=None, ip=None):
        # input could be memref/shape/layout for tmem based fragment
        if isinstance(input, _Tensor):
            input = input.value
        if isinstance(input, tuple):
            input = _pack_shape(input, loc=loc, ip=ip)
        return _cute_ir.mma_make_fragment(
            _cute_ir.MmaOperand.C,
            self._trait.value,
            input,
            loc=loc,
            ip=ip,
        )


class TiledMma(MmaAtom):
    """
    The tiled MMA class.
    """

    def __str__(self) -> str:
        res = "Tiled MMA\n"
        res += "  Thr Layout VMNK: " + pretty_str(self.thr_layout_vmnk) + "\n"
        res += "  Permutation MNK: " + pretty_str(self.permutation_mnk) + "\n"
        res += "MMA Atom\n"
        res += "  ThrID:           " + pretty_str(self.thr_id) + "\n"
        res += "  Shape MNK:       " + pretty_str(self.shape_mnk) + "\n"
        res += "  TV Layout A:     " + pretty_str(self.tv_layout_A) + "\n"
        res += "  TV Layout B:     " + pretty_str(self.tv_layout_B) + "\n"
        res += "  TV Layout C:     " + pretty_str(self.tv_layout_C)
        return res

    #
    # Properties
    #

    @property
    def tv_layout_A_tiled(self) -> Layout:
        return _cute_ir.static(self._trait.value.type.layout_a_tv_tiled)

    @property
    def tv_layout_B_tiled(self) -> Layout:
        return _cute_ir.static(self._trait.value.type.layout_b_tv_tiled)

    @property
    def tv_layout_C_tiled(self) -> Layout:
        return _cute_ir.static(self._trait.value.type.layout_c_tv_tiled)

    @property
    def permutation_mnk(self) -> Tile:
        return _unpack_x_tuple(self._trait.value.type.permutation_mnk)

    @property
    def thr_layout_vmnk(self) -> Layout:
        return _cute_ir.static(self._trait.value.type.thr_layout_vmnk)

    @property
    def size(self) -> int:
        return self._trait.value.type.size

    #
    # Tiler
    #

    def get_tile_size(self, mode_idx: int) -> Shape:
        assert (mode_idx >= 0) and (mode_idx < 3)
        perm_tile = self.permutation_mnk[mode_idx]
        if perm_tile is None:
            thr_layout_vmnk = self.thr_layout_vmnk
            atom_shape_mnk = self.shape_mnk
            return size(atom_shape_mnk, mode=[mode_idx]) * size(
                thr_layout_vmnk, mode=[mode_idx + 1]
            )
        else:
            return size(perm_tile)

    #
    # get_slice
    #

    def get_slice(self, thr_idx: Union[int, Int32]) -> "ThrMma":
        return ThrMma(self.op, self._trait, thr_idx)

    #
    # partition_shape
    #

    def _partition_shape(self, operand_id, shape, *, loc=None, ip=None):
        shape = _pack_shape(shape, loc=loc, ip=ip)
        return _unpack_x_tuple(
            _cute_ir.tiled_mma_partition_shape(
                operand_id, self._trait.value, shape, loc=loc, ip=ip
            ),
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def partition_shape_A(self, shape_mk, *, loc=None, ip=None):
        return self._partition_shape(_cute_ir.MmaOperand.A, shape_mk, loc=loc, ip=ip)

    @dsl_user_op
    def partition_shape_B(self, shape_nk, *, loc=None, ip=None):
        return self._partition_shape(_cute_ir.MmaOperand.B, shape_nk, loc=loc, ip=ip)

    @dsl_user_op
    def partition_shape_C(self, shape_mn, *, loc=None, ip=None):
        return self._partition_shape(_cute_ir.MmaOperand.C, shape_mn, loc=loc, ip=ip)

    #
    # _thrfrg
    #

    @overload
    def _thrfrg(self, operand_id, input: Layout, *, loc=None, ip=None) -> Layout: ...

    @overload
    def _thrfrg(self, operand_id, input: Tensor, *, loc=None, ip=None) -> Tensor: ...

    def _thrfrg(self, operand_id, input, *, loc=None, ip=None) -> Union[Tensor, Layout]:
        if isinstance(input, Tensor):
            return make_tensor(
                input.iterator,
                self._thrfrg(operand_id, input.layout, loc=loc, ip=ip),
            )
        elif isinstance(input, Layout):
            if not is_static(input.type):
                raise ValueError(f"Expects a static layout but got {input.type}")
            return _cute_ir.static(
                self._trait.value.type.thrfrg(operand_id, input), loc=loc, ip=ip
            )

        raise ValueError(
            f"Expects a layout or a tensor as input but got {type(input)=}"
        )

    def _thrfrg_A(
        self, input: Union[Layout, Tensor], *, loc=None, ip=None
    ) -> Union[Layout, Tensor]:
        return self._thrfrg(_cute_ir.MmaOperand.A, input, loc=loc, ip=ip)

    def _thrfrg_B(
        self, input: Union[Layout, Tensor], *, loc=None, ip=None
    ) -> Union[Layout, Tensor]:
        return self._thrfrg(_cute_ir.MmaOperand.B, input, loc=loc, ip=ip)

    def _thrfrg_C(
        self, input: Union[Layout, Tensor], *, loc=None, ip=None
    ) -> Union[Layout, Tensor]:
        return self._thrfrg(_cute_ir.MmaOperand.C, input, loc=loc, ip=ip)


class ThrMma(TiledMma):
    """
    The thread MMA class for modeling a thread-slice of a tiled MMA.
    """

    def __init__(self, op: Op, trait: Trait, thr_idx: Union[int, Int32]) -> None:
        super().__init__(op, trait)
        self._thr_idx = thr_idx

    def __new_from_mlir_values__(self, values):
        return self.__class__(
            self.op, new_from_mlir_values(self._trait, values), self.thr_idx
        )

    @property
    def thr_idx(self):
        return self._thr_idx

    @dsl_user_op
    def partition_A(self, input_mk: Tensor, *, loc=None, ip=None) -> Tensor:
        thr_idx = _pack_coord(self.thr_idx, loc=loc, ip=ip)
        return _cute_ir.tiled_mma_partition(
            _cute_ir.MmaOperand.A,
            self._trait.value,
            input_mk.value,
            thr_idx,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def partition_B(self, input_nk: Tensor, *, loc=None, ip=None) -> Tensor:
        thr_idx = _pack_coord(self.thr_idx, loc=loc, ip=ip)
        return _cute_ir.tiled_mma_partition(
            _cute_ir.MmaOperand.B,
            self._trait.value,
            input_nk.value,
            thr_idx,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def partition_C(self, input_mn: Tensor, *, loc=None, ip=None) -> Tensor:
        thr_idx = _pack_coord(self.thr_idx, loc=loc, ip=ip)
        return _cute_ir.tiled_mma_partition(
            _cute_ir.MmaOperand.C,
            self._trait.value,
            input_mn.value,
            thr_idx,
            loc=loc,
            ip=ip,
        )


@dsl_user_op
def make_mma_atom(op: MmaOp, *, loc=None, ip=None, **kwargs) -> MmaAtom:
    """
    Makes an MMA Atom from an MMA Operation.

    This function creates an MMA Atom from a given MMA Operation. Arbitrary kw arguments can be
    provided for Op-specific additional parameters. They are not used as of today.

    :param op: The MMA Operation to construct an Atom for
    :type op:  MmaOp
    :return:   The MMA Atom
    :rtype:    MmaAtom
    """
    trait = op._make_trait(loc=loc, ip=ip, **kwargs)
    return MmaAtom(op, trait)


@dsl_user_op
def make_tiled_mma(
    op_or_atom: Union[Op, MmaAtom],
    atom_layout_mnk=(1, 1, 1),
    permutation_mnk=None,
    *,
    loc=None,
    ip=None,
    **kwargs,
) -> TiledMma:
    """
    Makes a tiled MMA from an MMA Operation or an MMA Atom.

    :param op_or_atom:      The MMA Operation or Atom
    :type op_or_atom:       Union[Op, MmaAtom]
    :param atom_layout_mnk: A Layout describing the tiling of Atom across threads
    :type atom_layout_mnk:  Layout
    :param permutation_mnk: A permutation Tiler describing the tiling of Atom across values including any permutation of such tiling
    :type permutation_mnk:  Tiler
    :return:                The resulting tiled MMA
    :rtype:                 TiledMma
    """
    if isinstance(op_or_atom, Op):
        op = op_or_atom
        atom = make_mma_atom(op_or_atom, loc=loc, ip=ip, **kwargs)
    elif isinstance(op_or_atom, MmaAtom):
        op = op_or_atom.op
        atom = op_or_atom
    else:
        raise TypeError(
            f"expected an MMA Op or Atom, but got an instance of {type(op_or_atom)}"
        )
    if isinstance(atom_layout_mnk, tuple):
        atom_layout_mnk = make_layout(atom_layout_mnk, loc=loc, ip=ip)
    if rank(atom_layout_mnk) != 3:
        raise ValueError(f"expects rank-3 MNK atom layout, but got {atom_layout_mnk}")
    permutation_mnk_ty = None
    if permutation_mnk is not None:
        permutation_mnk_ty = _pack_tile(permutation_mnk, loc=loc, ip=ip).type
    ty = _cute_nvgpu_ir.TiledMmaType.get(
        atom._trait.value.type,
        atom_layout_mnk.type,
        permutation_mnk_ty,
    )
    val = _cute_ir.make_tiled_mma(ty, atom._trait.value, loc=loc, ip=ip)
    # Instead of modifying atom which might have been provided by the user, create a brand new
    # trait instance and replace the Atom ir.Value with the tiled one
    trait = new_from_mlir_values(atom._trait, [val])
    return TiledMma(op, trait)


####################################################################################################
#
# Copy Atoms, TiledCopy, and ThrCopy
#
####################################################################################################


class CopyAtom(Atom):
    """
    The Copy Atom class.
    """

    def __str__(self) -> str:
        res = "Copy Atom\n"
        res += "  ThrID:         " + str(self.thr_id) + "\n"
        res += "  TV Layout Src: " + str(self.layout_src_tv) + "\n"
        res += "  TV Layout Dst: " + str(self.layout_dst_tv) + "\n"
        res += "  Value type:    " + str(self._trait.value.type.value_type)
        return res

    #
    # Properties
    #

    @property
    def value_type(self) -> Type[Numeric]:
        return Numeric.from_mlir_type(self._trait.value.type.value_type)

    @property
    def thr_id(self) -> Layout:
        return _cute_ir.static(self._trait.value.type.thr_id)

    @property
    def layout_src_tv(self) -> Layout:
        return _cute_ir.static(self._trait.value.type.layout_src_tv)

    @property
    def layout_dst_tv(self) -> Layout:
        return _cute_ir.static(self._trait.value.type.layout_dst_tv)


class TiledCopy(CopyAtom):
    """
    The tiled Copy class.
    """

    def __str__(self) -> str:
        res = "Tiled Copy\n"
        res += "  Tiler MN:        " + pretty_str(self.tiler_mn) + "\n"
        res += "  TV Layout tiled: " + str(self.layout_tv_tiled) + "\n"
        res += "Copy Atom\n"
        res += "  ThrID:           " + str(self.thr_id) + "\n"
        res += "  TV Layout Src:   " + str(self.layout_src_tv) + "\n"
        res += "  TV Layout Dst:   " + str(self.layout_dst_tv) + "\n"
        res += "  Value type:      " + str(self._trait.value.type.value_type)
        return res

    #
    # Properties
    #

    @property
    def layout_tv_tiled(self) -> Layout:
        return _cute_ir.static(self._trait.value.type.layout_tv_tiled)

    @property
    def tiler_mn(self) -> Tile:
        return _unpack_x_tuple(self._trait.value.type.tiler_mn)

    @property
    def layout_src_tv_tiled(self) -> Layout:
        return _cute_ir.static(self._trait.value.type.layout_src_tv_tiled)

    @property
    def layout_dst_tv_tiled(self) -> Layout:
        return _cute_ir.static(self._trait.value.type.layout_dst_tv_tiled)

    @property
    def size(self) -> int:
        return self._trait.value.type.size

    #
    # get_slice and retile
    #

    def get_slice(self, thr_idx: Union[int, Int32]) -> "ThrCopy":
        return ThrCopy(self.op, self._trait, thr_idx)

    @dsl_user_op
    def retile(self, src, *, loc=None, ip=None):
        return _cute_ir.tiled_copy_retile(
            tiled_copy=self._trait.value, input=src.value, loc=loc, ip=ip
        )


class ThrCopy(TiledCopy):
    """
    The thread Copy class for modeling a thread-slice of a tiled Copy.
    """

    def __init__(self, op: Op, trait: Trait, thr_idx: Union[int, Int32]) -> None:
        super().__init__(op, trait)
        self._thr_idx = thr_idx

    def __new_from_mlir_values__(self, values):
        return self.__class__(
            self.op, new_from_mlir_values(self._trait, values), self.thr_idx
        )

    @property
    def thr_idx(self):
        return self._thr_idx

    @dsl_user_op
    def partition_S(self, src: Tensor, *, loc=None, ip=None) -> Tensor:
        thr_idx = _pack_coord(self.thr_idx, loc=loc, ip=ip)
        return _cute_ir.tiled_copy_partition_S(
            self._trait.value, src.value, thr_idx, loc=loc, ip=ip
        )

    @dsl_user_op
    def partition_D(self, dst: Tensor, *, loc=None, ip=None) -> Tensor:
        thr_idx = _pack_coord(self.thr_idx, loc=loc, ip=ip)
        return _cute_ir.tiled_copy_partition_D(
            self._trait.value, dst.value, thr_idx, loc=loc, ip=ip
        )


@dsl_user_op
def make_copy_atom(
    op: CopyOp, copy_internal_type: Type[Numeric], *, loc=None, ip=None, **kwargs
) -> CopyAtom:
    """
    Makes a Copy Atom from a Copy Operation.

    This function creates a Copy Atom from a given Copy Operation. Arbitrary kw arguments can be
    provided for Op-specific additional parameters.

    Example:

    .. code-block:: python

        op = cute.nvgpu.CopyUniversalOp()
        atom = cute.make_copy_atom(op, tensor_dtype, num_bits_per_copy=64)

    :param op:                 The Copy Operation to construct an Atom for
    :type op:                  CopyOp
    :param copy_internal_type: An internal data type used to construct the source/destination layouts in unit of tensor elements
    :type copy_internal_type:  Type[Numeric]
    :return:                   The Copy Atom
    :rtype:                    CopyAtom
    """
    trait = op._make_trait(copy_internal_type, loc=loc, ip=ip, **kwargs)
    return CopyAtom(op, trait)


@dsl_user_op
def make_layout_tv(
    thr_layout: Layout, val_layout: Layout, *, loc=None, ip=None
) -> Tuple[Shape, Layout]:
    """Create a thread-value layout for partitioning data tensors.

    This function creates a thread-value layout that maps between ``(thread_idx, value_idx)``
    coordinates and logical ``(M,N)`` coordinates. The thread layout must be compact to ensure
    proper partitioning.

    This implements the thread-value partitioning pattern shown in
    Figure TVLayout, where data is partitioned across threads and values within each thread.

    :param thr_layout: Layout mapping from ``(TileM,TileN)`` coordinates to thread IDs (must be compact)
    :type thr_layout: Layout
    :param val_layout: Layout mapping from ``(ValueM,ValueN)`` coordinates to value IDs within each thread
    :type val_layout: Layout
    :param loc: Source location for MLIR, defaults to None
    :type loc: Optional[Location], optional
    :param ip: Insertion point, defaults to None
    :type ip: Optional[InsertionPoint], optional

    :return: A tuple containing ``tiler_mn`` and ``layout_tv``
    :rtype: Tuple[Shape, Layout]

    where:
        * ``tiler_mn`` is tiler and ``shape(tiler_mn)`` is compatible with ``shape(zipped_divide(x, tiler_mn))[0]``
        * ``layout_tv``: Thread-value layout mapping (thread_idx, value_idx) -> (M,N)

    **Example:**

    .. code-block:: python

        tiler_mn, layout_tv = cute.make_layout_tv(
            cute.make_layout((4, 8), stride=(8, 1)), cute.make_layout(2, stride=1)
        )

    Above code creates a TV layout that maps between thread/value coordinates
    and the logical coordinates in a 8x8 matrix with:

    * thread block layout ``(4,8):(8,1)``
    * 2 elements per thread
    """

    if not isinstance(thr_layout, Layout):
        raise TypeError(f"expected a Layout for thr_layout, but got {type(thr_layout)}")
    if not isinstance(val_layout, Layout):
        raise TypeError(f"expected a Layout for val_layout, but got {type(val_layout)}")

    # Take the raked_products to compute the Layout_MN
    # (M,N) -> (thr_idx, val_idx)
    layout_mn = raked_product(thr_layout, val_layout, loc=loc, ip=ip)
    thr_size = size(thr_layout, loc=loc, ip=ip)
    val_size = size(val_layout, loc=loc, ip=ip)
    tmp = make_layout((thr_size, val_size), loc=loc, ip=ip)
    # (thr_idx, val_idx) -> (M,N)
    layout_tv = composition(
        right_inverse(layout_mn, loc=loc, ip=ip), tmp, loc=loc, ip=ip
    )

    tiler_mn = product_each(layout_mn.shape, loc=loc, ip=ip)

    return (tiler_mn, layout_tv)


def _make_tiled_copy(atom, layout_tv, tiler_mn, *, loc=None, ip=None):
    if type(tiler_mn) is tuple:
        tiler_mn = _pack_tile(tiler_mn, loc=loc, ip=ip)

    assert isinstance(tiler_mn, ir.Value) and _cute_ir.TileType.isinstance(
        tiler_mn.type
    ), f"tiler_mn must be a Tile, but got {type(tiler_mn)}"
    assert is_static(layout_tv.type) and is_static(
        tiler_mn.type
    ), "layout tv and tiler mn must be static"
    tiled_copy_ty = _cute_nvgpu_ir.TiledCopyType.get(
        atom.type, layout_tv.type, tiler_mn.type
    )

    val = _cute_ir.make_tiled_copy(tiled_copy_ty, atom._trait.value, loc=loc, ip=ip)
    # Instead of modifying atom which might have been provided by the user, create a brand new
    # trait instance and replace the Atom ir.Value with the tiled one
    trait = new_from_mlir_values(atom._trait, [val])
    return TiledCopy(atom.op, trait)


def make_tiled_copy(atom, layout_tv, tiler_mn, *, loc=None, ip=None):
    """Create a tiled type given a TV partitioner and tiler.

    :param atom: Copy atom, e.g. smit_copy and simt_async_copy, tma_load, etc.
    :type atom: CopyAtom
    :param layout_tv: Thread-value layout
    :type layout_tv: Layout
    :param tiler_mn: Tile size
    :type tiler_mn: Tiler
    :param loc: Source location for MLIR, defaults to None
    :type loc: Optional[Location], optional
    :param ip: Insertion point, defaults to None
    :type ip: Optional[InsertionPoint], optional

    :return: A tiled copy for the partitioner
    :rtype: TiledCopy
    """
    return _make_tiled_copy(atom, layout_tv, tiler_mn, loc=loc, ip=ip)


@dsl_user_op
def make_tiled_copy_tv(
    atom: CopyAtom, thr_layout: Layout, val_layout: Layout, *, loc=None, ip=None
) -> TiledCopy:
    """Create a tiled copy given separate thread and value layouts.

    A TV partitioner is inferred based on the input layouts. The input thread layout
    must be compact.

    :param atom: Copy atom
    :type atom: CopyAtom
    :param thr_layout: Layout mapping from ``(TileM,TileN)`` coordinates to thread IDs (must be compact)
    :type thr_layout: Layout
    :param val_layout: Layout mapping from ``(ValueM,ValueN)`` coordinates to value IDs
    :type val_layout: Layout
    :param loc: Source location for MLIR, defaults to None
    :type loc: Optional[Location], optional
    :param ip: Insertion point, defaults to None
    :type ip: Optional[InsertionPoint], optional

    :return: A tiled copy for the partitioner
    :rtype: TiledCopy
    """

    tiler_mn, layout_tv = make_layout_tv(thr_layout, val_layout, loc=loc, ip=ip)
    tiler_mn = _pack_tile(product_each(tiler_mn, loc=loc, ip=ip), loc=loc, ip=ip)
    return _make_tiled_copy(atom, layout_tv, tiler_mn, loc=loc, ip=ip)


@dsl_user_op
def make_tiled_copy_A(atom, tiled_mma, *, loc=None, ip=None):
    """Create a tiled copy out of the copy_atom that matches the A-Layout of tiled_mma.

    :param atom: Copy atom
    :type atom: CopyAtom
    :param tiled_mma: Tiled MMA
    :type tiled_mma: TiledMma
    :param loc: Source location for MLIR, defaults to None
    :type loc: Optional[Location], optional
    :param ip: Insertion point, defaults to None
    :type ip: Optional[InsertionPoint], optional

    :return: A tiled copy for the partitioner
    :rtype: TiledCopy
    """

    return _make_tiled_copy(
        atom,
        tiled_mma.tv_layout_A_tiled,
        (tiled_mma.get_tile_size(0), tiled_mma.get_tile_size(2)),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def make_tiled_copy_B(atom, tiled_mma, *, loc=None, ip=None):
    """Create a tiled copy out of the copy_atom that matches the B-Layout of tiled_mma.

    :param atom: Copy atom
    :type atom: CopyAtom
    :param tiled_mma: Tiled MMA
    :type tiled_mma: TiledMma
    :param loc: Source location for MLIR, defaults to None
    :type loc: Optional[Location], optional
    :param ip: Insertion point, defaults to None
    :type ip: Optional[InsertionPoint], optional

    :return: A tiled copy for the partitioner
    :rtype: TiledCopy
    """

    return _make_tiled_copy(
        atom,
        tiled_mma.tv_layout_B_tiled,
        (tiled_mma.get_tile_size(1), tiled_mma.get_tile_size(2)),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def make_tiled_copy_C(atom, tiled_mma, *, loc=None, ip=None):
    """Create a tiled copy out of the copy_atom that matches the C-Layout of tiled_mma.

    :param atom: Copy atom
    :type atom: CopyAtom
    :param tiled_mma: Tiled MMA
    :type tiled_mma: TiledMma
    :param loc: Source location for MLIR, defaults to None
    :type loc: Optional[Location], optional
    :param ip: Insertion point, defaults to None
    :type ip: Optional[InsertionPoint], optional

    :return: A tiled copy for the partitioner
    :rtype: TiledCopy
    """

    return _make_tiled_copy(
        atom,
        tiled_mma.tv_layout_C_tiled,
        (tiled_mma.get_tile_size(0), tiled_mma.get_tile_size(1)),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def make_tiled_copy_S(atom, tiled_copy, *, loc=None, ip=None):
    """Create a tiled copy out of the copy_atom that matches the Src-Layout of tiled_copy.

    :param atom: Copy atom
    :type atom: CopyAtom
    :param tiled_copy: Tiled copy
    :type tiled_copy: TiledCopy
    :param loc: Source location for MLIR, defaults to None
    :type loc: Optional[Location], optional
    :param ip: Insertion point, defaults to None
    :type ip: Optional[InsertionPoint], optional

    :return: A tiled copy for the partitioner
    :rtype: TiledCopy
    """

    return _make_tiled_copy(
        atom, tiled_copy.layout_src_tv_tiled, tiled_copy.tiler_mn, loc=loc, ip=ip
    )


@dsl_user_op
def make_tiled_copy_D(atom, tiled_copy, *, loc=None, ip=None):
    """Create a tiled copy out of the copy_atom that matches the Dst-Layout of tiled_copy.

    :param atom: Copy atom
    :type atom: CopyAtom
    :param tiled_copy: Tiled copy
    :type tiled_copy: TiledCopy
    :param loc: Source location for MLIR, defaults to None
    :type loc: Optional[Location], optional
    :param ip: Insertion point, defaults to None
    :type ip: Optional[InsertionPoint], optional

    :return: A tiled copy for the partitioner
    :rtype: TiledCopy
    """

    return _make_tiled_copy(
        atom, tiled_copy.layout_dst_tv_tiled, tiled_copy.tiler_mn, loc=loc, ip=ip
    )


@dsl_user_op
def make_tiled_copy_C_atom(atom: CopyAtom, mma: TiledMma, *, loc=None, ip=None):
    """Create the smallest tiled copy that can retile LayoutC_TV for use with pipelined epilogues with subtiled stores.

    :param atom: Copy atom
    :type atom: CopyAtom
    :param mma: Tiled MMA
    :type mma: TiledMma
    :param loc: Source location for MLIR, defaults to None
    :type loc: Optional[Location], optional
    :param ip: Insertion point, defaults to None
    :type ip: Optional[InsertionPoint], optional

    :return: A tiled copy for partitioner
    :rtype: TiledCopy

    :raises ValueError: If the number value of CopyAtom's source layout is greater than the size of TiledMma's LayoutC_TV
    """
    # Truncate the V-layout to just the Copy_Atom, keep the V-order
    layoutC_tv = mma.tv_layout_C_tiled
    val_layout_src = atom.layout_src_tv
    num_val_src = size(val_layout_src, mode=[1], loc=loc, ip=ip)
    num_val_layoutC_tv = size(layoutC_tv, mode=[1], loc=loc, ip=ip)
    if num_val_src > num_val_layoutC_tv:
        raise ValueError(
            f"The number value of CopyAtom's source layout {num_val_src} "
            f"is greater than the size of TiledMma's LayoutC_TV {num_val_layoutC_tv}"
        )
    layout_TV = composition(
        layoutC_tv,
        make_layout(
            (size(layoutC_tv, mode=[0], loc=loc, ip=ip), num_val_src), loc=loc, ip=ip
        ),
        loc=loc,
        ip=ip,
    )

    # Recompute tiler and restride the TV layout for the new tiler

    # Tiler -- Find the active elements in the MMA tensor and generate a tiler to extract them
    # Convert to the awkward by-mode tiler to preserve the modes of the tiled MMA
    mma_tiler = (mma.get_tile_size(0), mma.get_tile_size(1))

    tiler_0 = filter(
        composition(
            make_layout(mma_tiler, stride=(1, 0), loc=loc, ip=ip),
            layout_TV,
            loc=loc,
            ip=ip,
        ),
        loc=loc,
        ip=ip,
    )
    tiler_1 = filter(
        composition(
            make_layout(mma_tiler, stride=(0, 1), loc=loc, ip=ip),
            layout_TV,
            loc=loc,
            ip=ip,
        ),
        loc=loc,
        ip=ip,
    )
    tiler = (tiler_0, tiler_1)

    tile2mma = composition(
        make_layout(mma_tiler, loc=loc, ip=ip), tiler, loc=loc, ip=ip
    )
    layout_tv = composition(
        left_inverse(tile2mma, loc=loc, ip=ip), layout_TV, loc=loc, ip=ip
    )

    tiler_mn = _pack_tile(tiler, loc=loc, ip=ip)

    return _make_tiled_copy(atom, layout_tv, tiler_mn, loc=loc, ip=ip)


####################################################################################################
#
# cute.gemm and cute.copy
#
####################################################################################################


@dsl_user_op
def gemm(
    atom: MmaAtom,
    d: Tensor,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    *,
    loc=None,
    ip=None,
    **kwargs,
) -> None:
    """The GEMM algorithm.

    Computes ``D <- A * B + C`` where ``C`` and ``D`` can alias. Note that some MMA Atoms (e.g.
    warpgroup-wide or tcgen05 MMAs) require manually setting an "accumulate" boolean field.

    All tensors must be partitioned according to the provided MMA Atom.

    For MMA Atoms that require single-threaded execution, the gemm op automatically handles thread
    election internally. Manual thread selection is not required in such cases.

    Following dispatch rules are supported:

    - Dispatch [1]: (V) x (V) => (V)          => (V,1,1) x (V,1,1) => (V,1,1)
    - Dispatch [2]: (M) x (N) => (M,N)        => (1,M,1) x (1,N,1) => (1,M,N)
    - Dispatch [3]: (M,K) x (N,K) => (M,N)    => (1,M,K) x (1,N,K) => (1,M,N)
    - Dispatch [4]: (V,M) x (V,N) => (V,M,N)  => (V,M,1) x (V,N,1) => (V,M,N)
    - Dispatch [5]: (V,M,K) x (V,N,K) => (V,M,N)

    :param atom: MMA atom
    :type atom: MmaAtom
    :param d: Destination tensor
    :type d: Tensor
    :param a: First source tensor
    :type a: Tensor
    :param b: Second source tensor
    :type b: Tensor
    :param c: Third source tensor
    :type c: Tensor
    :param loc: Source location for MLIR, defaults to None
    :type loc: Optional[Location], optional
    :param ip: Insertion point for MLIR, defaults to None
    :type ip: Optional[InsertionPoint], optional
    :param kwargs: Additional keyword arguments
    :type kwargs: dict
    :return: None
    :rtype: None
    """

    a_rank = rank(a.shape)
    b_rank = rank(b.shape)
    c_rank = rank(c.shape)
    d_rank = rank(d.shape)

    if a_rank != b_rank:
        raise ValueError("`a` and `b` must have the same rank")

    if c_rank != d_rank:
        raise ValueError("`c` and `d` must have the same rank")

    if a_rank == 1:
        if c_rank > 2:
            raise ValueError("`c` must have rank <= 2 when `a` has rank 1")
    elif a_rank == 2:
        if c_rank not in (2, 3):
            raise ValueError("`c` must have rank 2 or 3 when `a` has rank 2")
    elif a_rank == 3:
        if c_rank != 3:
            raise ValueError("`c` must have rank 3 when `a` has rank 3")

    value = atom._unpack(loc=loc, ip=ip, **kwargs)
    return _cute_ir.gemm(value, d.value, a.value, b.value, c.value, loc=loc, ip=ip)


@dsl_user_op
def basic_copy(src: Tensor, dst: Tensor, *, loc=None, ip=None) -> None:
    """Performs a basic element-wise copy.

    This functions **assumes** the following pre-conditions:
    1. `size(src) == size(dst)`

    When the `src` and `dst` shapes are static, the pre-conditions are actually verified and the
    element-wise loop is fully unrolled.

    :param src: Source tensor
    :type src: Tensor
    :param dst: Destination tensor
    :type dst: Tensor
    :param loc: Source location for MLIR, defaults to None
    :type loc: Optional[Location], optional
    :param ip: Insertion point, defaults to None
    :type ip: Optional[InsertionPoint], optional
    """

    if is_static(src.shape) and is_static(dst.shape):
        simt_copy_ty = _cute_nvgpu_ir.CopyAtomSIMTSyncCopyType.get(
            src.element_type.mlir_type, src.element_type.width
        )
        simt_copy = _cute_ir.atom(simt_copy_ty, loc=loc, ip=ip)
        return _cute_ir.copy(simt_copy, src.value, dst.value, loc=loc, ip=ip)

    s = size(dst, loc=loc, ip=ip)
    # Always generate an scf.for Op when one of the tensors is dynamic
    for i in for_generate(0, s):
        dst[i] = src[i]
        yield_out()


@dsl_user_op
def basic_copy_if(pred: Tensor, src: Tensor, dst: Tensor, *, loc=None, ip=None) -> None:
    """Performs a basic predicated element-wise copy.

    This functions **assumes** the following pre-conditions:
    1. `size(src) == size(dst)`
    2. `size(src) == size(pred)`

    When all shapes are static, the pre-conditions are actually verified and the element-wise loop
    is fully unrolled.

    """
    if src.element_type.width != dst.element_type.width:
        raise NotImplementedError(
            "basic_copy_if currently only supports equal source and destination "
            "element type bit width"
        )

    if is_static(src.shape) and is_static(dst.shape) and is_static(pred.shape):
        return _basic_copy_if_static(pred, src, dst, loc=loc, ip=ip)

    s = size(dst, loc=loc, ip=ip)
    # Always generate an scf.for Op when one of the tensors is dynamic
    for i in for_generate(0, s):
        if_generate(pred[i], lambda: dst.__setitem__(i, src[i]))
        yield_out()


# Version of basic_copy_if when src and dst have static shapes
# - verify size(src) == size(dst) == size(prd)
# - fully unroll the loop for now
def _basic_copy_if_static(
    pred: Tensor, src: Tensor, dst: Tensor, *, loc=None, ip=None
) -> None:
    assert is_static(src.shape) and is_static(dst.shape) and is_static(pred.shape)
    if size(src, loc=loc, ip=ip) != size(dst, loc=loc, ip=ip):
        raise ValueError(
            "basic_copy expects the size of source, destination, and predicate tensors to match"
        )
    # Fully unrolled loop in the static case for now
    for i in range(size(dst, loc=loc, ip=ip)):
        if_generate(pred[i], lambda: dst.__setitem__(i, src[i]))


@dsl_user_op
def autovec_copy(src: Tensor, dst: Tensor, *, loc=None, ip=None) -> None:
    """
    Auto-vectorizing SIMT copy policy.

    Given a source and destination tensors that are statically shaped, this policy figures out the
    largest safe vector width that the copy instruction can take and performs the copy.
    """
    if src.element_type.width != dst.element_type.width:
        raise NotImplementedError(
            "autovec_copy currently only supports equal source and destination "
            "element type bit width"
        )

    # We are going to dispatch to copy-with-atom which requires shapes to be static
    if not is_static(src.shape) or not is_static(dst.shape):
        raise ValueError(
            "autovec_copy expects source and destination tensors to be statically shaped"
        )

    vec_layout = max_common_layout(src, dst, loc=loc, ip=ip)
    num_common_elements = size(vec_layout, loc=loc, ip=ip)

    # Next we construct an upper-bound on the number bits that can be vectorized by considering
    # - the maximum alignment of the layouts
    # - the maximum alignment of the pointers

    upper_bound = math.gcd(src.layout.max_alignment, dst.layout.max_alignment)
    upper_bound = math.gcd(upper_bound, num_common_elements)
    upper_bound *= src.element_type.width

    # For our instructions, the alignment of the pointer is an upper bound to the vector width
    # max_alignment, as opposed to alignment, takes into account possible address swizzling
    upper_bound = math.gcd(upper_bound, src.iterator.max_alignment * 8)
    upper_bound = math.gcd(upper_bound, dst.iterator.max_alignment * 8)

    # Finally, we put a cap at 128b
    num_bits_per_copy = math.gcd(upper_bound, 128)

    if (num_common_elements > 1) and (num_bits_per_copy % 8 == 0):
        num_common_elements = num_bits_per_copy // src.element_type.width

        # 2 step logical divides ensuring that the divides are valid at every step
        vec_src = logical_divide(src, vec_layout, loc=loc, ip=ip)
        vec_dst = logical_divide(dst, vec_layout, loc=loc, ip=ip)
        tiled_src = logical_divide(
            vec_src, make_layout(num_common_elements, loc=loc, ip=ip), loc=loc, ip=ip
        )
        tiled_dst = logical_divide(
            vec_dst, make_layout(num_common_elements, loc=loc, ip=ip), loc=loc, ip=ip
        )

        # Dispatch to copy with atom
        simt_type = _cute_nvgpu_ir.CopyAtomSIMTSyncCopyType.get(
            src.element_type.mlir_type, num_bits_per_copy
        )
        simt_copy = _cute_ir.atom(simt_type, loc=loc, ip=ip)
        return _cute_ir.copy(
            simt_copy, tiled_src.value, tiled_dst.value, loc=loc, ip=ip
        )

    # Failed to vectorize, use a basic copy
    basic_copy(src, dst, loc=loc, ip=ip)


@dsl_user_op
def copy(
    atom: CopyAtom,
    src: Tensor,
    dst: Tensor,
    *,
    pred: Optional[Tensor] = None,
    loc=None,
    ip=None,
    **kwargs,
) -> None:
    """
    The Copy algorithm.

    The "copy with Atom" expects source and destination tensors to be partitioned according to the
    provided Copy Atom. Some Atoms require additional Op-specific kw arguments, for example TMA
    copies:

    .. code-block:: python

        cute.copy(tma_atom, src, dst, tma_bar_ptr=mbar_ptr, mcast_mask=mask)

    An additional predication tensor can be provided. If the partitioned tensors have the following
    logical profile ``((ATOM_V,ATOM_REST),REST_M,...)``, the predication tensor must have a profile
    consistent with ``(ATOM_REST,REST_M,...)``.

    For Copy Atoms that require single-threaded execution, the copy op automatically handles thread
    election internally. Manual thread selection is not required in such cases.
    """
    if isinstance(src.type, _cute_ir.MemRefType) and isinstance(
        dst.type, _cute_ir.MemRefType
    ):
        if src.element_type.width != dst.element_type.width:
            raise TypeError(
                "`copy` currently only supports equal source and destination "
                "element type bit width"
            )

    value = atom._unpack(loc=loc, ip=ip, **kwargs)
    if isinstance(pred, Tensor):
        pred = pred.value
    return _cute_ir.copy(value, src.value, dst.value, pred=pred, loc=loc, ip=ip)


@dsl_user_op
def copy_atom_call(
    atom: CopyAtom,
    src: Tensor,
    dst: Tensor,
    *,
    pred: Optional[Tensor] = None,
    loc=None,
    ip=None,
    **kwargs,
) -> None:
    """
    Execute a single copy atom operation.

    The copy_atom_call operation executes a copy atom with the given operands.
    Following src/dst layout of atom are valid:
    * ((atom_v))
    * (atom_v)

    Note: The format ((atom_v, rest_v)) is NOT valid for copy_atom_call since it would
    require multiple atom operations, which contradicts the definition of a single copy atom call.

    Examples:

    .. code-block:: python

        # Call a copy atom operation
        cute.copy_atom_call(copy_atom, src_tensor, dst_tensor)

    An additional predication tensor can be provided. If the partitioned tensors have the following
    logical profile ``((ATOM_V,ATOM_REST),REST_M,...)``, the predication tensor must have a profile
    consistent with ``(ATOM_REST,REST_M,...)``.
    """
    if isinstance(src.type, _cute_ir.MemRefType) and isinstance(
        dst.type, _cute_ir.MemRefType
    ):
        if src.element_type.width != dst.element_type.width:
            raise TypeError(
                "`copy_atom_call` currently only supports equal source and destination "
                "element type bit width"
            )

    value = atom._unpack(loc=loc, ip=ip, **kwargs)
    if isinstance(pred, Tensor):
        pred = pred.value
    return _cute_ir.copy_atom_call(
        value, src.value, dst.value, pred=pred, loc=loc, ip=ip
    )


def prefetch(atom: CopyAtom, src: Tensor, *, loc=None, ip=None) -> None:
    """
    The Prefetch algorithm.

    The "prefetch" expects source tensors to be partitioned according to the provided Copy Atom.
    Prefetch is used for loading tensors from global memory to L2.

    Prefetch accepts Copy Atom but not all are allowed. Currently, only support for tma load tensor prefetch.

    .. code-block:: python

        cute.prefetch(tma_atom, src)

    For Copy Atoms that require single-threaded execution, the copy op automatically handles thread
    election internally. Manual thread selection is not required in such cases.
    """
    dummy_tma_bar_ptr = make_ptr(Int64, 0, AddressSpace.smem, loc=loc, ip=ip)
    value = atom._unpack(loc=loc, ip=ip, tma_bar_ptr=dummy_tma_bar_ptr)
    return _cute_ir.prefetch(value, src.value, loc=loc, ip=ip)

####################################################################################################
#
# TensorSSA class (experimental)
#
####################################################################################################


class ReductionOp(Enum):
    ADD = auto()
    MUL = auto()
    MAX = auto()
    MIN = auto()
    INC = auto()
    DEC = auto()
    AND = auto()
    OR = auto()
    XOR = auto()

    def __str__(self):
        return self.name.lower()


class TensorSSA(cutlass_arith.ArithValue):
    """A class representing thread local data from CuTe Tensor in value semantic and immutable.

    :param value: Flatten vector as ir.Value holding logic data of SSA Tensor
    :type value: ir.Value
    :param shape: The nested shape in CuTe of the vector
    :type shape: Shape
    :param dtype: Data type of the tensor elements
    :type dtype: Type[Numeric]

    :ivar _shape: The nested shape in CuTe of the vector
    :ivar _dtype: Data type of the tensor elements

    :raises ValueError: If shape is not static
    """

    def __init__(self, value, shape: Shape, dtype: Type[Numeric]):
        """Initialize a new TensorSSA object.

        :param value: Flatten vector as ir.Value holding logic data of SSA Tensor
        :type value: ir.Value
        :param shape: The nested shape in CuTe of the vector
        :type shape: Shape
        :param dtype: Data type of the tensor elements
        :type dtype: Type[Numeric]
        :raises ValueError: If shape is not static
        """
        if not is_static(shape):
            raise ValueError("dynamic shape is not supported")

        signed = dtype.signed if issubclass(dtype, Integer) else False
        super().__init__(value, signed)

        self._shape = shape
        self._dtype = dtype
        self._layout = None

    @property
    def dtype(self) -> Type[Numeric]:
        return self._dtype

    @property
    def element_type(self) -> Type[Numeric]:
        return self._dtype

    @abstractmethod
    def __extract_mlir_values__(self):
        return [self]

    @abstractmethod
    def __new_from_mlir_values__(self, values):
        return TensorSSA(values[0], self.shape, self.dtype)

    def __str__(self):
        return f"tensor_value<{self.type} o {self.shape}>"

    @property
    def shape(self):
        return self._shape

    @overload
    def _apply_op(self, op, other: "TensorSSA", flip, *, loc, ip) -> "TensorSSA": ...

    @overload
    def _apply_op(
        self, op, other: cutlass_arith.ArithValue, flip, *, loc, ip
    ) -> "TensorSSA": ...

    @overload
    def _apply_op(
        self, op, other: Union[int, float, bool], flip, *, loc, ip
    ) -> "TensorSSA": ...

    def _apply_op(self, op, other, flip=False, *, loc=None, ip=None):
        def get_attr_for_type(ty, value):
            if isinstance(ty, ir.IntegerType):
                return ir.IntegerAttr.get(ty, value)
            elif isinstance(ty, ir.FloatType):
                return ir.FloatAttr.get(ty, value)
            else:
                raise TypeError(f"unsupported type: {ty}")

        # Canonicalize into Numeric
        if isinstance(other, (int, float, bool)) or (
            not isinstance(other, TensorSSA)
            and isinstance(other, cutlass_arith.ArithValue)
        ):
            other = as_numeric(other)

        # Promote types
        lhs, rhs, res_type = _binary_op_type_promote(self, other)

        # Promote scalar to vector
        if not isinstance(rhs, TensorSSA):
            if isinstance(rhs, Numeric):
                vect_val = vector.broadcast(lhs.type, rhs.ir_value(loc=loc, ip=ip))
            else:
                elem_attr = get_attr_for_type(lhs.type.element_type, rhs)
                vect_attr = ir.DenseElementsAttr.get_splat(lhs.type, elem_attr)
                vect_val = arith.constant(lhs.type, vect_attr, loc=loc, ip=ip)
            rhs = TensorSSA(vect_val, lhs.shape, lhs.dtype)

        if flip:
            lhs, rhs = rhs, lhs

        if op in (
            operator.lt,
            operator.le,
            operator.gt,
            operator.ge,
            operator.eq,
            operator.ne,
        ):
            res_type = Boolean

        assert isinstance(rhs, TensorSSA), f"rhs must be TensorSSA but got {rhs}"

        def _broadcast(s, t):
            if s == 1:
                return t
            elif t == 1:
                return s
            elif s == t:
                return s
            else:
                raise ValueError(f"cannot broadcast {s} and {t}")

        max_rank = max(rank(lhs.shape), rank(rhs.shape))
        lhs_shape = append(lhs.shape, 1, up_to_rank=max_rank)
        rhs_shape = append(rhs.shape, 1, up_to_rank=max_rank)
        res_shape = transform_leaf(_broadcast, lhs_shape, rhs_shape)

        # broadcast to the same shape
        lhs = lhs.broadcast_to(res_shape)
        rhs = rhs.broadcast_to(res_shape)

        if (
            op in (operator.add, operator.sub)
            and lhs.dtype == Boolean
            and rhs.dtype == Boolean
        ):
            res = op(lhs.to(Int32), rhs.to(Int32))
            zero = zeros_like(res)
            res = res.__ne__(zero).to(res_type)
        else:
            lhs_val = lhs.maybe_downcast()
            rhs_val = rhs.maybe_downcast()

            if issubclass(lhs.dtype, Integer):
                lhs_val = lhs_val.with_signedness(lhs.dtype.signed)

            if issubclass(rhs.dtype, Integer):
                rhs_val = rhs_val.with_signedness(rhs.dtype.signed)

            res_vect = op(lhs_val, rhs_val)
            res = TensorSSA(res_vect, lhs._shape, res_type)

        return res

    def broadcast_to(self, target_shape: Shape, *, loc=None, ip=None) -> "TensorSSA":
        """
        Broadcast the tensor to the target shape.
        """
        # pad source shape to the same rank
        shape = append(self.shape, 1, up_to_rank=rank(target_shape))
        if shape == target_shape:
            return self

        def _check_broadcast(s, t):
            if s != t and s != 1:
                raise ValueError(
                    f"src_shape and target_shape must be the same when src_shape is not 1, but got {s} and {t}"
                )

        transform_leaf(_check_broadcast, shape, target_shape)

        # reshape to flatten N-D vector
        flat_shp = flatten_to_tuple(shape)
        temp_ty = ir.VectorType.get(list(flat_shp), self.dtype.mlir_type)
        temp_vect = vector.shape_cast(temp_ty, self, loc=loc, ip=ip)

        # broadcast to result N-D vector
        flat_tgt_shp = flatten_to_tuple(target_shape)
        temp_tgt_ty = ir.VectorType.get(list(flat_tgt_shp), self.dtype.mlir_type)
        temp_tgt_vect = vector.broadcast(temp_tgt_ty, temp_vect, loc=loc, ip=ip)

        res_1d_ty = ir.VectorType.get([size(target_shape)], self.dtype.mlir_type)  # type: ignore
        res_1d_vect = vector.shape_cast(res_1d_ty, temp_tgt_vect, loc=loc, ip=ip)

        return TensorSSA(res_1d_vect, target_shape, self.dtype)

    def __pow__(self, other, *, loc=None, ip=None) -> "TensorSSA":
        """
        Returns the results of tensor^other.

        :param other: The other tensor for exponent.
        :type other: TensorSSA
        :return: The power of the tensor.
        :rtype: TensorSSA
        """
        return self._apply_op(operator.pow, other, loc=loc, ip=ip)

    def __rpow__(self, other, *, loc=None, ip=None) -> "TensorSSA":
        """
        Returns the results of other^tensor.

        :param other: The other tensor to compute power with.
        :type other: TensorSSA
        :return: The element-wise power of two tensors with same shape as inputs.
        :rtype: TensorSSA
        """
        return self._apply_op(operator.pow, other, flip=True, loc=loc, ip=ip)

    def __add__(self, other, *, loc=None, ip=None) -> "TensorSSA":
        """
        Returns the sum of the tensor and another tensor.

        :param other: The other tensor to add.
        :type other: TensorSSA
        :return: The sum of the two tensors with the same shape as inputs.
        :rtype: TensorSSA
        """
        return self._apply_op(operator.add, other, loc=loc, ip=ip)

    def __radd__(self, other, *, loc=None, ip=None) -> "TensorSSA":
        """
        Returns the sum of the tensor and another tensor (reverse add)

        :param other: The other tensor to add.
        :type other: TensorSSA
        :return: The sum of the two tensors with the same shape as inputs.
        :rtype: TensorSSA
        """
        return self._apply_op(operator.add, other, flip=True, loc=loc, ip=ip)

    def __sub__(self, other, *, loc=None, ip=None) -> "TensorSSA":
        """
        Returns the difference of the tensor and another tensor.

        :param other: The other tensor to subtract.
        :type other: TensorSSA
        :return: The subtraction of two tensors with same shape as inputs.
        :rtype: TensorSSA
        """
        return self._apply_op(operator.sub, other, loc=loc, ip=ip)

    def __rsub__(self, other, *, loc=None, ip=None) -> "TensorSSA":
        """
        Returns the difference of the tensor and another tensor (reverse subtract)

        :param other: The other tensor to subtract.
        :type other: TensorSSA
        :return: The subtraction of two tensors with same shape as inputs.
        :rtype: TensorSSA
        """
        return self._apply_op(operator.sub, other, flip=True, loc=loc, ip=ip)

    def __mul__(self, other, *, loc=None, ip=None) -> "TensorSSA":
        """
        Returns the multiplication of the tensor and another tensor.

        :param other: The other tensor to multiply.
        :type other: TensorSSA
        :return: The multiplication of two tensors with same shape as inputs.
        :rtype: TensorSSA
        """
        return self._apply_op(operator.mul, other, loc=loc, ip=ip)

    def __rmul__(self, other, *, loc=None, ip=None) -> "TensorSSA":
        """
        Returns the multiplication of the tensor and another tensor (reverse multiply)

        :param other: The other tensor to multiply.
        :type other: TensorSSA
        :return: The multiplication of two tensors with same shape as inputs.
        :rtype: TensorSSA
        """
        return self._apply_op(operator.mul, other, flip=True, loc=loc, ip=ip)

    def __mod__(self, other, *, loc=None, ip=None) -> "TensorSSA":
        """
        Returns the modulo of the tensor and another tensor.

        :param other: The other tensor to compute modulo with.
        :type other: TensorSSA
        :return: The element-wise modulo of two tensors with same shape as inputs.
        :rtype: TensorSSA
        """
        return self._apply_op(operator.mod, other, loc=loc, ip=ip)

    def __rmod__(self, other) -> "TensorSSA":
        """
        Returns the modulo of the tensor and another tensor (reverse modulo)

        :param other: The other tensor to compute modulo with.
        :type other: TensorSSA
        :return: The element-wise modulo of two tensors with same shape as inputs.
        :rtype: TensorSSA
        """
        return self._apply_op(operator.mod, other, flip=True)

    def __floordiv__(self, other, *, loc=None, ip=None) -> "TensorSSA":
        """
        Returns the floordiv(//) of the tensor and another tensor.

        :param other: The other tensor to compute floordiv with.
        :type other: TensorSSA
        :return: The floordiv of two tensors with same shape as inputs.
        :rtype: TensorSSA
        """
        return self._apply_op(operator.floordiv, other, loc=loc, ip=ip)

    def __rfloordiv__(self, other, *, loc=None, ip=None) -> "TensorSSA":
        """
        Returns the floordiv(//) of the tensor and another tensor (reverse floordiv)

        :param other: The other tensor to compute floordiv with.
        :type other: TensorSSA
        :return: The floordiv of two tensors with same shape as inputs.
        :rtype: TensorSSA
        """
        return self._apply_op(operator.floordiv, other, flip=True, loc=loc, ip=ip)

    def __truediv__(self, other, *, loc=None, ip=None) -> "TensorSSA":
        """
        Returns the truediv(/) of the tensor and another tensor.

        :param other: The other tensor to compute truediv with.
        :type other: TensorSSA
        :return: The truediv of two tensors with same shape as inputs.
        :rtype: TensorSSA
        """
        return self._apply_op(operator.truediv, other, loc=loc, ip=ip)

    def __rtruediv__(self, other, *, loc=None, ip=None) -> "TensorSSA":
        """
        Returns the truediv(/) of the tensor and another tensor (reverse truediv)

        :param other: The other tensor to compute truediv with.
        :type other: TensorSSA
        :return: The truediv of two tensors with same shape as inputs.
        :rtype: TensorSSA
        """
        return self._apply_op(operator.truediv, other, flip=True, loc=loc, ip=ip)

    def __eq__(self, other, *, loc=None, ip=None) -> "TensorSSA":
        """
        Returns the comparison of the tensor and another tensor as mask

        :param other: The other tensor to compare.
        :type other: TensorSSA
        :return: The comparison of two tensors with same shape as inputs.
        :rtype: TensorSSA
        """
        return self._apply_op(operator.eq, other, loc=loc, ip=ip)

    def __ne__(self, other, *, loc=None, ip=None) -> "TensorSSA":
        """
        Returns the element-wise not equal comparison of the tensor and another tensor.

        :param other: The other tensor to compare.
        :type other: TensorSSA
        :return: A boolean tensor with same shape as inputs, True where self != other.
        :rtype: TensorSSA
        """
        return self._apply_op(operator.ne, other, loc=loc, ip=ip)

    def __lt__(self, other, *, loc=None, ip=None) -> "TensorSSA":
        """
        Returns the element-wise less than comparison of the tensor and another tensor.

        :param other: The other tensor to compare with.
        :type other: TensorSSA
        :return: A boolean tensor with same shape as inputs, True where self < other.
        :rtype: TensorSSA
        """
        return self._apply_op(operator.lt, other, loc=loc, ip=ip)

    def __le__(self, other) -> "TensorSSA":
        """
        Returns the element-wise less than or equal comparison of the tensor and another tensor.

        :param other: The other tensor to compare with.
        :type other: TensorSSA
        :return: A boolean tensor with same shape as inputs, True where self <= other.
        :rtype: TensorSSA
        """
        return self._apply_op(operator.le, other)

    def __gt__(self, other, *, loc=None, ip=None) -> "TensorSSA":
        """
        Returns the element-wise greater than comparison of the tensor and another tensor.

        :param other: The other tensor to compare with.
        :type other: TensorSSA
        :return: A boolean tensor with same shape as inputs, True where self > other.
        :rtype: TensorSSA
        """
        return self._apply_op(operator.gt, other)

    def __ge__(self, other, *, loc=None, ip=None) -> "TensorSSA":
        """
        Returns the element-wise greater than or equal comparison of the tensor and another tensor.

        :param other: The other tensor to compare with.
        :type other: TensorSSA
        :return: A boolean tensor with same shape as inputs, True where self >= other.
        :rtype: TensorSSA
        """
        return self._apply_op(operator.ge, other, loc=loc, ip=ip)

    def __xor__(self, other, *, loc=None, ip=None) -> "TensorSSA":
        """
        Returns the element-wise XOR of the tensor and another tensor.

        :param other: The other tensor to perform XOR with.
        :type other: TensorSSA
        :return: The element-wise XOR of two tensors with same shape as inputs.
        :rtype: TensorSSA
        """
        return self._apply_op(operator.xor, other)

    def __rxor__(self, other, *, loc=None, ip=None) -> "TensorSSA":
        """
        Returns the bitwise XOR of the tensor and another tensor.

        :param other: The other tensor to compute XOR with.
        :type other: TensorSSA
        :return: The element-wise bitwise XOR of two tensors with same shape as inputs.
        :rtype: TensorSSA
        """
        return self._apply_op(operator.xor, other, flip=True, loc=loc, ip=ip)

    def __or__(self, other, *, loc=None, ip=None) -> "TensorSSA":
        """
        Returns the element-wise OR of the tensor and another tensor.

        :param other: The other tensor to perform OR with.
        :type other: TensorSSA
        :return: The element-wise OR of two tensors with same shape as inputs.
        :rtype: TensorSSA
        """
        return self._apply_op(operator.or_, other)

    def __ror__(self, other, *, loc=None, ip=None) -> "TensorSSA":
        """
        Returns the element-wise OR of the tensor and another tensor.

        :param other: The other tensor to perform OR with.
        :type other: TensorSSA
        :return: The element-wise OR of two tensors with same shape as inputs.
        :rtype: TensorSSA
        """
        return self._apply_op(operator.or_, other, flip=True)

    def __and__(self, other, *, loc=None, ip=None) -> "TensorSSA":
        """
        Returns the element-wise AND of the tensor and another tensor.

        :param other: The other tensor to perform AND with.
        :type other: TensorSSA
        :return: The element-wise AND of two tensors with same shape as inputs.
        :rtype: TensorSSA
        """
        return self._apply_op(operator.and_, other)

    def __rand__(self, other, *, loc=None, ip=None) -> "TensorSSA":
        """
        Returns the element-wise AND of the tensor and another tensor.

        :param other: The other tensor to perform AND with.
        :type other: TensorSSA
        :return: The element-wise AND of two tensors with same shape as inputs.
        :rtype: TensorSSA
        """
        return self._apply_op(operator.and_, other, flip=True, loc=loc, ip=ip)

    def __neg__(self, *, loc=None, ip=None) -> "TensorSSA":
        """
        Returns the negation of the tensor.

        :return: The element-wise negation of the tensor
        :rtype: TensorSSA
        """

        return self._apply_op(operator.sub, 0, flip=True, loc=loc, ip=ip)

    def _flatten_shape_and_coord(self, crd, *, loc=None, ip=None):
        # Coalesce and flatten source layout at terminal of coordinate
        # (N_0,(N_1,...), ...) -> (N_0,N_1,N_2,...)
        crd_shp = product_like(self._shape, target_profile=crd, loc=loc, ip=ip)

        # Flatten coordinate
        flat_shp = flatten(crd_shp)
        assert isinstance(flat_shp, tuple) and is_static(flat_shp)
        # (C_0,(C_1,...), ...) -> (C_0,C_1,C_2,...)
        flat_crd = flatten(crd)

        assert isinstance(flat_crd, tuple) and is_static(flat_crd)
        return flat_shp, flat_crd

    def _build_result(self, res_vect, res_shp, *, loc=None, ip=None):
        if isinstance(res_shp, ir.Value):
            raise ValueError(
                f"expects static shape and coordinates, but got {self._shape} and {crd}"
            )

        # cast back to 1D vector
        res_1d_ty = ir.VectorType.get([size(res_shp)], self.type.element_type)
        res_1d_vect = vector.shape_cast(res_1d_ty, res_vect, loc=loc, ip=ip)
        return TensorSSA(res_1d_vect, res_shp, self.dtype)

    @dsl_user_op
    def __getitem__(
        self, crd: Coord, *, loc=None, ip=None
    ) -> Union["TensorSSA", Numeric]:
        """Access or slice tensor elements using coordinates.

        This method implements tensor evaluation T(c) = *(E + L(c)) where E is the iterator/engine
        and L is the layout. It supports both direct element access and slicing operations.

        :param crd: Coordinate or slice specification for accessing tensor elements
        :type crd: Coord
        :param loc: Source location for MLIR operation tracking, defaults to None
        :type loc: Optional[Location]
        :param ip: Insertion point for MLIR operation, defaults to None
        :type ip: Optional[InsertionPoint]
        :return: Tensor element value or sliced subtensor
        :rtype: Union[TensorSSA, Numeric]

        :raises ValueError: If coordinate access is invalid for the tensor layout

        **Examples:**

        .. code-block:: python

            # Create a fragment from rmem as shape (8, 4)
            layout = make_layout((8, 4))
            tensor = make_fragment(layout, Float32)
            frg = tensor.load()

            # Direct element access
            val = frg[0]  # Returns first element of fragment
            val = frg[(0, 1)]  # Returns element at (0, 1)

            # Slice access
            sliced = frg[(3, None)]  # Returns fragment slice
        """
        # short-cut to no-op
        if crd is None:
            return self

        if not has_underscore(crd):
            if self._layout is None:
                self._layout = make_layout(self._shape, loc=loc, ip=ip)
            idx = crd2idx(crd, self._layout, loc=loc, ip=ip)
            idx_val = as_numeric(idx).ir_value(loc=loc, ip=ip)
            res_val = vector.extractelement(self, position=idx_val, loc=loc, ip=ip)
            return self.dtype(res_val)

        if not is_static(crd):
            raise ValueError("dynamic coordinate is not supported")

        flat_shp, flat_crd = self._flatten_shape_and_coord(crd)

        multi_dim_ty = ir.VectorType.get(list(flat_shp), self.type.element_type)
        # vector<NxTy> -> vector<N_0xN_1x...xTy>
        tmp_vect = vector.shape_cast(multi_dim_ty, self)

        # Slice and keep dims matching `_` or None
        res_shp = slice_(self._shape, crd)
        if isinstance(res_shp, ir.Value):
            raise TypeError(
                f"expects static shape and coordinates, but got {self._shape} and {crd}"
            )

        # Offsets is index of coordinates if NOT `_` otherwise 0
        offsets = [c if c is not None else 0 for c in flat_crd]
        # Sizes is size of shapes if `_` otherwise 1
        sizes = [s if c is None else 1 for s, c in zip(flat_shp, flat_crd)]
        # Logic stride to index vector. Only support stride-1 by vector
        strides = [1] * rank(flat_shp)

        # Vector slice on N-D vector
        res_ty = ir.VectorType.get(list(sizes), self.type.element_type)
        res_vect = vector.extract_strided_slice(
            res_ty, tmp_vect, offsets=offsets, sizes=sizes, strides=strides
        )

        # Slice and keep dims matching `_` or None
        res_shp = slice_(self._shape, crd)
        return self._build_result(res_vect, res_shp, loc=loc, ip=ip)

    @dsl_user_op
    def to(self, dtype: Type[Numeric], *, loc=None, ip=None):
        """Convert the tensor to a different numeric type.

        :param dtype: The target numeric type to cast to.
        :type dtype: Type[Numeric]
        :return: A new tensor with the same shape but with elements cast to the target type.
        :rtype: TensorSSA
        :raises TypeError: If dtype is not a subclass of Numeric.
        :raises NotImplementedError: If dtype is an unsigned integer type.
        """
        if dtype is ir.Value:
            return self

        if not isclass(dtype) or not issubclass(dtype, Numeric):
            raise TypeError(f"dtype must be a type of Numeric, but got {type(dtype)}")

        src_dtype = self.dtype
        if src_dtype == dtype:
            return self

        # maybe downcast can lose signedness
        src = self.maybe_downcast().with_signedness(self.signed)
        if src_dtype.is_float and dtype.is_float:
            res_vect = cutlass_arith.cvtf(src, dtype.mlir_type, loc=loc, ip=ip)
        elif src_dtype.is_float and issubclass(dtype, Integer):
            res_vect = cutlass_arith.fptoi(
                src, dtype.signed, dtype.mlir_type, loc=loc, ip=ip
            )
        elif issubclass(src_dtype, Integer) and dtype.is_float:
            res_vect = cutlass_arith.itofp(
                src, src_dtype.signed, dtype.mlir_type, loc=loc, ip=ip
            )
        else:
            res_vect = cutlass_arith.int_to_int(src, dtype, loc=loc, ip=ip)

        return TensorSSA(res_vect, self._shape, dtype)

    def ir_value(self, *, loc=None, ip=None):
        return self

    def ir_value_int8(self, *, loc=None, ip=None):
        """
        Returns int8 ir value of Boolean tensor.
        When we need to store Boolean tensor ssa, use ir_value_int8().

        :param loc: Source location information, defaults to None
        :type loc: Optional[Location], optional
        :param ip: Insertion point for MLIR operations, defaults to None
        :type ip: Optional[InsertionPoint], optional
        :return: The int8 value of this Boolean
        :rtype: ir.Value
        """
        assert (
            self.element_type is Boolean
        ), f"Only boolean type needs to be converted to int8, got {self.element_type}"

        if not hasattr(self, "_value_int8"):
            self._value_int8 = arith.extsi(
                T.vector(self.type.shape[0], T.i8()), self, loc=loc, ip=ip
            )
        return self._value_int8

    def reduce(self, op, init_val, reduction_profile: Coord, *, loc=None, ip=None):
        """
        Perform reduce on selected modes with given predefined reduction op.

        :param op: The reduction operator to use (operator.add or operator.mul)
        :type op: operator
        :param init_val: The initial value for the reduction
        :type init_val: numeric
        :param reduction_profile: Specifies which dimensions to reduce. Dimensions marked with `None` are kept.
        :type reduction_profile: Coord

        :return: The reduced tensor
        :rtype: TensorSSA

        **Examples:**

        .. code-block:: python

            reduce(f32 o (4,))
              => f32

            reduce(f32 o (4, 5))
              => f32
            reduce(f32 o (4, (5, 4)), reduction_profile=(None, 1))
              => f32 o (4,)
            reduce(f32 o (4, (5, 4)), reduction_profile=(None, (None, 1)))
              => f32 o (4, (5,))
        """
        # short-cut to no-op
        if reduction_profile is None:
            return self

        if not is_weakly_congruent(reduction_profile, self.shape):
            raise ValueError(
                f"Expect reduction_profile be weakly congruent to the shape of the tensor, "
                f"but got {reduction_profile} and {self.shape}"
            )

        if op is ReductionOp.ADD:
            red_kind = vector.CombiningKind.ADD
        elif op is ReductionOp.MUL:
            red_kind = vector.CombiningKind.MUL
        elif op is ReductionOp.MAX:
            red_kind = vector.CombiningKind.MAXIMUMF
        elif op is ReductionOp.MIN:
            red_kind = vector.CombiningKind.MINIMUMF
        else:
            raise NotImplementedError(
                f"{op} is not supported, expects one of "
                f"{ReductionOp.ADD, ReductionOp.MUL, ReductionOp.MAX, ReductionOp.MIN}"
            )

        elem_ty = self.element_type
        # Canonicalize to `Numeric` and convert into MLIR value
        init_val = as_numeric(init_val).ir_value(loc=loc, ip=ip)

        if depth(reduction_profile) == 0:
            return vector.reduction(
                elem_ty.mlir_type, red_kind, self, acc=init_val, loc=loc, ip=ip
            )

        flat_shp, flat_prof = self._flatten_shape_and_coord(
            reduction_profile, loc=loc, ip=ip
        )
        assert depth(flat_shp) == 1 and depth(flat_prof) == 1
        assert rank(flat_shp) == rank(flat_prof)

        temp_ty = ir.VectorType.get(list(flat_shp), elem_ty.mlir_type)
        temp_vect = vector.shape_cast(temp_ty, self, loc=loc, ip=ip)

        if isinstance(flat_prof, tuple):
            red_dims = [i for i, x in enumerate(flat_prof) if x is not None]
        else:
            red_dims = [0]

        temp_acc_shp = slice_(flat_shp, flat_prof, loc=loc, ip=ip)
        temp_acc_ty = ir.VectorType.get(list(temp_acc_shp), elem_ty.mlir_type)

        init_val = vector.broadcast(temp_acc_ty, init_val, loc=loc, ip=ip)
        res_vect = vector.multi_reduction(
            red_kind, temp_vect, acc=init_val, reduction_dims=red_dims, loc=loc, ip=ip
        )

        # Slice and keep dims matching `_` or None
        res_shp = slice_(self.shape, reduction_profile, loc=loc, ip=ip)
        return self._build_result(res_vect, res_shp, loc=loc, ip=ip)


@dsl_user_op
def full(shape, fill_value, dtype: Type[Numeric], *, loc=None, ip=None) -> TensorSSA:
    """
    Return a new TensorSSA of given shape and type, filled with fill_value.

    :param shape: Shape of the new tensor.
    :type shape: tuple
    :param fill_value: Value to fill the tensor with.
    :type fill_value: scalar
    :param dtype: Data type of the tensor.
    :type dtype: Type[Numeric]
    :return: Tensor of fill_value with the specified shape and dtype.
    :rtype: TensorSSA
    """
    size = product(shape, loc=loc, ip=ip)
    if not is_static(size):
        raise ValueError("shape must be static")

    if isinstance(fill_value, (ir.Value, int, float, bool)):
        fill_value = dtype(fill_value)
    elif isinstance(fill_value, Numeric):
        fill_value = fill_value.to(dtype, loc=loc, ip=ip)
    else:
        raise ValueError(f"Expected fill_value be numeric type, but got {fill_value}")

    res_ty = T.vector(size, dtype.mlir_type)
    res_val = vector.splat(res_ty, fill_value.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    return TensorSSA(res_val, shape, dtype)


def full_like(
    a: Union[TensorSSA, Tensor],
    fill_value,
    dtype: Union[None, Type[Numeric]] = None,
    *,
    loc=None,
    ip=None,
) -> TensorSSA:
    """
    Return a full TensorSSA with the same shape and type as a given array.

    :param a: The shape and data-type of `a` define these same attributes of the returned array.
    :type a: array_like
    :param fill_value: Fill value.
    :type fill_value: array_like
    :param dtype: Overrides the data type of the result, defaults to None
    :type dtype: Union[None, Type[Numeric]], optional
    :return: Tensor of `fill_value` with the same shape and type as `a`.
    :rtype: TensorSSA

    .. seealso::
       :func:`empty_like`: Return an empty array with shape and type of input.
       :func:`ones_like`: Return an array of ones with shape and type of input.
       :func:`zeros_like`: Return an array of zeros with shape and type of input.
       :func:`full`: Return a new array of given shape filled with value.

    **Examples:**

    .. code-block:: python

        frg = cute.make_fragment(Float32, (2, 3))
        a = frg.load()
        b = cute.full_like(a, 1.0)
    """
    if not hasattr(a, "shape"):
        raise TypeError(f"Expect `a` be shaped type, but got {type(a)}")

    return full(
        a.shape, fill_value, dtype if dtype is not None else a.dtype, loc=loc, ip=ip
    )


def empty_like(a, dtype=None):
    """
    Return a new TensorSSA with the same shape and type as a given array, without initializing entries.

    :param a: The shape and data-type of `a` define these same attributes of the returned array.
    :type a: TensorSSA
    :param dtype: Overrides the data type of the result, defaults to None
    :type dtype: Type[Numeric], optional
    :return: Uninitialized tensor with the same shape and type (unless overridden) as `a`.
    :rtype: TensorSSA
    """
    return full_like(a, 0, dtype)


def ones_like(a, dtype=None):
    """
    Return a TensorSSA of ones with the same shape and type as a given array.

    :param a: The shape and data-type of `a` define these same attributes of the returned array.
    :type a: TensorSSA
    :param dtype: Overrides the data type of the result, defaults to None
    :type dtype: Type[Numeric], optional
    :return: Tensor of ones with the same shape and type (unless overridden) as `a`.
    :rtype: TensorSSA
    """
    return full_like(a, 1, dtype)


def zeros_like(a, dtype=None, *, loc=None, ip=None):
    """
    Return a TensorSSA of zeros with the same shape and type as a given array.

    :param a: The shape and data-type of `a` define these same attributes of the returned array.
    :type a: TensorSSA
    :param dtype: Overrides the data type of the result, defaults to None
    :type dtype: Type[Numeric], optional
    :return: Tensor of zeros with the same shape and type (unless overridden) as `a`.
    :rtype: TensorSSA
    """
    return full_like(a, 0, dtype, loc=loc, ip=ip)


def where(
    cond: TensorSSA, x: TensorSSA, y: TensorSSA, *, loc=None, ip=None
) -> TensorSSA:
    """
    Return elements chosen from x or y depending on condition.

    :param cond: Where True, yield x, where False, yield y.
    :type cond: TensorSSA
    :param x: Values from which to choose when condition is True.
    :type x: TensorSSA
    :param y: Values from which to choose when condition is False.
    :type y: TensorSSA
    :return: A tensor with elements from x where condition is True, and elements from y where condition is False.
    :rtype: TensorSSA
    """
    if x.dtype != y.dtype:
        raise ValueError(
            f"x and y must have the same dtype, but got {x.dtype} and {y.dtype}"
        )

    if cond.dtype != Boolean:
        raise ValueError(f"cond must be Boolean type, but got {cond.dtype}")

    return TensorSSA(
        arith.select(cond.ir_value(), x, y, loc=loc, ip=ip), x.shape, x.dtype
    )


def any_(x: TensorSSA, *, loc=None, ip=None) -> Boolean:
    """
    Test whether any tensor element evaluates to True.

    :param x: Input tensor.
    :type x: TensorSSA
    :return: Returns a TensorSSA scalar containing True if any element of x is True, False otherwise.
    :rtype: TensorSSA
    """
    is_true = x != full_like(x, 0, x.dtype, loc=loc, ip=ip)
    return Boolean(
        vector.reduction(T.bool(), vector.CombiningKind.OR, is_true, loc=loc, ip=ip)
    )


def all_(x: TensorSSA, *, loc=None, ip=None) -> Boolean:
    """
    Test whether all tensor elements evaluate to True.

    :param x: Input tensor.
    :type x: TensorSSA
    :return: Returns a TensorSSA scalar containing True if all elements of x are True, False otherwise.
    :rtype: TensorSSA
    """
    is_true = x != full_like(x, 0, x.dtype, loc=loc, ip=ip)
    return Boolean(
        vector.reduction(T.bool(), vector.CombiningKind.AND, is_true, loc=loc, ip=ip)
    )


##############################################################################
# User defined struct
##############################################################################


class struct:
    """
    Decorator to abstract C structure in Python DSL.

    **Usage:**

    .. code-block:: python

        # Supports base_dsl scalar int/float elements, array and nested struct:
        @cute.struct
        class complex:
            real : cutlass.Float32
            imag : cutlass.Float32


        @cute.struct
        class StorageA:
            mbarA : cute.struct.MemRange[cutlass.Int64, stage]
            compA : complex
            intA : cutlass.Int16


        # Supports aligment for its elements:
        @cute.struct
        class StorageB:
            a: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, size_a], 1024
            ]
            b: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, size_b], 1024
            ]
            x: cute.struct.Align[cutlass.Int32, 16]
            compA: cute.struct.Align[complex, 16]


        # Statically get size and alignment:
        size = StorageB.__sizeof__()
        align = StorageB.__alignof__()

        # Allocate and referencing elements:
        storage = allocator.allocate(StorageB)

        storage.a[0] ...
        storage.x ...
        storage.compA.real ...

    :param cls: The struct class with annotations.
    :return: The decorated struct class.
    """

    # inner class for defining a continuous memory region
    class _MemRangeMeta(type):
        """
        A metaclass for creating MemRange classes.

        This metaclass is used to dynamically create MemRange classes with specific
        data types and sizes.

        :ivar _dtype: The data type of the MemRange.
        :ivar _size: The size of the MemRange.
        """

        _dtype = None
        _size = None

        def __new__(cls, name, bases, dct):
            new_cls = super().__new__(cls, name, bases, dct)
            return new_cls

        def __getitem__(cls, params) -> Type["struct.MemRange"]:
            # get params from syntax: struct.MemRange[dtype, size]
            if len(params) == 2:
                dtype, size = params
            else:
                raise TypeError("Invalid struct.MemRange Arguments")

            if not struct._is_scalar_type(dtype):
                raise TypeError("MemRange only support dsl scalar type!")

            # Create new class with proper name and parameters
            new_cls = type(
                f"struct.MemRange[{dtype.__name__}, {size}]",
                (struct.MemRange,),
                {"_dtype": dtype, "_size": size},
            )
            return new_cls

        @property
        def size(cls):
            return cls._size

        @property
        def elem_width(cls):
            return cls._dtype.width

        @property
        def size_in_bytes(cls):
            return cls.size * cls.elem_width // 8

    class MemRange(metaclass=_MemRangeMeta):
        """
        Defines a range of memory by `MemRange[T, size]`.
        """

        pass

    class _MemRangeData:
        """
        Represents a range of memory.

        :param dtype: The data type.
        :param size: The size of the memory range in bytes.
        :param base: The base address of the memory range.
        """

        def __init__(self, dtype, size, base):
            """
            Initializes a new memory range.

            :param dtype: The data type.
            :param size: Size of the memory range in bytes. A size of **0** is accepted, but in that
                         case the range can only be used for its address (e.g. as a partition marker).
            :param base: The base address of the memory range.
            """
            self._dtype = dtype
            self._size = size
            self._base = base

        def data_ptr(self):
            """
            Returns start pointer to the data in this memory range.

            :return: A pointer to the start of the memory range.
            :raises AssertionError: If the size of the memory range is negative.
            """
            assert self._size >= 0
            return recast_ptr(self._base, dtype=self._dtype)

        def get_tensor(self, layout, swizzle=None, dtype=None):
            """
            Creates a tensor from the memory range.

            :param layout: The layout of the tensor.
            :param swizzle: Optional swizzle pattern.
            :param dtype: Optional data type; defaults to the memory range's data type if not specified.
            :return: A tensor representing the memory range.
            :raises TypeError: If the layout is incompatible with the swizzle.
            :raises AssertionError: If the size of the memory range is not greater than zero.
            """
            assert self._size > 0
            # make tensor
            if isinstance(layout, ComposedLayout) and (swizzle is not None):
                raise TypeError(f"incompatible layout with swizzle")
            elem_type = self._dtype if dtype is None else dtype
            ptr = recast_ptr(self._base, swizzle, dtype=elem_type)
            res = make_tensor(ptr, layout)
            return res

        def __getitem__(self, index: int) -> Any:
            """
            Returns the element at the specified index in the memory range.

            :param index: The index of the element to retrieve.
            :return: The element at the specified index.
            :raises AssertionError: If the index is out of range.
            """
            assert (index >= 0) and (index < self._size)
            return self.data_ptr() + index

    # inner class for aligning a member type
    class _AlignMeta(type):
        """
        Aligns the given object by setting its alignment attribute.

        :param v: The object to align. Must be a struct, MemRange, or a scalar type.
        :param align: The alignment value to set.
        :raises TypeError: If the object is not a struct, MemRange, or a scalar type.

        :ivar _dtype: The data type to be aligned.
        :ivar _align: The alignment of the data type.
        """

        _dtype = None
        _align = None

        def __new__(cls, name, bases, dct):
            return super().__new__(cls, name, bases, dct)

        def __getitem__(cls, params) -> Any:
            if len(params) == 2:
                dtype, align = params
                assert align > 0
            else:
                raise TypeError("Invalid struct.Align Arguments")

            if not struct._is_scalar_type(dtype) and not isinstance(
                dtype, (struct, struct._MemRangeMeta)
            ):
                raise TypeError(
                    "align only can be applied to struct/MemRange/base_dsl scalar"
                )

            # Create new class with alignment
            new_cls = type(
                f"struct.Align[{dtype.__name__}, {align}]",
                (struct.Align,),
                {"_dtype": dtype, "_align": align},
            )
            return new_cls

        @property
        def dtype(cls):
            return cls._dtype

        @property
        def align(cls):
            return cls._align

    class Align(metaclass=_AlignMeta):
        """
        Aligns the given type by `Align[T, alignment]`.
        """

        pass

    # util func for base dsl scalar types
    @staticmethod
    def _is_scalar_type(dtype):
        """
        Checks if the given type is a scalar numeric type.

        :param dtype: The type to check.
        :return: True if the type is a subclass of Numeric, False otherwise.
        """
        return isinstance(dtype, type) and issubclass(dtype, Numeric)

    # calculate size and alignment
    def __init__(self, cls):
        """
        Initializes a new struct decorator instance.

        :param cls: The class representing the structured data type.
        :raises TypeError: If the struct is empty.
        """
        self._cls = cls
        self.__name__ = f"struct::{cls.__name__}"
        # Get the class annotations
        self._annotations = cls.__annotations__
        # Create a dictionary to store the offsets
        self._offsets: Dict[str, int] = {}

        # Calculate the offsets and alignment
        offset = 0
        alignment = 1
        if len(self._annotations) == 0:
            raise TypeError("Empty struct is not supported!")
        for name, object in self._annotations.items():
            # get alignment of object
            sub_align = 1
            if isinstance(object, struct._AlignMeta):
                sub_align = object.align
                object = object.dtype

            # switch addition order to support dynamic size
            def add_offset(val):
                return val + offset if isinstance(val, ir.Value) else offset + val

            # size of scalar
            if struct._is_scalar_type(object):
                dtype_size = max(1, object.width // 8)
                sub_align = max(dtype_size, sub_align)
                offset = self.align_offset(offset, sub_align)
                self._offsets[name] = offset
                offset = add_offset(dtype_size)
            # size of array is size_in_bytes, alignment is elem_size
            elif isinstance(object, struct._MemRangeMeta):
                # Allow empty array as a free marker-only struct member.
                # Use max(sub_align, ) because we might have in the future some
                # object.elem_width less than 8, such as fp4, bit and others,
                # and align_offset() does not support an alignment of 0.
                sub_align = max(object.elem_width // 8, sub_align)
                offset = self.align_offset(offset, sub_align)
                self._offsets[name] = offset
                offset = add_offset(object.size_in_bytes)
            # size of struct
            elif isinstance(object, struct):
                sub_align = max(object.__alignof__(), sub_align)
                offset = self.align_offset(offset, sub_align)
                self._offsets[name] = offset
                offset = add_offset(object.__sizeof__())
            else:
                raise TypeError(
                    f"Struct element only support struct/array/base_dsl scalar, "
                    f"but got {object}"
                )
            # Total aligment determined by the strictest requirement
            alignment = max(alignment, sub_align)
        # Total size determined by alignment
        self._align_of = alignment
        self._size_of = self.align_offset(offset, alignment)

    # create the __init__ method for decorated struct
    def __call__(self, base: Any) -> None:
        """
        Creates a new instance of the decorated struct.

        :param base: The base address of the struct.
        :return: An instance of the decorated struct.
        :raises TypeError: If the base pointer is not byte-sized.
        """
        if base.type.value_type.width != 8:
            raise TypeError("struct base ptr value type must be byte sized.")
        # make an new object of user-defined decorated struct
        # otherwise it will override same self._cls when new instance created
        cls = self._cls()
        setattr(cls, "_base", base)
        for name, off in self._offsets.items():
            obj = self._annotations[name]
            if isinstance(obj, struct._AlignMeta):
                obj = obj.dtype
            if struct._is_scalar_type(obj):
                new_obj = recast_ptr(base + off, dtype=obj)
                setattr(cls, name, new_obj)
            elif isinstance(obj, struct._MemRangeMeta):
                new_obj = struct._MemRangeData(obj._dtype, obj._size, base + off)
                setattr(cls, name, new_obj)
            elif isinstance(obj, struct):
                new_obj = obj(base + off)
                setattr(cls, name, new_obj)
            else:
                raise TypeError(
                    f"Struct element only support struct/array/base_dsl scalar, "
                    f"but got {obj}"
                )
        return cls

    # get size
    def size_in_bytes(self) -> int:
        """
        Returns the size of the struct in bytes.

        :return: The size of the struct.
        """
        return self._size_of

    # get size
    def __sizeof__(self) -> int:
        return self._size_of

    # get alignment
    def __alignof__(self) -> int:
        return self._align_of

    # util func for aligning offset
    @staticmethod
    def align_offset(offset, align):
        """
        Return the round-up offset up to the next multiple of align.
        """
        assert align > 0 and not (
            align & (align - 1)
        ), "align should be a strictly positive power of 2."
        return (offset + (align - 1)) & ~(align - 1)
