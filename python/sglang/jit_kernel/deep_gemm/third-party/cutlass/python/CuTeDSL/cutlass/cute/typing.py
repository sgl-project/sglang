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

from abc import ABC, abstractmethod
from typing import ForwardRef, Tuple, Union, Any, Type, List

from cutlass.base_dsl.typing import *

from cutlass._mlir import ir
import cutlass._mlir.extras.types as T
from cutlass._mlir.dialects.cute import AddressSpace


Int = Union[int, Integer]


ScaledBasis = ForwardRef("ScaledBasis")


IntTuple = Union[Int, Tuple["IntTuple", ...]]
Shape = Union[Int, Tuple["Shape", ...]]
Stride = Union[Int, ScaledBasis, Tuple["Stride", ...]]
Coord = Union[Int, None, Tuple["Coord", ...]]


class Layout(ir.Value):
    def __init__(self, op_result):
        super().__init__(op_result)

    def __str__(self): ...

    def get_hier_coord(self, idx) -> Coord:
        """Return the (hierarchical) ND logical coordinate corresponding to the linear index"""
        ...

    @property
    def shape(self, *, loc=None, ip=None) -> Shape: ...

    @property
    def stride(self, *, loc=None, ip=None) -> Stride: ...


Tile = Union[Int, None, Layout, Tuple["Tile", ...]]

# XTuple is super set of above types
XTuple = Union[IntTuple, Shape, Stride, Coord, Tile]

Tiler = Union[Shape, Layout, Tile]


class Pointer(ABC):
    """
    Abstract base class for CuTe jit function and runtime _Pointer
    """

    @property
    def value_type(self) -> Type[Numeric]:
        return self.dtype

    @property
    def dtype(self) -> Type[Numeric]: ...

    def align(self, min_align: int) -> "Pointer": ...

    def __get_mlir_types__(self) -> List[ir.Type]: ...

    def __extract_mlir_values__(self) -> List[ir.Value]: ...

    def __new_from_mlir_values__(self, values) -> "Pointer": ...


class Tensor(ABC):
    """
    Abstract base class for CuTe jit function and runtime _Tensor

    A CuTe Tensor is iterator with layout

    :Examples:

    Create tensor from torch.tensor with Host Runtime:

    .. code-block:: python

        >>> import torch
        >>> from cutlass.cute.runtime import from_dlpack
        >>> mA = from_dlpack(torch.tensor([1, 3, 5], dtype=torch.int32))
        >>> mA.shape
        (3,)
        >>> mA.stride
        (1,)
        >>> mA.layout
        (3,):(1,)

    Define JIT function:

    .. code-block:: python

        @cute.jit
        def add(a: Tensor, b: Tensor, res: Tensor): ...

    Call JIT function from python:

    .. code-block:: python

        >>> import torch
        >>> a = torch.tensor([1, 3, 5], dtype=torch.int32)
        >>> b = torch.tensor([2, 4, 6], dtype=torch.int32)
        >>> c = torch.zeros([3], dtype=torch.int32)
        >>> mA = from_dlpack(a)
        >>> mB = from_dlpack(b)
        >>> mC = from_dlpack(c)
        >>> add(mA, mB, mC)
        >>> c
        tensor([3, 7, 11], dtype=torch.int32)
    """

    def __str__(self): ...

    @abstractmethod
    def __getitem__(self, idx) -> Union["Tensor", ir.Value, IntTuple]: ...

    @abstractmethod
    def __setitem__(self, idx, value): ...

    @property
    @abstractmethod
    def element_type(self) -> Union[Type[Numeric], Type[IntTuple]]: ...

    @element_type.setter
    def element_type(self, new_type): ...

    @property
    @abstractmethod
    def memspace(self) -> AddressSpace: ...

    @property
    @abstractmethod
    def iterator(self): ...

    @property
    def layout(self) -> Union[Layout, "ComposedLayout"]: ...

    @property
    def shape(self) -> Shape: ...

    def load(self, *, loc=None, ip=None) -> "TensorSSA": ...

    def store(self, data: "TensorSSA", *, loc=None, ip=None): ...

    def mark_layout_dynamic(self, leading_dim: int | None = None) -> "Tensor": ...

    def mark_compact_shape_dynamic(
        self,
        mode: int,
        stride_order: tuple[int, ...] | None = None,
        divisibility: int = 1,
    ) -> "Tensor": ...

    @abstractmethod
    def fill(self, value: Numeric) -> None: ...


__all__ = [
    "Coord",
    "Numeric",
    "Integer",
    "Boolean",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "Uint8",
    "Uint16",
    "Uint32",
    "Uint64",
    "Float",
    "Float16",
    "BFloat16",
    "TFloat32",
    "Float32",
    "Float64",
    "Float8E5M2",
    "Float8E4M3FN",
    "Float8E4M3B11FNUZ",
    "Float8E4M3",
    "Float8E8M0FNU",
    "Float4E2M1FN",
    "Float6E2M3FN",
    "Float6E3M2FN",
    "IntTuple",
    "Layout",
    "Pointer",
    "Shape",
    "Stride",
    "Tensor",
    "Tile",
    "Tiler",
    "XTuple",
]
