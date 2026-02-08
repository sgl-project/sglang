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

import ctypes
from functools import lru_cache
import itertools
import operator
from time import time
from typing import Union

# MLIR modules imports
from cutlass._mlir import ir
import cutlass._mlir.dialects.cute as _cute_ir

from cutlass.base_dsl.dsl import is_dynamic_expression
from cutlass.cutlass_dsl import JitArgAdapterRegistry

# Local modules imports
from .typing import (
    AddressSpace,
    Tensor,
    Type,
    Pointer,
    Boolean,
    Numeric,
    Float4E2M1FN,
    Int64,
    Int32,
    Int16,
    Int8,
    Uint64,
    Uint32,
    Uint16,
    Uint8,
    Float64,
    Float32,
    Float16,
    BFloat16,
    Float8E5M2,
)
from . import core
from .core import _Tensor as CoreTensor


class _Pointer(Pointer):
    """Runtime representation of a pointer that can inter-operate with various data structures,
    including numpy arrays and device memory.

    :param pointer: The pointer to the data
    :type pointer: int or pointer-like object
    :param dtype: Data type of the elements pointed to
    :type dtype: Type
    :param mem_space: Memory space where the pointer resides, defaults to generic
    :type mem_space: _cute_ir.AddressSpace, optional
    :param assumed_align: Assumed alignment of input pointer in bytes, defaults to None
    :type assumed_align: int, optional

    :ivar _pointer: The underlying pointer
    :ivar _dtype: Data type of the elements
    :ivar _addr_space: Memory space of the pointer
    :ivar _assumed_align: Alignment of the pointer in bytes
    :ivar _desc: C-type descriptor for the pointer
    :ivar _c_pointer: C-compatible pointer representation
    """

    def __init__(
        self,
        pointer,
        dtype,
        mem_space: _cute_ir.AddressSpace = _cute_ir.AddressSpace.generic,
        assumed_align=None,
    ):
        self._pointer = pointer
        self._dtype = dtype
        self._addr_space = mem_space

        if assumed_align is None:
            self._assumed_align = dtype.width // 8
        else:
            self._assumed_align = assumed_align

        self._c_pointer = None
        assert (
            int(self._pointer) % self._assumed_align == 0
        ), f"pointer must be {self._assumed_align} bytes aligned"

    def size_in_bytes(self) -> int:
        self._desc = ctypes.c_void_p(int(self._pointer))
        return ctypes.sizeof(self._desc)

    def __get_mlir_types__(self):
        return [self.mlir_type]

    def __c_pointers__(self):
        if self._c_pointer is None:
            self._desc = ctypes.c_void_p(int(self._pointer))
            self._c_pointer = ctypes.addressof(self._desc)
        return [self._c_pointer]

    def __new_from_mlir_values__(self, values):
        assert len(values) == 1
        return values[0]

    def __extract_mlir_values__(self):
        return [self._c_pointer]

    # Move mlir Type out of __init__ to decouple with mlir Context
    @property
    def mlir_type(self) -> ir.Type:
        return _cute_ir.PtrType.get(
            self._dtype.mlir_type, self._addr_space, self._assumed_align
        )

    @property
    def dtype(self) -> Type[Numeric]:
        return self._dtype

    @property
    def memspace(self):
        return self._addr_space

    def align(self, min_align: int, *, loc=None, ip=None) -> Pointer:
        raise NotImplementedError("align is not supported in runtime")

    def verify(self, expected_py_type):
        if expected_py_type is Pointer:
            return True
        elif isinstance(expected_py_type, ir.Value) and expected_py_type.ty is Pointer:
            return True

        return False

    def __str__(self) -> str:
        return f"Ptr<0x{int(self._pointer):016x}@{self._addr_space}>"

    def __repr__(self):
        return self.__str__()


class _Tensor(Tensor):
    def __init__(
        self,
        tensor,
        assumed_align=None,
    ):
        # If tensor is already a DLPack object, use it directly
        if hasattr(tensor, "__dlpack_device__") and not hasattr(tensor, "__dlpack__"):
            self._dlpack_data = tensor
        else:
            self._dlpack_data = tensor.__dlpack__()
        self._dltensor_wrapper = None
        self._assumed_align = assumed_align
        self._is_dynamic = False
        self._memref_desc = None
        self._dtype = None

    @property
    def __class__(self) -> Type[Tensor]:
        # Cheat to let `type(_Tensor())` to return cute.Tensor
        return Tensor

    @staticmethod
    def lazily_load_dltensor(func):
        """Decorator to lazily load the DLTensorWrapper.

        This decorator loads the DLTensorWrapper when needed,
        avoiding overhead in the critical path of calling JIT functions.
        """

        def wrapper(self, *args, **kwargs):
            if self._dltensor_wrapper is None:
                self._dltensor_wrapper = _cute_ir.DLTensorWrapper(self._dlpack_data)
            return func(self, *args, **kwargs)

        return wrapper

    @lazily_load_dltensor
    def mark_layout_dynamic(self, leading_dim: int | None = None):
        """Marks the tensor layout as dynamic based on the leading dimension.

        :param leading_dim: The leading dimension of the layout, defaults to None
        :type leading_dim: int, optional

        When ``leading_dim`` is None, automatically deduces the leading dimension from the tensor layout.
        The layout can be deduced only when exactly one dimension has a stride of 1. Raises an error
        if the layout cannot be automatically deduced.

        When ``leading_dim`` is explicitly specified, marks the layout as dynamic while setting the
        stride at ``leading_dim`` to 1. Also validates that the specified ``leading_dim`` is consistent
        with the existing layout by checking that the corresponding stride of that dimension is 1.

        Limitation: only support flat layout for now. Will work on supporting nested layout in the future.

        :return: The tensor with dynamic layout
        :rtype: _Tensor
        """
        self._dltensor_wrapper.mark_layout_dynamic(leading_dim)
        return self

    @lazily_load_dltensor
    def mark_compact_shape_dynamic(
        self,
        mode: int,
        stride_order: tuple[int, ...] | None = None,
        divisibility: int = 1,
    ):
        """Marks the tensor shape as dynamic and propagates dynamic and divisibility information to the corresponding strides.

        :param mode: The mode of the compact shape, defaults to 0
        :type mode: int
        :param stride_order: Consistent with `torch.Tensor.dim_order`. Defaults to None.
        Indicates the order of the modes (dimensions) if the current layout were converted to row-major order.
        It starts from the outermost to the innermost dimension.
        :type stride_order: tuple[int, ...], optional
        :param divisibility: The divisibility constraint for the compact shape, defaults to 1
        :type divisibility: int, optional
        :return: The tensor with dynamic compact shape
        :rtype: _Tensor

        If ``stride_order`` is not provided, the stride ordering will be automatically deduced from the layout.
        Automatic deduction is only possible when exactly one dimension has a stride of 1 (compact layout).
        An error is raised if automatic deduction fails.

        If ``stride_order`` is explicitly specified, it does the consistency check with the layout.

        For example:
        - Layout: (4,2):(1,4) has stride_order: (1,0) indicates the innermost dimension is 0(`4:1`), the outermost dimension is 1(`2:4`)
        - Layout: (5,3,2,4):(3,1,15,30) has stride_order: (3,2,0,1) indicates the innermost dimension is 1(`3:1`), the outermost dimension is 3(`4:30`).

        Using `torch.Tensor.dim_order()` to get the stride order of the torch tensor.
        .. code-block:: python
            a = torch.empty(3, 4)
            t = cute.runtime.from_dlpack(a)
            t = t.mark_compact_shape_dynamic(mode=0, stride_order=a.dim_order())
        """
        self._dltensor_wrapper.mark_compact_shape_dynamic(
            mode, stride_order, divisibility
        )
        return self

    @property
    @lazily_load_dltensor
    def element_type(self) -> Type[Numeric]:
        if self._dtype is None:
            self._dtype = self._dltensor_wrapper.dtype
        return self._dtype

    @element_type.setter
    def element_type(self, new_type):
        """Set the element type of the tensor.

        :warning: This API is added for narrow precision before we have a clean `recast_tensor` story.

        :note: It is only used for the case that frameworks don't natively support narrow precision but we get tensor
              from frameworks with storage type like uint8.

        **Example**:

        .. code-block:: python

            # Create a tensor from a numpy array
            import numpy as np
            from cutlass.cute import from_dlpack

            # Create a tensor with Float32 elements
            a = np.zeros(shape, dtype=np.uint8)
            tensor = from_dlpack(a)

            # Change the element type to Float4E2M1FN even storage type is uint8
            tensor.element_type = cutlass.Float4E2M1FN

            src = from_dlpack(... data tensor ...)
            # convert and initialize narrow precision tensor
            cute.testing.convert(src, tensor)
        """
        self._dtype = new_type

    @property
    @lazily_load_dltensor
    def memspace(self):
        return self._dltensor_wrapper.address_space

    @property
    @lazily_load_dltensor
    def size_in_bytes(self) -> int:
        return self._dltensor_wrapper.size_in_bytes()

    @property
    @lazily_load_dltensor
    def mlir_type(self) -> ir.Type:
        return self._dltensor_wrapper.get_type(
            self.element_type.mlir_type, self._assumed_align
        )

    @lazily_load_dltensor
    def __str__(self) -> str:
        return f"Tensor<0x{self._dltensor_wrapper.str}>"

    def __repr__(self):
        return self.__str__()

    def __setitem__(self, crd, value):
        raise TypeError(f"runtime._Tensor is not indexable")

    def __getitem__(self, crd):
        raise TypeError(f"runtime._Tensor is not indexable")

    @property
    @lazily_load_dltensor
    def iterator(self):
        return _Pointer(
            self._dltensor_wrapper.data_ptr,
            self.element_type,
            self.memspace,
            self._assumed_align,
        )

    @property
    def layout(self):
        raise NotImplementedError(
            f"layout property is not supported in runtime, support in future"
        )

    @property
    @lazily_load_dltensor
    def shape(self):
        return self._dltensor_wrapper.shape

    @property
    @lazily_load_dltensor
    def stride(self):
        strides = self._dltensor_wrapper.stride
        if strides is None:
            strides = itertools.accumulate(
                reversed(self.shape), func=operator.mul, initial=1
            )
            strides = tuple(reversed(list(strides)[:-1]))

        return strides

    @property
    @lru_cache(maxsize=128, typed=True)
    def leading_dim(self):
        """Get the leading dimension of this Tensor.

        :return: The leading dimension index or indices
        :rtype: int or tuple or None

        The return value depends on the tensor's stride pattern:

        * If a single leading dimension is found, returns an integer index
        * If nested leading dimensions are found, returns a tuple of indices
        * If no leading dimension is found, returns None
        """
        return core.leading_dim(self.shape, self.stride)

    def fill(self, value: Numeric):
        raise TypeError(f"fill function is not supported in runtime")

    @property
    @lazily_load_dltensor
    def data_ptr(self):
        return self._dltensor_wrapper.data_ptr

    @lazily_load_dltensor
    def __c_pointers__(self):
        self._memref_desc = self._dltensor_wrapper.build_memref_desc(
            self._assumed_align
        )
        return [_cute_ir.pycapsule_get_pointer(self._memref_desc)]

    def __get_mlir_types__(self):
        return [self.mlir_type]

    def __new_from_mlir_values__(self, values):
        assert len(values) == 1
        assert isinstance(values[0], CoreTensor)
        return CoreTensor(values[0].value, self._dtype)


def from_dlpack(
    tensor_dlpack,
    assumed_align=None,
) -> Tensor:
    """Convert from tensor object supporting __dlpack__() to a CuTe Tensor.

    :param tensor_dlpack: Tensor object that supports the DLPack protocol
    :type tensor_dlpack: object
    :param assumed_align: Assumed alignment of the tensor (bytes), defaults to None,
      if None, will use the element size bytes as the assumed alignment.
    :type assumed_align: int, optional
    :return: A CuTe Tensor object
    :rtype: Tensor

    Examples:
        .. code-block:: python

            import torch
            from cutlass.cute.runtime import from_dlpack
            x = torch.randn(100, 100)
            y = from_dlpack(x)
            y.shape
            # (100, 100)
            type(y)
            # <class 'cutlass.cute.Tensor'>
    """
    return _Tensor(
        tensor_dlpack,
        assumed_align=assumed_align,
    )


def make_ptr(
    dtype: Type[Numeric],
    value: Union[int, ctypes._Pointer],
    mem_space: AddressSpace = AddressSpace.generic,
    assumed_align=None,
) -> Pointer:
    """Create a pointer from a memory address

    :param dtype: Data type of the pointer elements
    :type dtype: Type[Numeric]
    :param value: Memory address as integer or ctypes pointer
    :type value: Union[int, ctypes._Pointer]
    :param mem_space: Memory address space, defaults to AddressSpace.generic
    :type mem_space: AddressSpace, optional
    :param align_bytes: Alignment in bytes, defaults to None
    :type align_bytes: int, optional
    :return: A pointer object
    :rtype: Pointer

    .. code-block:: python

        import numpy as np
        import ctypes

        from cutlass import Float32
        from cutlass.cute.runtime import make_ptr

        # Create a numpy array
        a = np.random.randn(16, 32).astype(np.float32)

        # Get pointer address as integer
        ptr_address = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Create pointer from address
        y = make_ptr(cutlass.Float32, ptr_address)

        # Check properties
        print(y.element_type)
        print(type(y))  # <class 'cutlass.cute.Pointer'>
    """
    # check if value is int or ctypes.POINTER
    if isinstance(value, int):
        address_value = value
    elif isinstance(value, ctypes._Pointer):
        # get address value
        address_value = ctypes.cast(value, ctypes.c_void_p).value
        assert address_value is not None, "Pointer address is None"
    else:
        raise TypeError(
            f"Expect int or ctypes.POINTER for value but got {type(value)=}"
        )

    return _Pointer(address_value, dtype, mem_space, assumed_align=assumed_align)


class TensorAdapter:
    """
    Convert a DLPack protocol supported tensor/array to a cute tensor.
    """

    def __init__(self, arg):
        self._arg = from_dlpack(arg).mark_layout_dynamic()

    def __new_from_mlir_values__(self, values):
        return self._arg.__new_from_mlir_values__(values)

    def __c_pointers__(self):
        return self._arg.__c_pointers__()

    def __get_mlir_types__(self):
        return self._arg.__get_mlir_types__()


# -------------------------------------------------------------------------
# Try to register_jit_arg_adapter for TensorAdapter
# -------------------------------------------------------------------------

try:  # Register for numpy.ndarray
    import numpy

    JitArgAdapterRegistry.register_jit_arg_adapter(numpy.ndarray)(TensorAdapter)
except ImportError:
    pass  # silent attempt, suppress error

try:  # Register for torch.Tensor
    import torch

    JitArgAdapterRegistry.register_jit_arg_adapter(torch.Tensor)(TensorAdapter)
except ImportError:
    pass  # silent attempt, suppress error
