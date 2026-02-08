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

from typing import Type, Union, overload

from cutlass.cutlass_dsl import Int8, Numeric, NumericMeta, CutlassBaseDSL

import cutlass.cute as cute
from cutlass.cute.arch import get_dyn_smem, get_dyn_smem_size


class SmemAllocator:
    """A class for managing shared memory allocation on GPU.

    This class manages a chunk of shared memory and provides APIs for sub-allocation
    inside the chunk.

    :ivar _base: The current base address of the shared memory as an i8 typed dynamic value.
    :type _base: cute.Pointer
    :ivar _allocated_bytes: The total number of bytes allocated in shared memory.
    :type _allocated_bytes: int

    .. note::
        This class is responsible for managing the allocation of tensors in shared memory.
        The base pointer is aligned to 1024 bytes upon initialization.
    """

    def __init__(self):
        """Initialize the SmemAllocator instance.

        Creates a dynamic shared memory base pointer of type i8, aligned to 1024 bytes.
        """
        self._base = get_dyn_smem(Int8, alignment=1024)
        self._allocated_bytes = 0
        CutlassBaseDSL.track_smem_allocator(self, lambda cls: cls._allocated_bytes)

    @overload
    def allocate(self, size_or_type: int, byte_alignment: int) -> cute.Pointer: ...

    @overload
    def allocate(
        self, size_or_type: cute.struct, byte_alignment: int
    ) -> cute.Pointer: ...

    def allocate(self, size_or_type, byte_alignment: int = 1) -> cute.Pointer:
        """Allocate a block of memory with specified size and alignment.

        This method adjusts the base pointer to ensure proper alignment and updates
        the internal state to track allocated memory.

        :param size_or_type: The number of bytes to allocate or a struct class
        :type size_or_type: Union[int, cute.struct]
        :param byte_alignment: The byte alignment requirement, defaults to 1 (no alignment)
        :type byte_alignment: int, optional
        :return: Pointer to the start of the allocated memory block or struct instance
        :rtype: cute.Pointer
        :raises ValueError: If size is negative or alignment is less than 1
        :raises RuntimeError: If allocation would exceed available shared memory
        """
        if isinstance(size_or_type, cute.struct):
            alignment = max(byte_alignment, size_or_type.__alignof__())
            base_ptr = self.allocate(size_or_type.__sizeof__(), alignment)
            return size_or_type(base_ptr)

        num_bytes = size_or_type
        if num_bytes < 0:
            raise ValueError("num_bytes must be non-negative")
        if byte_alignment < 1:
            raise ValueError("byte_alignment must be at least 1")

        self._base = self._base.align(byte_alignment)
        ptr = self._base
        self._base += num_bytes
        if self._allocated_bytes % byte_alignment != 0:
            self._allocated_bytes += (
                byte_alignment - self._allocated_bytes % byte_alignment
            )
        self._allocated_bytes += num_bytes

        # Check bounds against available dynamic shared memory
        cute.testing.assert_(
            self._allocated_bytes <= get_dyn_smem_size(),
            f"Allocation failed: shared memory allocation exceeds available memory set in kernel launch. "
            f"Allocated bytes: {self._allocated_bytes} bytes. "
            f"Please reduce the allocation or set a larger smem size in kernel launch.",
        )
        return ptr

    def allocate_array(self, element_type: Type[Numeric], num_elems: int = 1):
        """Allocate an array of elements in shared memory.

        :param element_type: The type of elements to allocate
        :type element_type: Type[Numeric]
        :param num_elems: Number of elements to allocate, defaults to 1
        :type num_elems: int, optional
        :return: Pointer to the start of the allocated array
        :rtype: cute.Pointer
        :raises ValueError: If num_elems is less than 1
        :raises TypeError: If element_type is not a Numeric type
        """
        if num_elems < 1:
            raise ValueError("num_elems must be at least 1")
        if not isinstance(element_type, NumericMeta):
            raise TypeError(
                f"value_ty must be a type of Numeric, but got {element_type}"
            )

        ptr = self.allocate(
            element_type.width // 8 * num_elems, element_type.width // 8
        )

        return cute.recast_ptr(ptr, dtype=element_type)

    def allocate_tensor(
        self,
        element_type: Type[Numeric],
        layout: Union[int, cute.Layout, cute.ComposedLayout],
        byte_alignment: int = 1,
        swizzle: cute.Swizzle = None,
    ):
        """Allocate a tensor in shared memory.

        :param element_type: The type of elements in the tensor
        :type element_type: Type[Numeric]
        :param layout: The layout specification for the tensor
        :type layout: Union[int, cute.Layout, cute.ComposedLayout]
        :param byte_alignment: The byte alignment requirement, defaults to 1
        :type byte_alignment: int, optional
        :param swizzle: Swizzle for position-dependent swizzling, defaults to None
        :type swizzle: cute.Swizzle, optional
        :return: The allocated tensor with specified properties
        :rtype: cute.Tensor
        :raises TypeError: If element_type is not a Numeric type or if swizzle conflicts with layout
        :raises ValueError: If allocation is not byte-aligned
        :raises NotImplementedError: If dynamic layout is specified
        """
        if not isinstance(element_type, NumericMeta):
            raise TypeError(
                f"value_ty must be a type of Numeric, but got {element_type}"
            )

        if (
            isinstance(layout, cute.ComposedLayout)
            and isinstance(layout.inner, cute.Swizzle)
        ) and (swizzle is not None):
            raise TypeError(
                f"Invalid tensor type: cannot be both iterator swizzle (PDSL) and swizzle layout(PISL) at the same time."
            )

        if isinstance(layout, int):
            layout = cute.make_layout(layout)

        profile = layout(0)
        if isinstance(profile, tuple):
            raise TypeError(
                f"cannot allocate a shared memory tensor with a non-integer iterator"
            )

        if not cute.is_static(layout.type):
            raise NotImplementedError(f"dynamic layout is not supported: {layout.type}")

        # At least align the allocation to the natural alignment given by the element type
        if element_type.width // 8 > byte_alignment:
            byte_alignment = element_type.width // 8

        # Relevant only for sub-byte data types: verify that the entire allocation is byte-aligned
        cosize_in_bits = cute.cosize(layout) * element_type.width
        assert isinstance(cosize_in_bits, int)
        if cosize_in_bits % 8 != 0:
            raise ValueError("invalid allocation that is not byte-aligned")

        num_bytes = cosize_in_bits // 8
        ptr = self.allocate(num_bytes, byte_alignment)
        ptr = cute.recast_ptr(ptr, swizzle, dtype=element_type)
        res = cute.make_tensor(ptr, layout)
        return res
