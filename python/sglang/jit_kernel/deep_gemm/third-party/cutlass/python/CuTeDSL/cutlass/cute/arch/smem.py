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

from typing import Optional, Type

from cutlass.cutlass_dsl import T, dsl_user_op

import cutlass._mlir.dialects.cute as _cute_ir
import cutlass._mlir.dialects.cute_nvgpu as _cute_nvgpu_ir
from cutlass._mlir import ir

from ..typing import Pointer, Numeric, NumericMeta


@dsl_user_op
def alloc_smem(
    element_type: Type[Numeric],
    size_in_elems: int,
    alignment: Optional[int] = None,
    *,
    loc=None,
    ip=None,
) -> Pointer:
    """
    Statically allocates SMEM.

    :param element_type:  The pointee type of the pointer.
    :type element_type:   Type[Numeric]
    :param size_in_elems: The size of the allocation in terms of number of elements of the
                          pointee type
    :type size_in_elems:  int
    :param alignment:     An optional pointer alignment for the allocation
    :type alignment:      int
    :return:              A pointer to the start of the allocation
    :rtype:               Pointer
    """
    if not isinstance(element_type, NumericMeta):
        raise TypeError(
            f"element_type must be a type of Numeric, but got {element_type}"
        )

    if alignment is None:
        # Default alignment based on the element type's width
        alignment = element_type.width // 8
    ptr_ty = _cute_ir.PtrType.get(
        element_type.mlir_type, _cute_ir.AddressSpace.smem, alignment
    )
    return _cute_nvgpu_ir.arch_alloc_smem(
        ptr=ptr_ty,
        input=ir.IntegerAttr.get(T.i32(), size_in_elems),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def get_dyn_smem(
    element_type: Type[Numeric],
    alignment: Optional[int] = None,
    *,
    loc=None,
    ip=None,
) -> Pointer:
    """
    Retrieves a pointer to a dynamic SMEM allocation.

    :param element_type:  The pointee type of the pointer.
    :type element_type:   Type[Numeric]
    :param alignment:     An optional pointer alignment, the result pointer is offset appropriately
    :type alignment:      int
    :return:              A pointer to the start of the dynamic SMEM allocation with a correct
                          alignement
    :rtype:               Pointer
    """
    if not isinstance(element_type, NumericMeta):
        raise TypeError(
            f"element_type must be a type of Numeric, but got {element_type}"
        )

    if alignment is None:
        # Default alignment based on the element type's width
        alignment = element_type.width // 8
    ptr_ty = _cute_ir.PtrType.get(
        element_type.mlir_type,
        _cute_ir.AddressSpace.smem,
        alignment,
    )
    return _cute_nvgpu_ir.arch_get_dyn_smem(ptr=ptr_ty, loc=loc, ip=ip)


@dsl_user_op
def get_dyn_smem_size(*, loc=None, ip=None) -> int:
    """
    Gets the size in bytes of the dynamic shared memory that was specified at kernel launch time.
    This can be used for bounds checking during shared memory allocation.

    :return: The size of dynamic shared memory in bytes
    :rtype:  int
    """
    return _cute_nvgpu_ir.arch_get_dyn_smem_size(loc=loc, ip=ip)
