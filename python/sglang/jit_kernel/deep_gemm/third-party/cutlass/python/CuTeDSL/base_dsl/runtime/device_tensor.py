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

import copy

from . import cuda as cuda_helpers
from .tensor_descriptor import *
from ..common import *


def allocate(tensor: TensorDescriptor, stream=None):
    """
    Allocates GPU memory
    """
    if tensor._check_is_managed_by_framework():
        raise DSLRuntimeError(
            "GPU tensors are managed by the framework and cannot be modified."
        )
    if not tensor.device_pointer is None:
        raise DSLRuntimeError("Tensor is already allocated on the device.")

    tensor.device_pointer = cuda_helpers.allocate(tensor.size_in_bytes, stream)

    log().info("Allocate done tensor=[%s] dev_ptr=[%s]", tensor, tensor.device_pointer)


def deallocate(tensor: TensorDescriptor, stream=None):
    """
    Deallocates GPU memory
    """
    if tensor._check_is_managed_by_framework():
        raise DSLRuntimeError(
            "GPU tensors are managed by the framework and cannot be modified."
        )
    if tensor.device_pointer is None:
        raise DSLRuntimeError("Tensor is not allocated on the device.")

    log().info(
        "Deallocating done tensor=[%s] dev_ptr=[%s]", tensor, tensor.device_pointer
    )

    cuda_helpers.deallocate(tensor.device_pointer, stream)
    tensor.device_pointer = None


def copy_to_gpu(tensor: TensorDescriptor, do_allocate=True, stream=None):
    """
    Copies data from host memory to the GPU memory.
    If do_allocate is True, it first calls allocate
    """
    log().info("copyin tensor=[%s] dev_ptr=[%s]", tensor, tensor.device_pointer)
    if do_allocate:
        allocate(tensor, stream)
    cuda_helpers.memcpy_h2d(
        tensor.data_ptr, tensor.device_pointer, tensor.size_in_bytes, stream
    )
    log().info("copyin done tensor=[%s] dev_ptr=[%s]", tensor, tensor.device_pointer)
    return tensor


def copy_from_gpu(tensor: TensorDescriptor, do_deallocate=True, stream=None):
    """
    Copies data from GPU memory back to the host.
    If do_deallocate is True, it calls deallocate
    """
    log().info("copyout tensor=[%s] dev_ptr=[%s]", tensor, tensor.device_pointer)
    if tensor._check_is_managed_by_framework():
        raise DSLRuntimeError(
            "GPU tensors are managed by the framework and cannot be modified."
        )
    if tensor.device_pointer is None:
        raise DSLRuntimeError("Tensor is not allocated on the device.")

    cuda_helpers.memcpy_d2h(
        tensor.data_ptr, tensor.device_pointer, tensor.size_in_bytes, stream
    )
    if do_deallocate:
        deallocate(tensor, stream)
    log().info("copyout done tensor=[%s] dev_ptr=[%s]", tensor, tensor.device_pointer)


def to_gpu(tensor, stream=None) -> TensorDescriptor:
    """
    Copies the tensor to the GPU memory from Host memory
    """
    if isinstance(tensor, TensorDescriptor):
        new_tensor = copy.copy(tensor)
        copy_to_gpu(new_tensor, stream=stream)
        return new_tensor

    if TensorDescriptor.can_transformed_to_dlpack(tensor):
        new_tensor = TensorDescriptor(tensor)
        copy_to_gpu(new_tensor, stream=stream)
        return new_tensor

    raise DSLRuntimeError("Unsupported type")


def from_gpu(tensor, stream=None) -> TensorDescriptor:
    """
    Copies the tensor to the GPU memory from Host memory
    """
    if isinstance(tensor, TensorDescriptor):
        new_tensor = copy.copy(tensor)
        copy_from_gpu(new_tensor, stream=stream)
        return new_tensor

    if TensorDescriptor.can_transformed_to_dlpack(tensor):
        new_tensor = TensorDescriptor(tensor)
        copy_from_gpu(new_tensor, stream=stream)
        return new_tensor

    raise DSLRuntimeError("Unsupported type")
