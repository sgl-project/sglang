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

# Helpers
import itertools, operator
import ctypes
from . import dlpack_types as _dpack
from .dlpack_runtime import (
    dlpack_to_tensor_desc,
    get_tensor_desc_data_ptr,
    get_tensor_desc_is_in_device,
    get_tensor_desc_element_type,
    get_tensor_desc_shape,
    get_tensor_desc_stride,
    get_tensor_desc_element_size_in_bytes,
    get_tensor_desc_ndim,
    get_tensor_desc_dtype_code,
    get_tensor_desc_dtype_bits,
    get_tensor_desc_device_type,
    get_tensor_desc_device_id,
)

from ..utils.logger import log
from ..common import *
from ..typing import (
    Boolean,
    Float8E5M2,
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
)


class TensorDescriptor:
    def __init__(self, tensor):
        """Initialize with a tensor that supports the DLPack protocol.

        Args:
            tensor: Any tensor object that implements __dlpack__ and __dlpack_device__
        """

        self.tensor = tensor
        self._capsule = dlpack_to_tensor_desc(tensor)

        self.data_ptr = get_tensor_desc_data_ptr(self._capsule)
        self.device_type = get_tensor_desc_device_type(self._capsule)
        self.device_type = _dpack.DLDeviceType(self.device_type)

        if self.device_type == _dpack.DLDeviceType.kDLGPU:
            self.device_pointer = self.data_ptr
        elif self.device_type == _dpack.DLDeviceType.kDLCPU:
            self.device_pointer = None
        else:
            raise DSLRuntimeError(
                f"DLPack device type is not supported {self.dl_tensor.device.device_type}"
            )

        log().info("TensorDescriptor is created = [%s]", self)

    @staticmethod
    def can_transformed_to_dlpack(dl_tensor):
        if not hasattr(dl_tensor, "__dlpack__") or not hasattr(
            dl_tensor, "__dlpack_device__"
        ):
            return False
        return True

    @property
    def is_in_device(self):
        """Check if the tensor is stored on a device."""
        return not self.device_pointer is None

    @property
    def device_id(self):
        """Return device id where tensor resides."""
        if self.is_in_device:
            return get_tensor_desc_device_id(self._capsule)
        return -1

    @property
    def element_type(self):
        """Return the corresponding Python type based on DLPack dtype metadata."""
        str_element_type = get_tensor_desc_element_type(self._capsule)
        dtype_map = {
            # bool is 8bit from numpy and torch
            "Bool": Boolean,
            "Int64": Int64,
            "Int32": Int32,
            "Int16": Int16,
            "Int8": Int8,
            "UInt64": Uint64,
            "UInt32": Uint32,
            "UInt16": Uint16,
            "UInt8": Uint8,
            "Float64": Float64,
            "Float32": Float32,
            "Float16": Float16,
            "BFloat16": BFloat16,
            "Float8E5M2": Float8E5M2,
        }

        if str_element_type not in dtype_map:
            raise KeyError(
                f"Unsupported element type in dlpack: '{str_element_type}'. Supported types are: {list(dtype_map.keys())}"
            )

        return dtype_map[str_element_type]

    @property
    def shape(self):
        """Return the shape of the tensor."""
        return get_tensor_desc_shape(self._capsule)

    @property
    def rank(self):
        """Return the rank of the tensor."""
        return get_tensor_desc_ndim(self._capsule)

    @property
    def strides(self):
        """Return the rank of the tensor."""
        return get_tensor_desc_stride(self._capsule)

    @property
    def element_size_in_bytes(self):
        """Calculate the element size in bytes of the DLPack tensor."""
        return get_tensor_desc_element_size_in_bytes(self._capsule)

    @property
    def size_in_bytes(self):
        """Calculate the total size in bytes of the DLPack tensor."""
        # Calculate the number of elements using the shape
        ndim = get_tensor_desc_ndim(self._capsule)
        shape = get_tensor_desc_shape(self._capsule)
        num_elements = 1
        for i in range(ndim):
            num_elements *= shape[i]

        # Total bytes
        total_bytes = self.element_size_in_bytes * num_elements
        return total_bytes

    def __str__(self):
        """Return a compact string representation of the device_tensor with a tensor prefix."""
        # Extract shape
        shape = "x".join(map(str, self.shape))

        # Extract dtype
        dtype_code = get_tensor_desc_dtype_code(self._capsule)
        dtype_bits = get_tensor_desc_dtype_bits(self._capsule)
        dtype = (
            f"i{dtype_bits}"
            if dtype_code == _dpack.DLDataTypeCode.kDLInt
            else f"f{dtype_bits}"
        )

        # Extract device
        device_type = "cpu" if not self.is_in_device else "gpu"

        return f"tensor<{shape}x{dtype}>_{device_type}"

    def _check_is_managed_by_framework(self):
        """
        Ensure the tensor is not managed by the framework (e.g., GPU tensor).
        Raises an exception if the tensor is framework-managed.
        """
        return self.device_type == _dpack.DLDeviceType.kDLGPU

    @staticmethod
    def is_compatible(maybe_tensor_descriptor) -> bool:
        """Check if the object is a TensorDescriptor or can be converted to one."""
        return isinstance(
            maybe_tensor_descriptor, TensorDescriptor
        ) or TensorDescriptor.can_transformed_to_dlpack(maybe_tensor_descriptor)


def from_tensor(tensor) -> TensorDescriptor:
    """Create a TensorDescriptor from a tensor object."""
    return TensorDescriptor(tensor)


def to_tensor(tensor_descriptor: TensorDescriptor):
    """Return tensor object from tensor descriptor."""
    return tensor_descriptor.tensor
