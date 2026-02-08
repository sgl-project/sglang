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
This module provides helper structs for dlpack.
DLPack is an open standard for in-memory tensor structures, enabling
seamless sharing of tensors across different frameworks.
Learn more at: https://github.com/dmlc/dlpack
"""

import ctypes
import enum


class DLDeviceType(enum.IntEnum):
    """Enums for device types based on the DLPack specification."""

    kDLCPU = 1
    kDLGPU = 2
    kDLCPUPinned = 3


class DLDataTypeCode:
    """Enums for data type codes based on the DLPack specification.

    see https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h
    """

    kDLInt = 0
    kDLUInt = 1
    kDLFloat = 2
    kDLOpaqueHandle = 3
    kDLBfloat = 4
    kDLComplex = 5
    kDLBool = 6


class DLDevice(ctypes.Structure):
    """Structure representing the device information in DLPack."""

    _fields_ = [
        ("device_type", ctypes.c_int),  # kDLCPU, kDLGPU, etc.
        ("device_id", ctypes.c_int),  # Device ID (e.g., GPU ID)
    ]


class DLDataType(ctypes.Structure):
    """Structure representing the data type in DLPack."""

    _fields_ = [
        ("code", ctypes.c_uint8),  # Data type code (e.g., kDLFloat)
        ("bits", ctypes.c_uint8),  # Number of bits per value
        ("lanes", ctypes.c_uint16),  # Number of lanes
    ]


class DLTensor(ctypes.Structure):
    """Structure representing the DLTensor in DLPack."""

    _fields_ = [
        ("data", ctypes.c_void_p),  # Pointer to tensor data
        ("device", DLDevice),  # Device info
        ("ndim", ctypes.c_int),  # Number of dimensions
        ("dtype", DLDataType),  # Data type
        ("shape", ctypes.POINTER(ctypes.c_int64)),  # Shape of tensor
        ("strides", ctypes.POINTER(ctypes.c_int64)),  # Strides of tensor
        ("byte_offset", ctypes.c_uint64),  # Byte offset to tensor data
    ]
