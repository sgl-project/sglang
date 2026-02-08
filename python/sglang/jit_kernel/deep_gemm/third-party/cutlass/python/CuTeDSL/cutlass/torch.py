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
from math import prod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Type, Union

from cutlass.cute.typing import (
    Numeric,
    Boolean,
    Float,
    Integer,
    TFloat32,
    Float8E4M3B11FNUZ,
    Float8E4M3FN,
    Float8E5M2,
    Float8E8M0FNU,
    Float4E2M1FN,
    Tensor,
)
from cutlass.cute.runtime import from_dlpack
import cutlass.cute as cute
import torch
import cuda.bindings.driver as cuda


def dtype(ty: Type[Numeric]):
    """
    Return the corresponding torch.dtype per the given DSL type
    """
    torch_dtype = getattr(torch, ty.__name__.lower(), None)

    torch_type_map = {
        Boolean: torch.bool,
        # TFloat32 is just alias of float32
        TFloat32: torch.float32,
        Float8E5M2: torch.float8_e5m2,
        Float8E4M3FN: torch.float8_e4m3fn,
        Float8E4M3B11FNUZ: torch.float8_e4m3fnuz,
    }
    if torch_dtype is None:
        torch_dtype = torch_type_map.get(ty)

    if torch_dtype is None:
        raise TypeError(f"{ty} is not supported by torch")
    return torch_dtype


def as_tensor(pointer, shape, torch_type):
    """Convert a pointer to a torch tensor"""
    if torch_type.itemsize == 1:
        cytype = ctypes.c_uint8
    elif torch_type.itemsize == 2:
        cytype = ctypes.c_uint16
    elif torch_type.itemsize == 4:
        cytype = ctypes.c_uint32
    elif torch_type.itemsize == 8:
        cytype = ctypes.c_uint64
    else:
        raise ValueError(f"Unsupported torch dtype: {torch_type}")
    cpointer = ctypes.cast(pointer, ctypes.POINTER(cytype))
    arr = (cpointer._type_ * prod(shape)).from_address(
        ctypes.addressof(cpointer.contents)
    )
    return torch.frombuffer(arr, dtype=torch_type).view(*shape)


@dataclass
class ScalarInitConfig:
    """Configuration for scalar initialization"""

    value: float = 0.0


@dataclass
class RandomInitConfig:
    """Configuration for random initialization"""

    min_val: int = -2
    max_val: int = 2


@dataclass
class GaussianInitConfig:
    """Configuration for Gaussian initialization"""

    mean: float = 0.0
    std: float = 1.0
    scale: float = 1.0


class TensorInitType(Enum):
    """Enumeration of tensor initialization types"""

    SKIP = "skip"
    SCALAR = "scalar"
    RANDOM = "random"
    GAUSSIAN = "gaussian"


def create_and_permute_torch_tensor(
    shape,
    dtype: "torch.dtype",
    permute_order=None,
    init_type: TensorInitType = TensorInitType.RANDOM,
    init_config: Optional[
        Union[RandomInitConfig, ScalarInitConfig, GaussianInitConfig]
    ] = None,
    device: Optional[torch.device] = None,
) -> "torch.Tensor":
    """
    Create a torch tensor with specified shape and dtype. Optionally permute it and initialize it with specified init type and config
    """
    init_dtype = torch.int32 if init_type == TensorInitType.RANDOM else torch.float32
    init_torch_tensor = torch.empty(*shape, dtype=init_dtype, device=device)
    if init_type == TensorInitType.SKIP:
        assert init_config is None
        f32_torch_tensor = init_torch_tensor
    elif init_type == TensorInitType.SCALAR:
        if init_config is None:
            init_config = ScalarInitConfig()
        else:
            if not isinstance(init_config, ScalarInitConfig):
                raise ValueError("init_config must be ScalarInitConfig()")
        f32_torch_tensor = init_torch_tensor.fill_(init_config.value)
    elif init_type == TensorInitType.RANDOM:
        if init_config is None:
            init_config = RandomInitConfig()
        else:
            if not isinstance(init_config, RandomInitConfig):
                raise ValueError("init_config must be RandomInitConfig()")
        f32_torch_tensor = init_torch_tensor.random_(
            init_config.min_val, init_config.max_val
        ).to(dtype=torch.float32)
    elif init_type == TensorInitType.GAUSSIAN:
        if init_config is None:
            init_config = GaussianInitConfig()
        else:
            if not isinstance(init_config, GaussianInitConfig):
                raise ValueError("init_config must be GaussianInitConfig()")
        f32_torch_tensor = init_torch_tensor.normal_(init_config.mean, init_config.std)
        f32_torch_tensor = f32_torch_tensor * init_config.scale
    else:
        raise ValueError(f"Invalid init type: {init_type}")

    if permute_order is not None:
        f32_torch_tensor = f32_torch_tensor.permute(permute_order)

    dtype_torch_tensor = f32_torch_tensor.to(dtype=dtype)

    return dtype_torch_tensor


def convert_cute_tensor(
    f32_torch_tensor: "torch.Tensor",
    cute_tensor: Tensor,
    dtype: Type[Numeric],
    is_dynamic_layout: bool = True,
) -> Tensor:
    """
    Change the value of the cute tensor to make its value converted from a fp32 torch tensor.
    Used for fp8 types tensor creatation now.
    """
    # if torch_tensor is on cpu, create a gpu copy
    if f32_torch_tensor.device.type == "cpu":
        f32_torch_tensor = f32_torch_tensor.cuda()

    # Fp8 type need explicit type conversion
    if dtype in {
        Float8E5M2,
        Float8E4M3FN,
        Float8E8M0FNU,
        Float4E2M1FN,
    }:
        fp32_cute_tensor = from_dlpack(f32_torch_tensor)
        if is_dynamic_layout:
            fp32_cute_tensor = fp32_cute_tensor.mark_layout_dynamic(
                f32_torch_tensor.dim_order()[-1]
            )
        # Copy and convert from f32 cute tensor to dtype cute tensor
        cute.testing.convert(fp32_cute_tensor, cute_tensor)
    return cute_tensor


def default_stream() -> cuda.CUstream:
    """
    Get default CUstream from torch stream
    """
    torch_stream = torch.cuda.default_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    return stream


def current_stream() -> cuda.CUstream:
    """
    Get current CUstream from torch stream
    """
    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    return stream


def matrix(
    l: int,
    mode0: int,
    mode1: int,
    is_mode0_major: bool,
    cutlass_dtype: Type[Numeric],
    init_type: TensorInitType = TensorInitType.RANDOM,
    init_config: Optional[
        Union[RandomInitConfig, ScalarInitConfig, GaussianInitConfig]
    ] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create a torch tensor for matrix

    :param l: length of the matrix
    :param mode0: mode0 of the matrix
    :param mode1: mode1 of the matrix
    :param is_mode0_major: whether the matrix is mode0 major
    :param cutlass_dtype: cutlass dtype of the matrix
    :param init_type: type of initialization
    :param init_config: configuration for initialization
    :param device: target torch device
    """

    shape = (l, mode1, mode0) if is_mode0_major else (l, mode0, mode1)
    permute_order = (2, 1, 0) if is_mode0_major else (1, 2, 0)

    if cutlass_dtype.is_float and cutlass_dtype.width <= 8:
        torch_dtype = torch.int8
    else:
        torch_dtype = dtype(cutlass_dtype)

    if init_type == TensorInitType.RANDOM and init_config is None:
        if torch_dtype.is_signed:
            min_val = -2
            max_val = 2
        else:
            min_val = 0
            max_val = 4
        init_config = RandomInitConfig(min_val=min_val, max_val=max_val)

    # Create dtype torch tensor
    torch_tensor = create_and_permute_torch_tensor(
        shape,
        torch_dtype,
        permute_order=permute_order,
        init_type=init_type,
        init_config=init_config,
        device=device,
    )

    return torch_tensor


def cute_tensor_like(
    data_ref: torch.Tensor,
    cutlass_dtype: Type[Numeric],
    is_dynamic_layout: bool,
    assumed_align: Optional[int] = None,
) -> tuple[Tensor, torch.Tensor]:
    """
    Create a cute tensor use a torch tensor as the data source

    :param data_ref: torch tensor as the data source
    :param cutlass_dtype: cutlass dtype of the cute tensor
    :param is_dynamic_layout: whether the cute tensor uses dynamic layout
    :param assumed_align: assumed alignment of the cute tensor
    """

    # allocate device buffer for cute tensor
    if cutlass_dtype.is_float and cutlass_dtype.width <= 8:
        torch_dtype = torch.int8
    else:
        torch_dtype = dtype(cutlass_dtype)
    torch_tensor = torch.empty_like(data_ref, dtype=torch_dtype, device="cuda")

    # create cute tensor using the device buffer
    cute_tensor = from_dlpack(torch_tensor, assumed_align=assumed_align)
    cute_tensor.element_type = cutlass_dtype
    if is_dynamic_layout:
        for i, stride in enumerate(torch_tensor.stride()):
            if stride == 1:
                leading_dim = i
                break
        cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)

    # initialize the cute tensor data
    if cutlass_dtype.is_float and cutlass_dtype.width <= 8:
        cute_tensor = convert_cute_tensor(
            data_ref.to(dtype=torch.float32),
            cute_tensor,
            cutlass_dtype,
            is_dynamic_layout,
        )
    else:
        torch_tensor.copy_(data_ref.to(dtype=torch_dtype))

    return cute_tensor, torch_tensor
