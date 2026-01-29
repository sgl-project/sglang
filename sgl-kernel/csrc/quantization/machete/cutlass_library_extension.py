# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm project
# Adapted form https://github.com/vllm-project/vllm/blob/main/csrc/cutlass_extensions/vllm_cutlass_library_extension.py

import enum
from typing import Union

from cutlass_library import *

#
#   Extend cutlass library with custom types, and missing values
#


class SGLANGDataType(enum.Enum):
    u4b8 = enum_auto()
    u8b128 = enum_auto()


class MixedInputKernelScheduleType(enum.Enum):
    TmaWarpSpecialized = enum_auto()
    TmaWarpSpecializedPingpong = enum_auto()
    TmaWarpSpecializedCooperative = enum_auto()


SGLANGDataTypeNames: dict[Union[SGLANGDataType, DataType], str] = {
    **DataTypeNames,  # type: ignore
    **{
        SGLANGDataType.u4b8: "u4b8",
        SGLANGDataType.u8b128: "u8b128",
    },
}

SGLANGDataTypeTag: dict[Union[SGLANGDataType, DataType], str] = {
    **DataTypeTag,  # type: ignore
    **{
        SGLANGDataType.u4b8: "cutlass::sglang_uint4b8_t",
        SGLANGDataType.u8b128: "cutlass::sglang_uint8b128_t",
    },
}

SGLANGDataTypeSize: dict[Union[SGLANGDataType, DataType], int] = {
    **DataTypeSize,  # type: ignore
    **{
        SGLANGDataType.u4b8: 4,
        SGLANGDataType.u8b128: 8,
    },
}

SGLANGDataTypeSGLANGScalarTypeTag: dict[Union[SGLANGDataType, DataType], str] = {
    SGLANGDataType.u4b8: "sglang::kU4B8",
    SGLANGDataType.u8b128: "sglang::kU8B128",
    DataType.u4: "sglang::kU4",
    DataType.u8: "sglang::kU8",
    DataType.s4: "sglang::kS4",
    DataType.s8: "sglang::kS8",
    DataType.f16: "sglang::kFloat16",
    DataType.bf16: "sglang::kBfloat16",
}

SGLANGDataTypeTorchDataTypeTag: dict[Union[SGLANGDataType, DataType], str] = {
    DataType.u8: "at::ScalarType::Byte",
    DataType.s8: "at::ScalarType::Char",
    DataType.e4m3: "at::ScalarType::Float8_e4m3fn",
    DataType.s32: "at::ScalarType::Int",
    DataType.f16: "at::ScalarType::Half",
    DataType.bf16: "at::ScalarType::BFloat16",
    DataType.f32: "at::ScalarType::Float",
}

SGLANGKernelScheduleTag: dict[
    Union[MixedInputKernelScheduleType, KernelScheduleType], str
] = {
    **KernelScheduleTag,  # type: ignore
    **{
        MixedInputKernelScheduleType.TmaWarpSpecialized: "cutlass::gemm::KernelTmaWarpSpecialized",
        MixedInputKernelScheduleType.TmaWarpSpecializedPingpong: "cutlass::gemm::KernelTmaWarpSpecializedPingpong",
        MixedInputKernelScheduleType.TmaWarpSpecializedCooperative: "cutlass::gemm::KernelTmaWarpSpecializedCooperative",
    },
}
