"""Rendering Python values as C++ template arguments."""

from __future__ import annotations

from typing import TypeAlias, Union

import torch

CPP_TEMPLATE_TYPE: TypeAlias = Union[int, float, str, bool, torch.dtype]


class CPPArgList(list):
    def __str__(self) -> str:
        return ", ".join(self)


CPP_DTYPE_MAP = {
    torch.float64: "double",
    torch.float32: "fp32_t",
    torch.float16: "fp16_t",
    torch.bfloat16: "bf16_t",
    # The fnuz variants are the ROCm-side torch dtypes; fp8_*_t resolves to
    # the matching HIP type there (see HIP_FP8_TYPE_* in utils.cuh).
    torch.float8_e4m3fn: "fp8_e4m3_t",
    torch.float8_e4m3fnuz: "fp8_e4m3_t",
    torch.float8_e5m2: "fp8_e5m2_t",
    torch.float8_e5m2fnuz: "fp8_e5m2_t",
    torch.int8: "int8_t",
    torch.int16: "int16_t",
    torch.int32: "int32_t",
    torch.int64: "int64_t",
    torch.uint8: "uint8_t",
    torch.uint16: "uint16_t",
    torch.uint32: "uint32_t",
    torch.uint64: "uint64_t",
    torch.bool: "bool",
}


def make_cpp_args(*args: CPP_TEMPLATE_TYPE) -> CPPArgList:
    def _convert(arg: CPP_TEMPLATE_TYPE) -> str:
        if isinstance(arg, bool):
            return "true" if arg else "false"
        if isinstance(arg, (int, str, float)):
            return str(arg)
        if isinstance(arg, torch.dtype):
            return CPP_DTYPE_MAP[arg]
        raise TypeError(f"Unsupported argument type for cpp template: {type(arg)}")

    return CPPArgList(_convert(arg) for arg in args)
