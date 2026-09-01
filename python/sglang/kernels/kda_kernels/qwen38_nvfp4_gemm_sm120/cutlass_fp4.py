# SPDX-License-Identifier: Apache-2.0

# KDA provenance: this kernel was automatically optimized by the Humanize2
# workflow (https://github.com/PolyArch/humanize) and Kernel Design Agents
# (https://github.com/mit-han-lab/kernel-design-agents).
# Source: https://github.com/BBuf/KDA-Pilot/pull/195 @
# 516c976cee824a236679adf6eb525275a0a9a120.
"""CUTLASS SM120 NVFP4 GEMM, compiled through sglang.kernels.jit / tvm-ffi."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_ROOT = Path(__file__).resolve().parent
_CUDA_FILE = str(_ROOT / "cutlass_fp4_cuda.cu")


def _cutlass_fp4_cuda_flags() -> list[str]:
    return [
        "-DNDEBUG",
        "-DCUTE_USE_PACKED_TUPLE=1",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        "--expt-extended-lambda",
        "-static-global-template-stub=false",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
    ]


@cache_once
def _jit_cutlass_fp4_module() -> Module:
    return load_jit(
        "kda_qwen38_cutlass_fp4_sm120",
        cuda_files=[_CUDA_FILE],
        extra_dependencies=["cutlass"],
        extra_cuda_cflags=_cutlass_fp4_cuda_flags(),
        header_only=False,
    )


def cutlass_fp4_gemm(
    output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scales: torch.Tensor,
    weight_scales: torch.Tensor,
    alpha: torch.Tensor,
) -> None:
    _jit_cutlass_fp4_module().cutlass_fp4_gemm(
        output,
        input,
        weight,
        input_scales,
        weight_scales,
        alpha,
    )
