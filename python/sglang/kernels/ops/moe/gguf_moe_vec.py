"""GGUF vector expert projections with consumer-owned inactive routes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    get_jit_cuda_arch,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_gguf_moe_vec_module(dtype: torch.dtype, quant_type: int) -> Module:
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Unsupported GGUF input dtype: {dtype}")
    if quant_type not in (
        2,
        3,
        6,
        7,
        8,
        10,
        11,
        12,
        13,
        14,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        29,
    ):
        raise ValueError(f"Unsupported GGUF vector quantization type: {quant_type}")
    args = make_cpp_args(dtype, quant_type)
    arch = get_jit_cuda_arch()
    # Match the AOT common_ops SM90 fast-math build; its other build is precise.
    flags = ["--use_fast_math"] if (arch.major, arch.minor) == (9, 0) else []
    return load_jit(
        "gguf_moe_vec",
        *args,
        cuda_files=["moe/gguf_moe_vec.cuh"],
        cuda_wrappers=[("project", f"gguf_moe_a8_vec<{args}>")],
        extra_cuda_cflags=flags,
    )


@debug_kernel_api
def gguf_moe_a8_vec(
    input: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    type: int,
    row: int,
    tokens: int,
) -> torch.Tensor:
    """Project packed GGUF experts, writing zero for every negative route.

    Inputs and uint8 packed weights are contiguous. IDs are contiguous int32
    local expert indices or negative sentinels; their total size is tokens * top_k.
    The down projection retains the original route shape with top_k=1.
    Quantization, vector reduction and output rounding match the AOT operation.
    """
    output = torch.zeros((tokens * top_k, row), dtype=input.dtype, device=input.device)
    padded = (input.shape[1] + 511) // 512 * 512
    quantized_input = torch.empty(
        (tokens, padded // 32 * 9), dtype=torch.int32, device=input.device
    )
    _jit_gguf_moe_vec_module(input.dtype, int(type)).project(
        input, weight, topk_ids, output, quantized_input, top_k, row, tokens
    )
    return output
