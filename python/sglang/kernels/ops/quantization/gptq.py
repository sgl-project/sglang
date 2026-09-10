from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.cpp_extension import include_paths, library_paths

from sglang.kernel_api_logging import debug_kernel_api
from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_gptq_module() -> Module:
    torch_library_flags = [f"-L{path}" for path in library_paths()]
    return load_jit(
        "gptq_kernel",
        cuda_files=["gemm/gptq/gptq_kernel.cuh"],
        cuda_wrappers=[
            ("gptq_gemm", "gptq_gemm_jit"),
            ("gptq_shuffle", "gptq_shuffle_jit"),
        ],
        extra_cuda_cflags=[
            "--use_fast_math",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ],
        extra_include_paths=include_paths(),
        extra_ldflags=torch_library_flags
        + ["-lcublas", "-ltorch", "-ltorch_cpu", "-ltorch_cuda", "-lc10", "-lc10_cuda"],
    )


@debug_kernel_api
def gptq_gemm(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_gptq_qzeros: torch.Tensor,
    b_gptq_scales: torch.Tensor,
    b_g_idx: torch.Tensor,
    use_shuffle: bool,
    bit: int,
) -> torch.Tensor:
    out = torch.empty((a.shape[0], b_q_weight.shape[1]), dtype=a.dtype, device=a.device)
    temp_dq = torch.empty(
        (b_q_weight.shape[0] * 32 // bit, b_q_weight.shape[1]),
        dtype=a.dtype,
        device=a.device,
    )
    _jit_gptq_module().gptq_gemm(
        a,
        b_q_weight,
        b_gptq_qzeros,
        b_gptq_scales,
        b_g_idx,
        out,
        temp_dq,
        use_shuffle,
        bit,
    )
    return out


@debug_kernel_api
def gptq_shuffle(q_weight: torch.Tensor, q_perm: torch.Tensor, bit: int) -> None:
    _jit_gptq_module().gptq_shuffle(q_weight, q_perm, bit)
