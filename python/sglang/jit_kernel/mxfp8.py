from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    load_jit,
    make_cpp_args,
    override_jit_cuda_arch,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def _mxfp8_cuda_flags() -> list[str]:
    return [
        "-DNDEBUG",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        "-DCUTLASS_VERSIONS_GENERATED",
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
        "--expt-extended-lambda",
    ]


def _mxfp8_arch_env():
    if not torch.cuda.is_available():
        raise RuntimeError("MXFP8 JIT kernels require CUDA.")
    major, minor = torch.cuda.get_device_capability()
    if major < 10:
        raise RuntimeError(
            f"MXFP8 JIT kernels require compute capability >= 10.0, got {major}.{minor}."
        )
    # MXFP8 kernels use architecture-family-specific instructions and must be
    # compiled for `sm_*a` targets (e.g. sm_100a), not plain sm_100.
    # JIT compilation targets only the current device, unlike AOT fat-binaries;
    # adding extra architectures here would clash with the single SGL_CUDA_ARCH
    # value injected by load_jit().
    return override_jit_cuda_arch(major, minor, suffix="a")


@cache_once
def _jit_es_sm100_mxfp8_blockscaled_group_quant(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    with _mxfp8_arch_env():
        return load_jit(
            "es_sm100_mxfp8_blockscaled_group_quant",
            *args,
            cuda_files=[
                "moe/expert_specialization/es_sm100_mxfp8_blockscaled_group_quant.cuh"
            ],
            cuda_wrappers=[
                (
                    "es_sm100_mxfp8_blockscaled_group_quant",
                    f"EsSm100MXFP8BlockscaledGroupQuant<{args}>::run",
                )
            ],
            extra_dependencies=["cutlass"],
            extra_cuda_cflags=_mxfp8_cuda_flags(),
        )


@cache_once
def _jit_es_sm100_mxfp8_blockscaled_moe_group_gemm(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    with _mxfp8_arch_env():
        return load_jit(
            "es_sm100_mxfp8_blockscaled_moe_group_gemm",
            *args,
            cuda_files=[
                "moe/expert_specialization/es_sm100_mxfp8_blockscaled_moe_group_gemm.cuh"
            ],
            cuda_wrappers=[
                (
                    "es_sm100_mxfp8_blockscaled_moe_group_gemm",
                    f"EsSm100MXFP8BlockscaledMoeGroupGemm<{args}>::run",
                )
            ],
            extra_dependencies=["cutlass"],
            extra_cuda_cflags=_mxfp8_cuda_flags(),
        )


def es_sm100_mxfp8_blockscaled_grouped_quant(
    input: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    quant_output: torch.Tensor,
    scale_factor: torch.Tensor,
) -> None:
    module = _jit_es_sm100_mxfp8_blockscaled_group_quant(input.dtype)
    module.es_sm100_mxfp8_blockscaled_group_quant(
        input,
        tokens_per_expert,
        expert_offsets,
        blockscale_offsets,
        quant_output,
        scale_factor,
    )


def es_sm100_mxfp8_blockscaled_moe_grouped_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    workspace: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    num_experts, m, tokens = a.shape[0], a.shape[1], b.shape[0]
    d = torch.empty((tokens, m), device=a.device, dtype=dtype)
    d_ptrs = torch.empty((num_experts,), device=a.device, dtype=torch.int64)
    b_ptrs = torch.empty((num_experts,), device=a.device, dtype=torch.int64)
    sfb_ptrs = torch.empty((num_experts,), device=a.device, dtype=torch.int64)
    module = _jit_es_sm100_mxfp8_blockscaled_moe_group_gemm(dtype)
    module.es_sm100_mxfp8_blockscaled_moe_group_gemm(
        a,
        b,
        sfa,
        sfb,
        expert_offsets,
        blockscale_offsets,
        tokens_per_expert,
        b_ptrs,
        sfb_ptrs,
        d,
        d_ptrs,
        workspace,
    )
    return d
