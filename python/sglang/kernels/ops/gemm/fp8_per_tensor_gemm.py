from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernel_api_logging import debug_kernel_api
from sglang.kernels.jit.utils import (
    cache_once,
    get_jit_cuda_arch,
    load_jit,
    override_jit_cuda_arch,
)
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_CUDA_FLAGS = [
    "-DNDEBUG",
    "-DCUTE_USE_PACKED_TUPLE=1",
    "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
    "-DCUTLASS_VERSIONS_GENERATED",
    "-DCUTLASS_TEST_LEVEL=0",
    "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1",
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
]


@contextmanager
def _cutlass_arch_env():
    # CUTLASS gates its SM90+ arch macros on the sm_*a target; sm_*f misses them.
    arch = get_jit_cuda_arch()
    suffix = "a" if arch.major >= 9 else arch.suffix
    with override_jit_cuda_arch(arch.major, arch.minor, suffix=suffix):
        yield


@cache_once
def _jit_fp8_per_tensor_module() -> Module:
    with _cutlass_arch_env():
        return load_jit(
            "fp8_per_tensor_scaled_mm",
            cuda_files=["gemm/fp8_per_tensor/fp8_per_tensor_scaled_mm_entry.cuh"],
            cuda_wrappers=[
                ("fp8_per_tensor_scaled_mm", "fp8_per_tensor_scaled_mm"),
            ],
            extra_dependencies=["cutlass"],
            extra_cuda_cflags=_CUDA_FLAGS,
        )


@register_custom_op(
    op_name="fp8_per_tensor_scaled_mm",
    mutates_args=["out"],
)
def _fp8_per_tensor_scaled_mm_custom_op(
    out: torch.Tensor,
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> None:
    module = _jit_fp8_per_tensor_module()
    module.fp8_per_tensor_scaled_mm(out, mat_a, mat_b, scales_a, scales_b, bias)


@debug_kernel_api
def fp8_per_tensor_scaled_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert out_dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"out_dtype must be Half or BFloat16, got {out_dtype}"

    out = torch.empty(
        (mat_a.shape[0], mat_b.shape[1]),
        dtype=out_dtype,
        device=mat_a.device,
    )
    _fp8_per_tensor_scaled_mm_custom_op(out, mat_a, mat_b, scales_a, scales_b, bias)
    return out
