from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernel_api_logging import debug_kernel_api
from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def _fp8_scaled_mm_cuda_flags() -> list[str]:
    return [
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


@cache_once
def _jit_fp8_scaled_mm_module() -> Module:
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (8, 9) and major not in (9, 10, 12):
        raise RuntimeError(
            "fp8_scaled_mm JIT kernel requires SM89, SM90, SM100/SM103, or SM120."
        )
    return load_jit(
        "fp8_scaled_mm",
        cuda_files=["gemm/fp8_scaled_mm/fp8_scaled_mm.cuh"],
        cuda_wrappers=[("fp8_scaled_mm", "fp8_scaled_mm")],
        extra_dependencies=["cutlass"],
        extra_cuda_cflags=_fp8_scaled_mm_cuda_flags(),
    )


@register_custom_op(op_name="fp8_scaled_mm_jit", mutates_args=["out"])
def _fp8_scaled_mm_custom_op(
    out: torch.Tensor,
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> None:
    module = _jit_fp8_scaled_mm_module()
    module.fp8_scaled_mm(out, mat_a, mat_b, scales_a, scales_b, bias)


@debug_kernel_api
def fp8_scaled_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute ``(mat_a @ mat_b) * scales_a * scales_b (+ bias)``."""
    if out_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"out_dtype must be Half or BFloat16, got {out_dtype}")

    out = torch.empty(
        (mat_a.shape[0], mat_b.shape[1]), dtype=out_dtype, device=mat_a.device
    )
    _fp8_scaled_mm_custom_op(out, mat_a, mat_b, scales_a, scales_b, bias)
    return out


__all__ = ["fp8_scaled_mm"]
