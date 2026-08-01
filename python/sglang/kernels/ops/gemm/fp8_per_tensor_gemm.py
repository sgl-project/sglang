from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernel_api_logging import debug_kernel_api
from sglang.kernels.jit.utils import cache_once, load_jit, override_jit_cuda_arch
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_ARCH_SUFFIX = {89: "", 90: "a", 100: "a", 120: "a"}


def _resolve_gemm_arch() -> int:
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    if sm >= 120:
        return 120
    if sm >= 100:
        return 100
    if sm >= 90:
        return 90
    if sm == 89:
        return 89
    raise RuntimeError(
        f"fp8_per_tensor_scaled_mm has no implementation for compute capability {major}.{minor}"
    )


def _fp8_per_tensor_cuda_flags(arch: int) -> list[str]:
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
        f"-DSGL_FP8_GEMM_SM={arch}",
    ]


@contextmanager
def _fp8_per_tensor_arch_env(arch: int):
    major, minor = torch.cuda.get_device_capability()
    with override_jit_cuda_arch(major, minor, suffix=_ARCH_SUFFIX[arch]):
        yield


@cache_once
def _jit_fp8_per_tensor_module(arch: int) -> Module:
    with _fp8_per_tensor_arch_env(arch):
        return load_jit(
            "fp8_per_tensor_scaled_mm",
            str(arch),
            cuda_files=["gemm/fp8_per_tensor/fp8_per_tensor_scaled_mm_entry.cuh"],
            cuda_wrappers=[
                ("fp8_per_tensor_scaled_mm", "fp8_per_tensor_scaled_mm"),
            ],
            extra_dependencies=["cutlass"],
            extra_cuda_cflags=_fp8_per_tensor_cuda_flags(arch),
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
    module = _jit_fp8_per_tensor_module(_resolve_gemm_arch())
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
