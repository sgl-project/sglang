from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, override_jit_cuda_arch
from sglang.kernel_api_logging import debug_kernel_api
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def _fp8_blockwise_cuda_flags() -> list[str]:
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


@contextmanager
def _fp8_blockwise_arch_env():
    if not torch.cuda.is_available():
        raise RuntimeError("fp8_blockwise_scaled_mm JIT kernel requires CUDA.")
    major, minor = torch.cuda.get_device_capability()
    if major < 12:
        raise RuntimeError(
            "fp8_blockwise_scaled_mm JIT kernel requires compute capability >= 12.0 "
            f"(SM120/Blackwell), got {major}.{minor}."
        )
    # The SM120 blockwise kernels use architecture-family-specific instructions
    # and must be compiled for an `sm_*a` target (e.g. sm_120a), not plain sm_120.
    with override_jit_cuda_arch(major, minor, suffix="a"):
        yield


@cache_once
def _jit_fp8_blockwise_module() -> Module:
    """Compile and cache the SM120 fp8 blockwise GEMM module (handles fp16 + bf16)."""
    with _fp8_blockwise_arch_env():
        return load_jit(
            "fp8_blockwise_scaled_mm",
            cuda_files=["gemm/fp8_blockwise/fp8_blockwise_scaled_mm_entry.cuh"],
            cuda_wrappers=[
                ("fp8_blockwise_scaled_mm", "fp8_blockwise_scaled_mm"),
                ("fp8_blockwise_scaled_mm_noswap", "fp8_blockwise_scaled_mm_noswap"),
                ("fp8_blockwise_scaled_mm_swapab32", "fp8_blockwise_scaled_mm_swapab32"),
                ("fp8_blockwise_scaled_mm_swapab64", "fp8_blockwise_scaled_mm_swapab64"),
            ],
            extra_dependencies=["cutlass"],
            extra_cuda_cflags=_fp8_blockwise_cuda_flags(),
        )


@register_custom_op(
    op_name="fp8_blockwise_scaled_mm",
    mutates_args=["out"],
)
def _fp8_blockwise_scaled_mm_custom_op(
    out: torch.Tensor,
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
) -> None:
    module = _jit_fp8_blockwise_module()
    module.fp8_blockwise_scaled_mm(out, mat_a, mat_b, scales_a, scales_b)


@debug_kernel_api
def fp8_blockwise_scaled_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """FP8 (e4m3) block-wise scaled matmul on SM120 (Blackwell).

    Computes ``out = (mat_a * scales_a) @ (mat_b * scales_b)`` where scales are
    per-128-block. ``mat_a`` is row-major [M, K], ``mat_b`` column-major [K, N];
    output is [M, N] in ``out_dtype`` (fp16 or bf16). ``scales_a`` must be
    M-major (column-major, ``stride(0) == 1``).

    No M padding is done here: the kernel routes small or non-4-aligned M through
    the swapAB path (tokens on the N tile), which has no M-alignment requirement.
    """
    assert out_dtype in (torch.float16, torch.bfloat16), (
        f"out_dtype must be Half or BFloat16, got {out_dtype}"
    )

    out = torch.empty(
        (mat_a.shape[0], mat_b.shape[1]),
        dtype=out_dtype,
        device=mat_a.device,
    )
    _fp8_blockwise_scaled_mm_custom_op(out, mat_a, mat_b, scales_a, scales_b)
    return out
