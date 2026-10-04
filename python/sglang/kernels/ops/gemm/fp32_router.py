"""BF16 activations with unmodified FP32 router weights and FP32 output."""

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    get_jit_cuda_arch,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _module(m: int, block: int, epb: int, groups: int) -> "Module":
    if not 1 <= m <= 64 or not is_arch_support_pdl():
        raise ValueError("fp32_router requires 1 <= M <= 64 and CUDA SM90+")
    args = make_cpp_args(m, block, epb, groups)
    return load_jit(
        "fp32_router",
        *args,
        cuda_files=["gemm/fp32_router.cuh"],
        cuda_wrappers=[("run", f"Fp32RouterKernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


@register_custom_op(op_name="fp32_router", mutates_args=["out"])
def _run(
    x: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor,
    block: int,
    epb: int,
    groups: int,
) -> None:
    _module(x.shape[0], block, epb, groups).run(x, w, out)


@cache_once
def _geometry(m: int) -> tuple[int, int, int]:
    # Rotating 49 layer weights under CUDA graph replay on H200 and B300.
    # Avoid extra token groups for the smallest decode batches.
    if m == 16:
        return 128, 2, 2
    if m in (24, 32, 48, 64):
        block = (
            160
            if m == 24 or (m in (32, 64) and get_jit_cuda_arch().major == 9)
            else 128
        )
        return block, 2, 4
    return 128, 1, 1


@debug_kernel_api
def fp32_router_gemm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Compute ``x[M,5120] @ w[256,5120].T`` for 1 <= M <= 64.

    Activations are contiguous BF16, weights and outputs are FP32. Both inputs
    must be 16-byte aligned on the same CUDA device (SM90+). The reduction uses
    FP32 arithmetic, without TF32 or a reduced-precision weight conversion.
    Weights are read on every call, including CUDA graph replay.
    """
    out = torch.empty((x.shape[0], 256), device=x.device, dtype=torch.float32)
    _run(x, w, out, *_geometry(x.shape[0]))
    return out
