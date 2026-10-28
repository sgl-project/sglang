from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import KERNEL_PATH, cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

DEEP_SELECT_INPUT_STRIDE_ALIGNMENT = 256
_DEEP_SELECT_SUPPORTED_K = (512,)


@cache_once
def _jit_deep_select_topk_module(topk: int) -> Module:
    if topk not in _DEEP_SELECT_SUPPORTED_K:
        raise RuntimeError(
            f"Unsupported topk {topk}. Supported: {_DEEP_SELECT_SUPPORTED_K}"
        )
    root = (KERNEL_PATH / "csrc" / "deep_select").resolve()
    args = make_cpp_args(topk)
    return load_jit(
        "deep_select_topk",
        *args,
        cuda_files=["elementwise/deep_select_topk.cuh"],
        cuda_wrappers=[("topk", f"DeepSelectTopKFp32Kernel<{args}>::run")],
        extra_include_paths=[str(root), str(root / "kerutils" / "include")],
        extra_dependencies=["flashinfer"],
        extra_cflags=["-DNDEBUG", "-DKERUTILS_IS_BUILD_ON_CUDA"],
        extra_cuda_cflags=[
            "-DNDEBUG",
            "-DKERUTILS_IS_BUILD_ON_CUDA",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-extended-lambda",
            "--use_fast_math",
            "--ftz=false",
        ],
    )


def deep_select_topk(
    scores: torch.Tensor,
    ends: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    _jit_deep_select_topk_module(out.shape[1]).topk(scores, ends, out)
    return out
