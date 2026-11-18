from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_argmax_module(dtype: torch.dtype) -> Module:
    assert dtype in _SUPPORTED_DTYPES, f"{dtype} not in {_SUPPORTED_DTYPES}"
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        "sampling_argmax",
        *args,
        cuda_files=["sampling/argmax.cuh"],
        cuda_wrappers=[("argmax", f"ArgmaxKernel<{args}>::run")],
    )


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@cache_once
def _is_cuda_platform() -> bool:
    from sglang.kernels.spec import PlatformInfo

    return PlatformInfo.detect().is_cuda


def can_argmax(logits: torch.Tensor) -> bool:
    """Whether `argmax` accepts ``logits``: CUDA, a supported dtype, rows of aligned vectors."""
    if not _is_cuda_platform():
        return False
    if logits.dtype not in _SUPPORTED_DTYPES or logits.ndim != 2:
        return False
    # Mirrors `kMaxVecBytes` in `argmax.cuh`
    vec_bytes = current_arch().cuda.max_vec_bytes
    vec_size = vec_bytes // logits.element_size()
    return (
        logits.size(1) % vec_size == 0
        and logits.stride(1) == 1
        and logits.stride(0) % vec_size == 0
        and logits.data_ptr() % vec_bytes == 0
    )


def argmax(
    logits: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    index_dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    if out is None:
        out = torch.empty(logits.shape[0], dtype=index_dtype, device=logits.device)
    _jit_argmax_module(logits.dtype).argmax(logits, out, workspace)
    return out
