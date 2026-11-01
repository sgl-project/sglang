from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

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
def _max_vec_bytes(device_index: int) -> int:
    # Mirrors `kMaxVecBytes`: 256-bit vectors on Blackwell, 128-bit before
    return 32 if torch.cuda.get_device_capability(device_index)[0] >= 10 else 16


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
    vec_bytes = _max_vec_bytes(logits.device.index)
    vec_size = vec_bytes // logits.element_size()
    return (
        logits.size(1) % vec_size == 0
        and logits.stride(1) == 1
        and logits.stride(0) % vec_size == 0
        and logits.data_ptr() % vec_bytes == 0
    )


def create_argmax_workspace(num_rows: int, device: torch.device) -> torch.Tensor:
    """Zeroed scratch for batches past the built-in workspace (`kMaxBatchSize` in `argmax.cuh`).

    The kernel hands every slot back zeroed, so one buffer is reusable across calls.
    """
    return torch.zeros(num_rows, dtype=torch.int64, device=device)


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
