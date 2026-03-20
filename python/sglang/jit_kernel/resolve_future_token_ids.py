from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_SUPPORTED_DTYPES = (torch.int32, torch.int64)


@cache_once
def _jit_resolve_future_token_ids_module(dtype: torch.dtype) -> Module:
    """Compile and cache the JIT module for a given dtype."""
    args = make_cpp_args(dtype)
    return load_jit(
        "resolve_future_token_ids",
        *args,
        cuda_files=["elementwise/resolve_future_token_ids.cuh"],
        cuda_wrappers=[
            (
                "resolve_future_token_ids",
                f"ResolveFutureTokenIds<{args}>::run",
            )
        ],
    )


def resolve_future_token_ids_cuda(
    input_ids: torch.Tensor, future_token_ids_map: torch.Tensor
) -> None:
    """Resolve future token IDs in-place on CUDA.

    For each negative value in input_ids, replaces it with
    future_token_ids_map[-value]. Non-negative values are unchanged.

    Supported dtypes: torch.int32, torch.int64.
    """
    if not input_ids.is_cuda:
        raise RuntimeError("input_ids must be a CUDA tensor")
    if input_ids.dtype not in _SUPPORTED_DTYPES:
        raise RuntimeError(
            f"Unsupported dtype {input_ids.dtype}. Supported: {_SUPPORTED_DTYPES}"
        )
    module = _jit_resolve_future_token_ids_module(input_ids.dtype)
    module.resolve_future_token_ids(input_ids, future_token_ids_map)
