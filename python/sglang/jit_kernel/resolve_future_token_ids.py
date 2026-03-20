from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_resolve_future_token_ids_module(dtype: torch.dtype) -> Module:
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
    module = _jit_resolve_future_token_ids_module(input_ids.dtype)
    module.resolve_future_token_ids(input_ids, future_token_ids_map)
