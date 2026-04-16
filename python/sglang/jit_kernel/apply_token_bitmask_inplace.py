from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Union

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

BITS_PER_BLOCK = 32


@cache_once
def _jit_apply_token_bitmask_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "apply_token_bitmask_inplace",
        *args,
        cuda_files=["grammar/apply_token_bitmask_inplace.cuh"],
        cuda_wrappers=[
            (
                "apply_token_bitmask_inplace",
                f"apply_token_bitmask_inplace<{args}>",
            )
        ],
    )


def apply_token_bitmask_inplace_jit(
    logits: torch.Tensor,
    bitmask: torch.Tensor,
    indices: Optional[Union[List[int], torch.Tensor]] = None,
) -> None:
    """Apply a token bitmask to logits in-place using a JIT-compiled CUDA kernel.

    Masked positions (bitmask bit = 0) are set to -inf. This kernel provides
    vectorized memory access and early-exit optimizations over the Triton
    equivalent.

    Args:
        logits: 1D or 2D float/half/bfloat16 CUDA tensor.
        bitmask: 1D or 2D int32 CUDA tensor of packed bitmask.
        indices: Optional 1D int32 tensor or list specifying which logits
            rows to apply the mask to. If None, bitmask rows map 1:1 to
            logits rows.
    """
    if indices is not None:
        if isinstance(indices, list):
            indices = torch.tensor(indices, dtype=torch.int32, device=logits.device)
        else:
            indices = indices.to(dtype=torch.int32, device=logits.device)

    num_logits_rows = logits.size(0) if logits.dim() == 2 else 1

    use_indices = indices is not None and indices.numel() > 0
    if use_indices:
        num_rows = indices.size(0)
    else:
        num_rows = num_logits_rows

    dummy_indices = torch.empty(0, dtype=torch.int32, device=logits.device)
    indices_tensor = indices if use_indices else dummy_indices

    module = _jit_apply_token_bitmask_module(logits.dtype)
    module.apply_token_bitmask_inplace(
        logits,
        bitmask,
        indices_tensor,
        num_rows,
        1 if use_indices else 0,
    )
