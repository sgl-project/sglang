from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


@cache_once
def _jit_verify_penalties_module(dtype: torch.dtype) -> Module:
    assert dtype in _SUPPORTED_DTYPES, f"{dtype} not in {_SUPPORTED_DTYPES}"
    args = make_cpp_args(dtype)
    return load_jit(
        "sampling_verify_penalties",
        *args,
        cuda_files=["sampling/verify_penalties.cuh"],
        cuda_wrappers=[("apply_verify_penalties", f"apply_verify_penalties<{args}>")],
    )


def apply_verify_penalties(
    logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    next_child: Optional[torch.Tensor],
    next_sibling: Optional[torch.Tensor],
    batch_indices: torch.Tensor,
    penalty_indices: torch.Tensor,
    counts: torch.Tensor,
    lens: torch.Tensor,
    frequency: Optional[torch.Tensor],
    presence: Optional[torch.Tensor],
    repetition: Optional[torch.Tensor],
    min_new_tokens: Optional[torch.Tensor],
    stop_token_ids: Optional[torch.Tensor],
) -> None:
    """Apply each verify row's own sampling penalties to ``logits`` in place, bit for
    bit as non-speculative decoding would.

    ``logits`` holds ``width = draft_tokens.shape[1]`` rows per request; column 0
    of ``draft_tokens`` is the tree root. Request ``i`` sits at batch position
    ``batch_indices[i]`` with its history in row ``penalty_indices[i]`` of
    ``counts`` and ``lens``. A row counts that history plus the in-vocab tokens
    from the root (exclusive) to its node (inclusive). ``next_child`` and
    ``next_sibling`` (``[bs, width]``, -1 for none) give the tree; both None is a
    chain. Penalty tables are None when no request of the step uses that penalty.
    """
    _jit_verify_penalties_module(logits.dtype).apply_verify_penalties(
        logits,
        draft_tokens,
        next_child,
        next_sibling,
        batch_indices,
        penalty_indices,
        counts,
        lens,
        frequency,
        presence,
        repetition,
        min_new_tokens,
        stop_token_ids,
    )
