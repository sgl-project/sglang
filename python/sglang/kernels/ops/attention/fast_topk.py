from __future__ import annotations

import os
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

_FAST_TOPK_SUPPORTED_K = (512, 2048)
_Q38C_TOPK_ZERO_START = os.environ.get("SGLANG_Q38C_TOPK_ZERO_START", "0") == "1"


@cache_once
def _jit_fast_topk_module(topk: int, zero_start: bool = False) -> Module:
    """Compile and cache the JIT fast top-k module for a given k."""
    # Checks on the compile key live here, not in `fast_topk`: `cache_once`
    # keys on topk and zero-start mode, so this runs once per specialisation.
    if topk not in _FAST_TOPK_SUPPORTED_K:
        raise RuntimeError(
            f"Unsupported topk {topk}. Supported: {_FAST_TOPK_SUPPORTED_K}"
        )
    args = make_cpp_args(topk, is_arch_support_pdl(), zero_start)
    return load_jit(
        "fast_topk",
        *args,
        cuda_files=["elementwise/fast_topk.cuh"],
        cuda_wrappers=[("fast_topk", f"FastTopKKernel<{args}>::run")],
    )


def fast_topk(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Per-row top-k selection over a fp32 score matrix.

    Row b selects the `topk` largest values in
    ``score[b, row_starts[b] : row_starts[b] + lengths[b]]`` and returns their
    indices relative to ``row_starts[b]``. Slots beyond ``lengths[b]`` are -1.
    Output order within a row is unspecified (atomic collection order).

    Parameters
    ----------
    score      : CUDA fp32 tensor [B, L]
    lengths    : CUDA int32 tensor [B]
    topk       : number of indices per row; 512 or 2048
    row_starts : optional CUDA int32 tensor [B]; defaults to zeros

    Returns
    -------
    CUDA int32 tensor [B, topk]
    """
    batch = score.shape[0]
    zero_start = _Q38C_TOPK_ZERO_START and row_starts is None
    if row_starts is None and not zero_start:
        row_starts = torch.zeros(batch, dtype=torch.int32, device=score.device)
    indices = score.new_empty((batch, topk), dtype=torch.int32)

    module = _jit_fast_topk_module(topk, zero_start)
    module.fast_topk(score, row_starts, indices, lengths)
    return indices
