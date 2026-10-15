"""One-launch JIT CUDA versions of two draft preparation kernels.

``draft_cache_locs`` writes what ``cache_locs.assign_draft_cache_locs_contiguous``
writes for a topk-1 draft with positions; ``draft_extend_layout`` returns what
``eagle.prepare_draft_extend_layout`` returns. Both are integer-only, so the
outputs are identical; the C++ side validates the layout and reports an
unsupported one instead of launching, so callers keep the Triton path as the
fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_draft_prep_module() -> Module:
    return load_jit(
        "draft_prep",
        cuda_files=["speculative/draft_prep.cuh"],
        cuda_wrappers=[
            ("draft_cache_locs", "DraftPrep::cache_locs"),
            ("draft_extend_layout", "DraftPrep::extend_layout"),
        ],
    )


def draft_cache_locs(
    *,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    positions: torch.Tensor,
    mrope: Optional[torch.Tensor],
    cols: int,
) -> bool:
    """Fill ``out_cache_loc`` ([bs * cols]), ``positions`` and ``mrope``; False
    (nothing written) when the tensors are outside the supported layout."""
    module = _jit_draft_prep_module()
    status = module.draft_cache_locs(
        req_pool_indices, req_to_token, seq_lens, out_cache_loc, positions, mrope, cols
    )
    return status == 0


def draft_extend_layout(
    seq_lens: torch.Tensor, num_draft_tokens: int, with_mrope: bool
):
    """Same outputs as ``prepare_draft_extend_layout``, or None when the layout
    is not supported."""
    bs = seq_lens.shape[0]
    device = seq_lens.device
    prefix = torch.empty((bs,), dtype=torch.int32, device=device)
    extend = torch.empty_like(prefix)
    post = torch.empty((bs,), dtype=seq_lens.dtype, device=device)
    positions = torch.empty((bs * num_draft_tokens,), dtype=torch.int64, device=device)
    starts = torch.empty_like(prefix)
    mrope = (
        torch.empty((3, bs * num_draft_tokens), dtype=torch.int64, device=device)
        if with_mrope
        else None
    )
    module = _jit_draft_prep_module()
    status = module.draft_extend_layout(
        seq_lens, prefix, extend, post, starts, positions, mrope, num_draft_tokens
    )
    if status != 0:
        return None
    return prefix, extend, post, (positions, starts, mrope)


__all__ = ["draft_cache_locs", "draft_extend_layout"]
