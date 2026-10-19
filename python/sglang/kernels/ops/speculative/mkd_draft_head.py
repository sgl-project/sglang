from __future__ import annotations

from typing import TYPE_CHECKING

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
def _module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "mkd_draft_head",
        *args,
        cuda_files=["speculative/mkd_draft_head.cuh"],
        cuda_wrappers=[("run", f"MkdHead<{args}>::run")],
    )


def mkd_draft_head(
    x: torch.Tensor,
    weight: torch.Tensor,
    logits: torch.Tensor,
    part: torch.Tensor,
    reverse: bool = False,
) -> None:
    """Draft lm_head of one token, ``logits`` [1, N] bf16 = ``x`` [1, 4096] @ ``weight`` [N, 4096]^T with
    fp32 accumulation, and per CTA (``part.numel()`` CTAs) the packed maximum of its rows for mkd_xchg.
    ``reverse`` walks the rows from the end (same values; re-reads the most recently read rows first)."""
    _module().run(x, weight, logits, part, int(reverse))
