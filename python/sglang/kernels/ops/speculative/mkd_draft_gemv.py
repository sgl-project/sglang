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
def _module(k: int, rows: int, push: bool) -> Module:
    args = make_cpp_args(k, rows, is_arch_support_pdl(), push)
    return load_jit(
        "mkd_draft_gemv",
        *args,
        cuda_files=["speculative/mkd_draft_gemv.cuh"],
        cuda_wrappers=[("run", f"MkdGemv<{args}>::run")],
    )


def mkd_draft_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    hop_site: Optional[torch.Tensor] = None,
    hop_token: int = 0,
) -> torch.Tensor:
    """``out`` [1, N] bf16 = ``x`` [1, K] @ ``weight`` [N, K]^T with fp32 accumulation, one warp per row.
    hop_site: host int64 [7] from Q35Hop.site_fields(); the output (N = 4096) is then also pushed into that
    hop as row hop_token, and its all-reduce only consumes."""
    push = hop_site is not None
    _module(x.shape[-1], 8 if push else 4, push).run(
        x, weight, out, hop_site if push else x, hop_token
    )
    return out
