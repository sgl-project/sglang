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

# Weight rows held in shared memory per CTA; one CTA per SM.
_MAX_SMEM_BYTES = 216 * 1024


# Rows of x per launch (draft steps at bs 1-2); wider batches keep the cuBLAS partial.
MAX_TOKENS = 2


@cache_once
def _module(half: int, width: int, out_rows: int, max_rows: int, tokens: int) -> Module:
    args = make_cpp_args(half, width, out_rows, max_rows, tokens, is_arch_support_pdl())
    return load_jit(
        "draft_fc_partial",
        *args,
        cuda_files=["speculative/draft_fc_partial.cuh"],
        cuda_wrappers=[("run", f"DraftFcPartial<{args}>::run")],
    )


def draft_fc_partial_rows(out_rows: int, width: int, num_sms: int) -> int | None:
    """Weight rows per CTA for the fused kernel, or None if they do not fit in shared memory."""
    rows = -(-out_rows // num_sms)
    static = MAX_TOKENS * width * 2
    return rows if rows * width * 2 + static <= _MAX_SMEM_BYTES else None


def draft_fc_partial(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    weight_slice: torch.Tensor,
    col_start: int,
    out: torch.Tensor,
    max_rows: int,
) -> torch.Tensor:
    """Gemma RMSNorm of the input half x [M, H] (bf16 output; M <= MAX_TOKENS), then this
    rank's fc slice: out [M, N] fp32 = norm(x)[:, col_start : col_start + W] @ weight_slice [N, W]^T."""
    _module(x.shape[-1], weight_slice.shape[-1], weight_slice.shape[0], max_rows, x.shape[0]).run(
        x, norm_weight, weight_slice, out, eps, col_start
    )
    return out
