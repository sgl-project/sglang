from __future__ import annotations

from typing import Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def _jit_module():
    return load_jit(
        "moe_finalize_fuse_shared",
        cuda_files=["moe/moe_finalize_fuse_shared.cu"],
        extra_dependencies=["cutlass"],
        header_only=False,
    )


def moe_finalize_fuse_shared(
    gemm2_out: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_weights: torch.Tensor,
    shared_output: Optional[torch.Tensor],
    top_k: int,
    enable_pdl: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``out`` lets the caller place the result, e.g. straight into symmetric memory."""
    assert gemm2_out.dtype == torch.bfloat16
    assert expert_weights.dtype in (torch.float32, torch.bfloat16)
    assert expanded_idx_to_permuted_idx.dtype == torch.int32
    assert gemm2_out.dim() == 2
    assert expert_weights.dim() == 2

    num_tokens, top_k_check = expert_weights.shape
    assert top_k_check == top_k
    hidden_dim = gemm2_out.shape[1]

    if shared_output is not None:
        assert shared_output.dtype == torch.bfloat16
        assert shared_output.dim() == 2
        assert shared_output.shape[0] == num_tokens
        hidden_dim = shared_output.shape[1]
        assert hidden_dim <= gemm2_out.shape[1]

    if out is None:
        out = gemm2_out.new_empty(num_tokens, hidden_dim)
    if shared_output is None:
        shared_output = gemm2_out.new_empty((0, 0), dtype=torch.bfloat16)

    _jit_module().moe_finalize_fuse_shared(
        out,
        gemm2_out,
        expanded_idx_to_permuted_idx,
        expert_weights,
        shared_output,
        int(top_k),
        bool(enable_pdl),
    )
    return out
